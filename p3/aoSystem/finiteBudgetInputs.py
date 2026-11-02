"""Forward-only input-band moments for optional finite residual accounting.

The view shares the native estimator, prior and controller responses. Only
forward atmospheric moments are partitioned; no reconstructor is rebuilt.
It is private to finiteErrorBudget and never replaces the prediction backend.
"""
from copy import copy

import numpy as np
from scipy import fft, ndimage

from .finiteAperture import _fold, _stationary_terms
from .finiteCovariance import _fourier_sum_many, residual_stream_moments


class _CommandCovariance:
    """Common covariance in command factors, avoiding another dense pupil array."""
    def __init__(self, basis, covariance):
        self.basis = basis
        self.product = basis @ covariance

    def diagonal(self):
        return np.einsum('ij,ij->i', self.product, self.basis)

    def __getitem__(self, key):
        rows, columns = key
        return self.product[rows] @ self.basis[columns].T


def _measurement_moments(b, crop, power):
    two_stage = b.native.twoStageEnabled
    qx, qy = b.qx[crop, crop], b.qy[crop, crop]
    values = np.zeros((5 if two_stage else 1, b.nc, b.nc, b.span, b.span))
    for layer, (a, h) in enumerate(zip(b.contraction, b.heights)):
        h1 = b.temporal[layer, crop, crop]
        if two_stage:
            wind = np.deg2rad(b.native.ao.atm.wDir[layer])
            frequency = -(qx*np.cos(wind)+qy*np.sin(wind))*b.native.ao.atm.wSpeed[layer]
            h2 = b.stage.gain*np.exp(-2j*np.pi*frequency*b.stage.delay/b.native.ao.rtc.holoop['rate'])
            multipliers = np.stack((abs(h1)**2, np.ones_like(h1), h1.conj(),
                                    h1*h2.conj(), abs(h1)**2*h2.conj()))
        else:
            multipliers = abs(h1[None])**2
        for guide in range(b.ng):
            for other in range(b.ng):
                shift = a*(b.offsets[guide]-b.offsets[other])+h*(b.guide[guide]-b.guide[other])
                for i, j in ((0, 0), (0, 1), (1, 1)):
                    density = power[layer]*b.gradients[layer, i, crop, crop]*b.gradients[layer, j, crop, crop].conj()
                    value = _fourier_sum_many(multipliers*density, b.dk,
                        shift-a*b.pitch*(b.ncell-1), a*b.pitch, b.span).real
                    values[:, 2*guide+i, 2*other+j] += value
                    if i != j:
                        values[:, 2*guide+j, 2*other+i] += value
    return [value.reshape(b.nc, b.nc, -1)[:, :, b.didx].transpose(0, 2, 1, 3).reshape(b.nc*b.nv, b.nc*b.nv)
            for value in values]


def _projected_moments(b, crop, power):
    projection = copy(b.projected_feedback)
    projection.covariance = np.zeros_like(projection.covariance)
    projection.cross = np.zeros_like(projection.cross)
    qx, qy = b.qx[crop, crop], b.qy[crop, crop]
    index = np.rint(b.cells/b.pitch+(b.ncell-1)/2).astype(int)
    for layer, (a, h) in enumerate(zip(b.contraction, b.heights)):
        z = projection.layers[layer][:, crop, crop]
        zr = z.reshape(3*b.nc, -1)
        projection.covariance += ((zr*power[layer].ravel()) @ zr.conj().T).real
        h1 = b.temporal[layer, crop, crop]
        wind = np.deg2rad(b.native.ao.atm.wDir[layer])
        frequency = (qx*np.cos(wind)+qy*np.sin(wind))*b.native.ao.atm.wSpeed[layer]
        h2 = (b.stage.gain*np.exp(2j*np.pi*frequency*b.stage.delay/b.native.ao.rtc.holoop['rate'])
              if b.native.twoStageEnabled else np.zeros_like(h1))
        for guide in range(b.ng):
            shift = a*b.offsets[guide]+h*b.guide[guide]
            for axis in range(2):
                channel = 2*guide+axis
                base = power[layer]*b.gradients[layer, axis, crop, crop]*z.conj()
                for stream, multiplier in enumerate((h1, h2, h1*h2)):
                    values = _fourier_sum_many(base*multiplier, b.dk,
                        shift-a*b.pitch*(b.ncell-1)/2, a*b.pitch, b.ncell).real
                    projection.cross[stream, channel*b.nv:(channel+1)*b.nv] += values[:, index[:, 0], index[:, 1]].T
    return projection


def band_view(b, sampling):
    """Return a CPU forward view containing only inputs inside the DM cutoff.

    Call after native residual preparation so optical bases and mean-slope
    feedback moments exist. Shared arrays are read-only; the view owns every
    modified forward moment. Optical quadrature and interpolation are unchanged.
    """
    cutoff = float(b.native.freq.kcMax_)
    half = min(b.count//2, int(np.ceil(cutoff/b.dk)))
    crop = slice(b.count//2-half, b.count//2+half)
    mask = (b.qx**2+b.qy**2 < cutoff**2 if b.native.ao.dms.AoArea == 'circle'
            else (abs(b.qx) < cutoff) & (abs(b.qy) < cutoff))
    low = copy(b)
    # A new pupil must not retain the previous reporting cache through this view.
    if hasattr(low, '_error_budget'):
        del low._error_budget
    low.raw = b.raw*mask[None]
    power = low.raw[:, crop, crop]*b.dk**2
    moments = _measurement_moments(b, crop, power)
    low.cov_raw = moments[0]
    if b.native.twoStageEnabled:
        c11, c00, c10, c12, c1_12 = moments
        low.caa, low.cbb, low.cab = residual_stream_moments(c11,
            b.stage.gain**2*c00, b.stage.gain**2*c11, c12, c1_12,
            b.stage.gain**2*c10, b.feedback)
        low.common_science_cross = b.operator @ low.cab
        low.device = None
    if b.native.removeMeanSlopes:
        low.projected_feedback = _projected_moments(b, crop, power)
        low.projected_feedback.b = low
    low.correction = _CommandCovariance(b.common_basis, b.weights@low.cov_raw@b.weights.T)
    low.source_terms = _stationary_terms(power.sum(0), b.kernels[:, crop, crop], sampling, sampling.n)
    low._source_covariance = None
    low.auxiliary = np.zeros_like(b.auxiliary)
    low.additional_covariance = None
    low.interpolation = []
    if b.native.twoStageEnabled:
        low.raw_interpolation = []
    for layer in range(len(b.heights)):
        targets = [(low.interpolation, b.temporal[layer, crop, crop].conj())]
        if b.native.twoStageEnabled:
            targets.append((low.raw_interpolation, 1.))
        for target, control in targets:
            values = np.empty((3, 2, b.kernel_grid, b.kernel_grid))
            for axis in range(2):
                density = power[layer]*b.gradients[layer, axis, crop, crop].conj()*control
                for i, multiplier in enumerate((1., b.kernels[0, crop, crop], b.kernels[1, crop, crop])):
                    covariance = fft.ifft2(_fold(density*multiplier, b.kernel_grid)).real*b.kernel_grid**2
                    values[i, axis] = ndimage.spline_filter(covariance, order=3, mode='grid-wrap')
            target.append(values)
    return low

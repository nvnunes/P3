"""Residual-driven directional mirrors within P3's alias-aware Fourier model.

The common and science commands share every atmospheric input. Their coherent
sum is squared before independent layer powers are added. This module does not
supply finite-aperture covariance, influence-function fitting or optical code.
"""
import numpy as nnp

from . import np
from . import FourierUtils


def applied_science_response(frequency, rate, gain, delay):
    """Direct command with a frame queue, for exp(2*pi*i*f*t) inputs."""
    return gain * np.exp(-2j*np.pi*frequency*delay/rate)


def coupled_command(common, directional, common_in_direction, h1, h2):
    """Combine command responses to the same input, retaining interference.

    Inputs broadcast over frequencies, science directions and physical layers.
    ``common_in_direction`` is the directional estimate's response to the
    common mirror, not an independent atmospheric sample.
    """
    return h1*common + h2*(directional-common_in_direction*h1*common)


class TwoStageFourier:
    """Fixed spatial MMSE rows with common POLC and residual-driven science DMs.

    Outputs are phase spectral densities in radian² metre² on P3's AO grid.
    The outer model owns unresolved turbulence, conversion to nm²/bin, static
    optics and PSFs. Direction batches bound query temporaries.
    """

    def __init__(self, model, settings):
        self.model = model
        self.settings = settings
        model._validate_tomographic_aliasing()
        if model.reconstructionModel != 'alias_aware':
            raise ValueError('Science correction currently requires ReconstructionModel=alias_aware')
        if nnp.any(nnp.asarray(model.ao.wfs.processing.noiseVar) != 0):
            raise ValueError('Science correction initially supports zero injected HO noise only')
        if model.ao.rtc.holoop['gain'] <= 0:
            raise ValueError('Science correction initially requires an active common POLC loop')
        if (not nnp.array_equal(model.atm_mod.heights, model.ao.atm.heights)
                or not nnp.allclose(model.atm_mod.weights, model.ao.atm.weights, rtol=1e-14, atol=0)):
            raise ValueError('Science correction requires all physical reconstruction layers')
        if settings.pitch < float(model.ao.dms.pitch[0]):
            raise ValueError('Science pitch must be >= common pitch on the current shared Fourier grid')
        self.kx, self.ky = model.freq.kxAO_, model.freq.kyAO_
        self.heights = model.ao.atm.heights*model.strechFactor
        self.common_row = (model.PbetaDM[0] @ model.W)[..., 0, :]
        self.dm_measurement = model._wfs_xy_measurement(
            self.kx, self.ky, [0.], [True])[..., 0]
        if model.ao.dms.AoArea == 'circle':
            support = np.hypot(self.kx, self.ky) <= 1/(2*settings.pitch)
        else:
            support = ((abs(self.kx) <= 1/(2*settings.pitch))
                       & (abs(self.ky) <= 1/(2*settings.pitch)))
        self.science_support = support
        shape = self.kx.shape + (model.ao.src.nSrc, self.common_row.shape[-1])
        self.science_rows = np.empty(shape, dtype=model.complex_dtype)
        hmod = model.atm_mod.heights*model.strechFactor_mod
        for source in range(model.ao.src.nSrc):
            angle = (self.kx*model.ao.src.direction[0, source]
                     + self.ky*model.ao.src.direction[1, source])
            target = np.exp(2j*np.pi*angle[..., None]*hmod)
            self.science_rows[..., source, :] = np.einsum(
                '...l,...lm->...m', target, model.Wtomo)*support[..., None]
        self.common_in_science = np.einsum(
            '...sm,...m->...s', self.science_rows, self.dm_measurement)

    def _prepare_wave(self, qx, qy):
        """Reuse source-independent measurements and timing within one query."""
        model = self.model
        measured = model._wfs_xy_measurement(
            qx, qy, self.heights, model.sensedLayers, exposure=True)
        common = np.einsum('...m,...ml->...l', self.common_row, measured)
        frequency = model._layer_temporal_frequencies(qx, qy)
        h1 = model._polc_response(frequency)[..., None, :]
        h2 = applied_science_response(frequency, model.ao.rtc.holoop['rate'],
                                     self.settings.gain, self.settings.delay)[..., None, :]
        return measured, common, h1, h2

    def command(self, qx, qy, batch, prepared=None):
        """Combined atmospheric command for one bounded set of science rows."""
        measured, common, h1, h2 = self._prepare_wave(qx, qy) if prepared is None else prepared
        directional = np.einsum('...sm,...ml->...sl',
                                self.science_rows[..., batch, :], measured)
        return coupled_command(common[..., None, :], directional,
                               self.common_in_science[..., batch, None], h1, h2)

    def main_psd(self, piston_filter=True, optical_pair=False):
        """Residual from science minus the *combined* common/science command."""
        model = self.model
        variance = np.zeros(self.kx.shape+(model.ao.src.nSrc,), dtype=model.dtype)
        optical = np.zeros_like(variance) if optical_pair else None
        pf = model.freq.pistonFilterAO_[..., None]
        raw = np.divide(model.Cphi, pf, out=np.zeros_like(model.Cphi), where=pf > 0)
        # Native covariance includes the piston-removed variance weight. Optics
        # must instead receive residual phase power before that weight.
        layer_power = model.Cphi if piston_filter else raw
        prepared = self._prepare_wave(self.kx, self.ky)
        for first in range(0, model.ao.src.nSrc, 16):
            batch = slice(first, min(first+16, model.ao.src.nSrc))
            direction = model.ao.src.direction[:, batch]
            phase = (self.kx[..., None]*direction[0] + self.ky[..., None]*direction[1])
            science = np.exp(2j*np.pi*phase[..., None]*self.heights)
            power = abs(science-self.command(self.kx, self.ky, batch, prepared))**2
            variance[..., batch] = np.sum(power*layer_power[..., None, :], axis=-1)
            if optical_pair:
                optical[..., batch] = np.sum(power*raw[..., None, :], axis=-1)
        variance *= model.freq.mskInAO_[..., None]
        if optical_pair:
            optical *= model.freq.mskInAO_[..., None]
            return variance, optical
        return variance

    def alias_psd(self, piston_filter=True, optical_pair=False):
        """Actual aliases through the same coupled rows and physical timing."""
        model = self.model
        pitch = model.ao.wfs.optics[0].dsub
        variance = np.zeros(self.kx.shape+(model.ao.src.nSrc,), dtype=model.dtype)
        optical = np.zeros_like(variance) if optical_pair else None
        for m in model._alias_shift_grid():
            for n in model._alias_shift_grid():
                if m == 0 and n == 0:
                    continue
                qx, qy = self.kx-m/pitch, self.ky-n/pitch
                kernel = ((qx*qx+qy*qy+1/model.ao.atm.L0**2)**(-11/6)
                          *model.ao.atm.r0**(-5/3)*.0229)
                pf = FourierUtils.pistonFilter(model.ao.tel.D, np.hypot(qx, qy), dtype=model.dtype)
                factor = pf if piston_filter else ((qx != 0) | (qy != 0))
                prepared = self._prepare_wave(qx, qy)
                for first in range(0, model.ao.src.nSrc, 16):
                    batch = slice(first, min(first+16, model.ao.src.nSrc))
                    power = np.sum(abs(self.command(qx, qy, batch, prepared))**2*model.ao.atm.weights,
                                   axis=-1)
                    variance[..., batch] += (kernel*factor)[..., None]*power
                    if optical_pair:
                        optical[..., batch] += (kernel*((qx != 0) | (qy != 0)))[..., None]*power
        variance *= model.freq.mskInAO_[..., None]
        if optical_pair:
            optical *= model.freq.mskInAO_[..., None]
            return variance, optical
        return variance

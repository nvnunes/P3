"""Finite-measurement MOAO composed with the current finite-aperture backend.

Shared temporal moments and symmetric residual updates retain all correlations.
Reconstruction and source physics remain in double precision. No old engine,
low-rank truncation or independent-stage variance sum is consumed here.
"""
import time
import numpy as np
from scipy import fft, ndimage
from scipy.linalg import cho_solve

from .finiteAperture import (FiniteResidualCovariance,
                             _fold, _stationary_terms)

from .finiteActuator import FiniteActuatorReconstructor
from .mirrorSurface import GaussianSurface
from .finiteCovariance import (
    _fourier_sum_many, CoupledFiniteResidual, residual_stream_moments,
)


class FiniteTwoStageReconstructor(FiniteActuatorReconstructor):
    """Current finite common estimator plus directional residual correction.

    Common reconstruction, valid measurements, source spectrum and optics are
    inherited. Both mirrors fit their physical actuator surfaces. All
    atmospheric layers contribute to every changed command and cross term.
    """
    def __init__(self, native):
        started = time.perf_counter()
        super().__init__(native)
        self.stage = native.scienceMirror
        if self.stage.pitch < float(native.ao.dms.pitch[0]):
            raise ValueError('Finite science pitch must be >= common pitch')
        if native.ao.rtc.holoop['gain'] <= 0:
            raise ValueError('Finite two-stage MOAO requires an active common loop')
        self.science_surface = GaussianSurface(native.my_data_map['DM_science'],
            self.cells, self.pitch, native.ao.tel.pupil, native.ao.resolve_path,
            diameter=float(native.ao.tel.D), actuators=self.stage.actuators,
            mirror_pitch=self.stage.pitch)
        self._prepare_science_prior(self.science_surface)
        self.feedback = self._common_feedback()
        self._joint_moments()
        self._coupled_sampling = None
        self.timings['two_stage_construction_seconds'] = time.perf_counter()-started

    def _measurement_moments(self):
        """Four additional raw temporal moments in shared finite coordinates."""
        started = time.perf_counter(); span = self.span
        values = np.zeros((4,self.nc,self.nc,span,span))
        rate = self.native.ao.rtc.holoop['rate']
        for layer,(a,h) in enumerate(zip(self.contraction,self.heights)):
            wind = np.deg2rad(self.native.ao.atm.wDir[layer])
            f = -(self.qx*np.cos(wind)+self.qy*np.sin(wind))*self.native.ao.atm.wSpeed[layer]
            h1 = self.temporal[layer]
            h2 = self.stage.gain*np.exp(-2j*np.pi*f*self.stage.delay/rate)
            multipliers = np.stack((np.ones_like(h1),h1.conj(),h1*h2.conj(),abs(h1)**2*h2.conj()))
            for guide in range(self.ng):
                for other in range(self.ng):
                    shift = a*(self.offsets[guide]-self.offsets[other])+h*(self.guide[guide]-self.guide[other])
                    for i,j in ((0,0),(0,1),(1,1)):
                        density = self.raw[layer]*self.dk**2*self.gradients[layer,i]*self.gradients[layer,j].conj()
                        v = _fourier_sum_many(multipliers*density,self.dk,
                            shift-a*self.pitch*(self.ncell-1),a*self.pitch,span).real
                        values[:,2*guide+i,2*other+j] += v
                        if i!=j: values[:,2*guide+j,2*other+i] += v
        matrices = [v.reshape(self.nc,self.nc,-1)[:,:,self.didx].transpose(0,2,1,3).reshape(self.nc*self.nv,self.nc*self.nv) for v in values]
        self.timings['shared_temporal_moments_seconds'] = time.perf_counter()-started
        return matrices

    def _joint_moments(self):
        c00,c10,c12,c1_12 = self._measurement_moments()
        self.caa,self.cbb,self.cab = residual_stream_moments(self.cov_raw,
            self.stage.gain**2*c00,self.stage.gain**2*self.cov_raw,c12,c1_12,
            self.stage.gain**2*c10,self.feedback)

    def prepare_optics(self,sampling,dense_source=True):
        super().prepare_optics(sampling,dense_source)
        if sampling.n!=sampling.native_n:
            raise ValueError('Finite two-stage validation requires native pupil quadrature')
        if self._coupled_sampling == self._sampling: return
        started = time.perf_counter()
        self.common_science_cross = self.operator@self.cab
        self.science_basis=self.fitted_surface(self.science_surface,sampling)
        self.raw_interpolation = []
        for layer in range(len(self.heights)):
            values = np.empty((3,2,self.kernel_grid,self.kernel_grid))
            for j in range(2):
                density = self.raw[layer]*self.dk**2*self.gradients[layer,j].conj()
                for i,multiplier in enumerate((1,self.kernels[0],self.kernels[1])):
                    c = fft.ifft2(_fold(density*multiplier,self.kernel_grid)).real*self.kernel_grid**2
                    values[i,j] = ndimage.spline_filter(c,order=3,mode='grid-wrap')
            self.raw_interpolation.append(values)
        self._coupled_sampling = self._sampling
        self.timings['coupled_optical_preparation_seconds'] = time.perf_counter()-started

    def science_operator(self, sampling, field):
        """Fit the directional phase estimate to the science actuator surface."""
        target = self.science_surface.fit @ self._prior_cross(field)
        rows = cho_solve(self.factor, target.T, check_finite=False).T
        return self.science_basis @ rows

    def _query_crosses(self,sampling,field):
        beta = self.native.ao.src.direction[:,field]
        cross = np.zeros((3,len(sampling.xs),self.nc*self.nv))
        values = np.empty((2,len(sampling.xs),self.nv))
        rate = self.native.ao.rtc.holoop['rate']
        for layer,(a,h) in enumerate(zip(self.contraction,self.heights)):
            wind = np.deg2rad(self.native.ao.atm.wDir[layer])
            shift = self.native.ao.atm.wSpeed[layer]*self.stage.delay/rate*np.array([np.cos(wind),np.sin(wind)])
            for guide in range(self.ng):
                origin = h*(beta-self.guide[guide])-a*self.offsets[guide]
                for stream,kernel,offset,gain in (
                    (0,self.interpolation[layer],origin,1.),
                    (1,self.raw_interpolation[layer],origin-shift,self.stage.gain),
                    (2,self.interpolation[layer],origin-shift,self.stage.gain)):
                    self._science_query.evaluate(kernel[0],offset,a,values)
                    base = np.array([offset[1]-a*self.cells[:,1],offset[0]-a*self.cells[:,0]])*self.kernel_grid/self.period
                    for axis in range(2):
                        fit = np.array([ndimage.map_coordinates(kernel[i,axis],base,order=3,mode='grid-wrap',prefilter=False) for i in (1,2)])
                        values[axis] -= sampling.ramps.T@fit
                        cross[stream,:,(2*guide+axis)*self.nv:(2*guide+axis+1)*self.nv] += gain*values[axis]
        return cross

    def residual_blocks(self,sampling,field):
        self.prepare_optics(sampling,dense_source=False)
        started = time.perf_counter(); before = time.perf_counter()
        crosses = self._query_crosses(sampling,field)
        query_seconds = time.perf_counter()-before; before = time.perf_counter()
        target=self.science_surface.fit@self._prior_cross(field)
        rows=cho_solve(self.factor,target.T,check_finite=False).T
        basis=self.science_basis
        operator=basis@rows
        operator_seconds = time.perf_counter()-before
        independent = _stationary_terms(self._independent_power(field),self.aux_kernels,sampling,sampling.n)
        common,science,product = crosses
        update = .5*(operator@self.cbb)+self.common_science_cross-science+product@self.feedback.T
        update=update@rows.T
        operator=basis
        common=common@self.weights.T
        common_operator=self.common_basis
        base = FiniteResidualCovariance(sampling,self.source_terms,self.correction,
            self.additional_covariance,common,common_operator,independent)
        residual = CoupledFiniteResidual(base,operator,update)
        profile=dict(science_measurement_seconds=query_seconds,
            science_operator_seconds=operator_seconds,residual_covariance_seconds=time.perf_counter()-started)
        return self._project_measurement_means(residual,sampling,field,profile)

    def residual_covariance(self,sampling,field):
        residual,profile = self.residual_blocks(sampling,field)
        covariance = np.empty((len(residual.diagonal),len(residual.diagonal)))
        for first,block in residual.blocks():
            covariance[first:first+len(block),first:] = block
        i,j = np.triu_indices(len(covariance),1);covariance[j,i]=covariance[i,j]
        return covariance,profile

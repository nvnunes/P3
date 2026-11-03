"""Gaussian command targets within the accepted finite-measurement prior."""
import time
import numpy as np
from scipy import fft,ndimage
from scipy.linalg import cho_solve

from .finiteAperture import FiniteApertureReconstructor,_ScienceMeasurementQuery,_fold
from .mirrorSurface import GaussianSurface


class FiniteActuatorReconstructor(FiniteApertureReconstructor):
    """Common physical Gaussian surface within finite reconstruction.

    A Gaussian estimator fits the estimated raw phase on WFS reconstruction
    nodes, then projects actuator tilt. Its covariance uses the same source
    band, atmospheric prior, measurement factor and relative floor as the
    finite Fourier estimator, without imposing an ideal mirror cutoff first.
    """
    def _reconstruct(self):
        started=time.perf_counter()
        self.common_surface=GaussianSurface(self.native.my_data_map['DM'],self.cells,self.pitch,
            self.native.ao.tel.pupil,self.native.ao.resolve_path,
            diameter=float(self.native.ao.tel.D), actuators=float(self.native.ao.dms.nActu1D[0]),
            mirror_pitch=float(self.native.ao.dms.pitch[0]))
        # Physical actuator targets use the configured phase average itself.
        # The scalar gain inferred from the Fourier projector is inapplicable.
        self.gain=1.
        self.phase=np.array([(np.exp(2j*np.pi*h*(self.q@self.targets.T))@self.target_weights)
                             for h in self.heights]).T
        self._prepare_prior_query(self.common_surface)
        target=self.common_surface.fit@self._prior_cross(average=True)
        self.weights=cho_solve(self.factor,target.T,check_finite=False).T
        self.normal_equation_error=float(np.linalg.norm(self.weights@self.cov_regular-target)/np.linalg.norm(target))
        self.timings['reconstruction_seconds']=time.perf_counter()-started

    def _prepare_prior_query(self,surface):
        if hasattr(self,'prior_query'):
            np.testing.assert_allclose(surface.fit_points,self.phase_fit_points,rtol=0,atol=1e-12)
            return
        self.phase_fit_points=surface.fit_points
        self.prior_grid=2**int(np.ceil(np.log2(max(self.count,4*self.native_n))))
        self.prior_query=_ScienceMeasurementQuery(surface.fit_points,self.cells,self.prior_grid,self.period)
        self.prior_interpolation=None
        self._zero_height_prior={}

    def _prior_kernel(self,layer,multiplier=1):
        values=[]
        for axis in range(2):
            density=self.prior[layer]*self.dk**2*self.gradients[layer,axis].conj()*multiplier
            covariance=fft.ifft2(_fold(density,self.prior_grid)).real*self.prior_grid**2
            values.append(ndimage.spline_filter(covariance,order=3,mode='grid-wrap'))
        return np.array(values)

    def _prepare_science_prior(self,surface):
        self._prepare_prior_query(surface)
        if self.prior_interpolation is None:
            self.prior_interpolation=[self._prior_kernel(layer) for layer in range(len(self.heights))]

    def _prior_cross(self,field=None,average=False):
        cross=np.zeros((len(self.phase_fit_points),self.nc*self.nv))
        # Bound grouped output to 32 MiB, or one guide if it exceeds that.
        # The averaged common estimator keeps its original summation order.
        group=1 if average else max(1,min(self.ng,(32*1024**2)//(16*len(self.phase_fit_points)*self.nv)))
        values=np.empty((2,group,len(self.phase_fit_points),self.nv))
        for layer,(a,h) in enumerate(zip(self.contraction,self.heights)):
            if not average and h==0 and layer in self._zero_height_prior:
                cross+=self._zero_height_prior[layer]
                continue
            if average:
                phase=np.zeros_like(self.qx,dtype=complex)
                for beta,weight in zip(self.targets,self.target_weights):
                    phase+=weight*np.exp(2j*np.pi*h*(self.qx*beta[0]+self.qy*beta[1]))
                kernel=self._prior_kernel(layer,self.gain*phase)
                beta=np.zeros(2)
            else:
                kernel=self.prior_interpolation[layer]
                beta=self.native.ao.src.direction[:,field]
            target=np.zeros_like(cross) if not average and h==0 else cross
            for first in range(0,self.ng,group):
                last=min(first+group,self.ng)
                origins=h*(beta-self.guide[first:last])-a*self.offsets[first:last]
                if last-first==1:
                    self.prior_query.evaluate(kernel,origins[0],a,values[:,0])
                else:
                    self.prior_query.evaluate_group(kernel,origins,a,values[:,:last-first])
                for local,guide in enumerate(range(first,last)):
                    for axis in range(2):
                        cross_channel=slice((2*guide+axis)*self.nv,(2*guide+axis+1)*self.nv)
                        target[:,cross_channel]+=values[axis,local]
            if not average and h==0:
                self._zero_height_prior[layer]=target
                cross+=target
        return cross

    def _prepare_common_operator(self,sampling):
        basis=self.fitted_surface(self.common_surface,sampling)
        self.common_basis=basis
        self.operator=basis@self.weights

    @staticmethod
    def fitted_surface(surface,sampling):
        """Physical applied response, followed by the unchanged final pupil fit."""
        # The command projection was already applied in surface.fit. This
        # second fit is the optical removal of residual pupil tilt, not a
        # second actuator-command projection.
        basis=surface.basis(sampling.xy)
        fine=sampling.fit.shape[1]
        axis=(-sampling.native_side/2+np.arange(fine)*sampling.native_side/fine)*sampling.period/sampling.native_n
        g=np.exp(np.log(surface.coupling)*((axis[:,None]-surface.axis)/surface.pitch)**2)
        moments=np.array([(g.T@fit.T@g)[surface.ix,surface.iy] for fit in sampling.fit])
        return basis-sampling.ramps.T@moments

    def _common_feedback(self):
        """Raw finite-lenslet response of the common actuator command."""
        started = time.perf_counter()
        feedback = self.common_surface.slopes(self.cells,self.pitch,self.offsets)@self.weights
        self.timings['common_feedback_seconds'] = time.perf_counter()-started
        return feedback

    def _project_measurement_means(self,residual,sampling,field,profile,science_rows=None):
        if not self.native.removeMeanSlopes:return residual,profile
        if not hasattr(self,'projected_feedback'):
            from .finiteMeanSlopes import ProjectedFeedback
            if not hasattr(self,'feedback'):self.feedback=self._common_feedback()
            self.projected_feedback=ProjectedFeedback(self)
            self.timings['measurement_projection_seconds']=self.projected_feedback.seconds
        started=time.perf_counter()
        result=self.projected_feedback.wrap(residual,sampling,field,science_rows=science_rows)
        profile=dict(profile,measurement_projection_field_seconds=time.perf_counter()-started)
        return result,profile

    def residual_blocks(self,sampling,field):
        residual,profile=super().residual_blocks(sampling,field)
        return self._project_measurement_means(residual,sampling,field,profile)

    def residual_covariance(self,sampling,field):
        """Dense diagnostic view of the same blocked residual calculation."""
        residual,profile=self.residual_blocks(sampling,field)
        covariance=np.empty((len(residual.diagonal),len(residual.diagonal)))
        for first,block in residual.blocks():covariance[first:first+len(block),first:]=block
        i,j=np.triu_indices(len(covariance),1);covariance[j,i]=covariance[i,j]
        return covariance,profile

"""Exact low-rank sensor-mean/controller correction for finite reconstruction.

This adds all changed command and atmospheric cross covariances to the existing
residual. No temporal series or eigenmode truncation is used.
"""
import time
import numpy as np
from scipy import fft

from .finiteCovariance import _fourier_sum_many, CoupledFiniteResidual


def _fold_many(spectra,n):
    """The same physical Fourier folding with a bounded leading stream batch."""
    count=spectra.shape[-1]
    index=(np.arange(count)-count//2)%n
    result=np.zeros((len(spectra),n,n),complex)
    for first in range(0,count,n):
        last=min(first+n,count)
        for other in range(0,count,n):
            end=min(other+n,count)
            result[:,index[None,other:end],index[first:last,None]]+=spectra[:,first:last,other:end]
    return result


def row_response(backend, rows, layer):
    """Evaluate fixed slope rows on every represented atmospheric plane wave."""
    b=backend; a=b.contraction[layer]; h=b.heights[layer]
    indices=np.rint(b.cells/b.pitch+(b.ncell-1)/2).astype(int)
    result=np.zeros((len(rows),b.count,b.count),complex)
    for guide in range(b.ng):
        phase=np.exp(2j*np.pi*(h*(b.qx*b.guide[guide,0]+b.qy*b.guide[guide,1])
            +a*(b.qx*b.offsets[guide,0]+b.qy*b.offsets[guide,1])))
        for axis in range(2):
            channel=2*guide+axis
            grid=np.zeros((len(rows),b.ncell,b.ncell))
            grid[:,indices[:,0],indices[:,1]]=rows[:,channel*b.nv:(channel+1)*b.nv]
            transformed=_fourier_sum_many(grid,b.pitch,
                np.repeat(-b.count//2*b.dk*a,2),b.dk*a,b.count)
            centre=(b.ncell//2-(b.ncell-1)/2)*b.pitch*a
            transformed*=np.exp(2j*np.pi*centre*(b.qx+b.qy))
            result+=transformed*(phase*b.gradients[layer,axis])[None]
    return result


class ProjectedFeedback:
    """Shared moments of the exact change caused by per-WFS slope projection.

    The eight (two per WFS) removed slope modes produce three small command
    streams. Frequency-dependent POLC feedback is solved without truncation;
    field-specific contractions preserve every atmospheric/command cross term.
    Instances belong to one finite backend and its immutable configuration.
    """
    def __init__(self,backend):
        started=time.perf_counter(); b=backend
        self.b=b
        stage=b.native.scienceMirror if b.native.twoStageEnabled else None
        self.science_gain=stage.gain if stage is not None else 0.
        self.science_delay=stage.delay if stage is not None else 0
        self.means=np.kron(np.eye(b.nc),np.ones((b.nv,1))/np.sqrt(b.nv))
        self.feedback_means=b.feedback@self.means
        small=self.means.T@self.feedback_means
        rows=np.concatenate((self.means.T,self.means.T@b.feedback))
        self.covariance=np.zeros((3*b.nc,3*b.nc))
        self.cross=np.zeros((3,b.nc*b.nv,3*b.nc))
        self.layers=[]
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            response=row_response(b,rows,layer)
            h1=b.temporal[layer]
            direction=np.deg2rad(b.native.ao.atm.wDir[layer])
            frequency=(b.qx*np.cos(direction)+b.qy*np.sin(direction))*b.native.ao.atm.wSpeed[layer]
            h2=self.science_gain*np.exp(2j*np.pi*frequency*self.science_delay/b.native.ao.rtc.holoop['rate'])
            rhs=-response[:b.nc]+h1*response[b.nc:]
            t=np.empty_like(rhs)
            flat=rhs.reshape(b.nc,-1);target=t.reshape(b.nc,-1);control=h1.ravel()
            for first in range(0,len(control),4096):
                sl=slice(first,min(first+4096,len(control)))
                matrix=np.eye(b.nc)[None]-control[sl,None,None]*small
                target[:,sl]=(np.linalg.solve(matrix,flat[:,sl].T[...,None])[...,0]*control[sl,None]).T
            z=np.concatenate((t,h2*rhs,h2*t))
            power=b.raw[layer]*b.dk**2
            self.layers.append(z)
            zr=z.reshape(3*b.nc,-1);pr=power.ravel()
            self.covariance+=((zr*pr)@zr.conj().T).real
            for guide in range(b.ng):
                shift=a*b.offsets[guide]+h*b.guide[guide]
                start=shift-a*b.pitch*(b.ncell-1)/2
                transform=None
                # Small target grids favor exact rectangular float64 products.
                if b.count>=4*b.ncell:
                    positions=start[:,None]+np.arange(b.ncell)[None]*a*b.pitch
                    spatial_frequency=(np.arange(b.count)-b.count//2)*b.dk
                    transform=np.exp(2j*np.pi*positions[:,:,None]*spatial_frequency[None,None])
                for axis in range(2):
                    channel=2*guide+axis
                    base=power*b.gradients[layer,axis]*z.conj()
                    for stream,multiplier in enumerate((h1,h2,h1*h2)):
                        if transform is None:
                            values=_fourier_sum_many(base*multiplier,b.dk,
                                start,a*b.pitch,b.ncell).real
                        else:
                            values=(transform[0]@(base*multiplier)@transform[1].T).real
                        index=np.rint(b.cells/b.pitch+(b.ncell-1)/2).astype(int)
                        self.cross[stream,channel*b.nv:(channel+1)*b.nv]+=values[:,index[:,0],index[:,1]].T
        self.projected_feedback=self.feedback_means-self.means@(self.means.T@self.feedback_means)
        self.science_cross=self.cross[1]-b.feedback@self.cross[2]
        self._cross_owner=self.cross
        self._optics_sampling=None
        self._optics_operator=None
        self.seconds=time.perf_counter()-started

    def wrap(self,residual,sampling,field,science_rows=None):
        """Return a field-owned symmetric update of the original covariance."""
        b=self.b
        # Input-band reporting views copy this object but own new cross moments.
        changed=self._cross_owner is not self.cross
        if changed:
            self.science_cross=self.cross[1]-b.feedback@self.cross[2]
            self._cross_owner=self.cross
        if changed or self._optics_sampling is not sampling or self._optics_operator is not b.operator:
            self._common_means=b.operator@self.means
            self._common_cross=b.operator@self.cross[0]
            self._optics_sampling=sampling
            self._optics_operator=b.operator
        if self.science_gain:
            if science_rows is None:
                from scipy.linalg import cho_solve
                target=b.science_surface.fit@b._prior_cross(field)
                science_rows=cho_solve(b.factor,target.T,check_finite=False).T
            basis=b.science_basis
            science_means=basis@(science_rows@self.means)
            science_feedback=basis@(science_rows@self.projected_feedback)
            old_cross=self._common_cross+basis@(science_rows@self.science_cross)
        else:
            science_means=np.zeros_like(self._common_means)
            science_feedback=np.zeros_like(self._common_means)
            old_cross=self._common_cross
        operator=np.concatenate((self._common_means,science_means,-science_feedback),axis=1)
        device=getattr(b,'device',None)
        if (device is not None and hasattr(device,'projected_atmosphere')
                and getattr(device,'complex_feedback_supported',True)):
            atmospheric=device.projected_atmosphere(self,sampling,field)
            update=old_cross-atmospheric+.5*operator@self.covariance
            return CoupledFiniteResidual(residual,operator,update)
        atmospheric=np.zeros_like(operator)
        beta=b.native.ao.src.direction[:,field]
        ys=(sampling.ys-sampling.n//2)%sampling.n
        xs=(sampling.xs-sampling.n//2)%sampling.n
        kernels=b.kernels.reshape(len(b.kernels),-1)
        for layer,h in enumerate(b.heights):
            phase=np.exp(2j*np.pi*h*(b.qx*beta[0]+b.qy*beta[1]))
            weighted=b.raw[layer]*b.dk**2*phase
            for first in range(0,len(self.layers[layer]),6):
                last=min(first+6,len(self.layers[layer]))
                density=weighted[None]*self.layers[layer][first:last].conj()
                values=fft.ifft2(_fold_many(density,sampling.n),axes=(-2,-1)).real*sampling.n**2
                moments=(kernels@density.reshape(len(density),-1).T).real
                atmospheric[:,first:last]+=values[:,ys,xs].T-sampling.ramps.T@moments
        update=old_cross-atmospheric+.5*operator@self.covariance
        return CoupledFiniteResidual(residual,operator,update)

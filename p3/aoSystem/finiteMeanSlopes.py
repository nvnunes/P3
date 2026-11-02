"""Exact low-rank sensor-mean/controller correction for finite reconstruction.

This adds all changed command and atmospheric cross covariances to the existing
residual. No temporal series or eigenmode truncation is used.
"""
import time
import numpy as np
from scipy import fft

from .finiteAperture import _fold
from .finiteCovariance import _fourier_sum_many, CoupledFiniteResidual


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

    The two removed slope modes per WFS alter the common command stream. Frequency-dependent POLC feedback is solved without truncation;
    field-specific contractions preserve every atmospheric/command cross term.
    Instances belong to one finite backend and its immutable configuration.
    """
    def __init__(self,backend):
        started=time.perf_counter(); b=backend
        self.b=b
        self.means=np.kron(np.eye(b.nc),np.ones((b.nv,1))/np.sqrt(b.nv))
        self.feedback_means=b.feedback@self.means
        small=self.means.T@self.feedback_means
        rows=np.concatenate((self.means.T,self.means.T@b.feedback))
        self.covariance=np.zeros((b.nc,b.nc))
        self.cross=np.zeros((b.nc*b.nv,b.nc))
        self.layers=[]
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            response=row_response(b,rows,layer)
            h1=b.temporal[layer]
            rhs=-response[:b.nc]+h1*response[b.nc:]
            t=np.empty_like(rhs)
            flat=rhs.reshape(b.nc,-1);target=t.reshape(b.nc,-1);control=h1.ravel()
            for first in range(0,len(control),4096):
                sl=slice(first,min(first+4096,len(control)))
                matrix=np.eye(b.nc)[None]-control[sl,None,None]*small
                target[:,sl]=(np.linalg.solve(matrix,flat[:,sl].T[...,None])[...,0]*control[sl,None]).T
            z=t
            power=b.raw[layer]*b.dk**2
            self.layers.append(z)
            zr=z.reshape(b.nc,-1);pr=power.ravel()
            self.covariance+=((zr*pr)@zr.conj().T).real
            for guide in range(b.ng):
                shift=a*b.offsets[guide]+h*b.guide[guide]
                for axis in range(2):
                    channel=2*guide+axis
                    base=power*b.gradients[layer,axis]*z.conj()
                    values=_fourier_sum_many(base*h1,b.dk,
                        shift-a*b.pitch*(b.ncell-1)/2,a*b.pitch,b.ncell).real
                    index=np.rint(b.cells/b.pitch+(b.ncell-1)/2).astype(int)
                    self.cross[channel*b.nv:(channel+1)*b.nv]+=values[:,index[:,0],index[:,1]].T
        self.seconds=time.perf_counter()-started

    def wrap(self,residual,sampling,field):
        """Return a field-owned symmetric update of the original covariance."""
        b=self.b
        operator=b.operator@self.means
        old_cross=b.operator@self.cross
        atmospheric=np.zeros_like(operator)
        beta=b.native.ao.src.direction[:,field]
        ys=(sampling.ys-sampling.n//2)%sampling.n
        xs=(sampling.xs-sampling.n//2)%sampling.n
        for layer,h in enumerate(b.heights):
            phase=np.exp(2j*np.pi*h*(b.qx*beta[0]+b.qy*beta[1]))
            power=b.raw[layer]*b.dk**2
            for col,z in enumerate(self.layers[layer]):
                density=power*phase*z.conj()
                values=fft.ifft2(_fold(density,sampling.n)).real*sampling.n**2
                moments=np.sum(b.kernels*density[None],axis=(1,2)).real
                atmospheric[:,col]+=values[ys,xs]-sampling.ramps.T@moments
        update=old_cross-atmospheric+.5*operator@self.covariance
        return CoupledFiniteResidual(residual,operator,update)

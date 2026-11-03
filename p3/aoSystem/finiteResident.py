"""Device-owned complete covariance tiles with a bounded shared background cache."""
import hashlib
import time
import numpy as np
from copy import copy
from scipy import fft

from .finiteCovariance import CoupledFiniteResidual
from .finiteAperture import FiniteResidualCovariance


def _background_digest(base):
    digest=hashlib.blake2b(digest_size=16)
    for array in base.source_terms:
        values=np.ascontiguousarray(array)
        digest.update(str((values.shape,values.dtype.str)).encode())
        digest.update(values.view(np.uint8))
    return digest.digest()


class _ResidentBackground:
    def __init__(self,owner,base,digest):
        self.sampling=base.sampling
        self.correction=base.correction
        self.additional=base.additional
        self.digest=digest
        # A common covariance offset cancels exactly from every structure function.
        # Remove the large incoming piston level before single-precision storage.
        self.optical_offset=float(base.source_terms[0][0,0])
        shared=copy(base)
        shared.lags,shared.points,shared.tilt=base.source_terms
        count=len(base.diagonal)
        dtype=getattr(owner,'host_covariance_dtype',np.float64)
        host=np.empty((count,count),dtype=dtype)
        intensity=base.sampling.intensity
        quadratic=0.
        for first in range(0,count,256):
            last=min(first+256,count)
            block=shared.background_block(first,last)
            host[first:last,first:]=block-self.optical_offset
            host[last:,first:last]=block[:,last-first:].T-self.optical_offset
            upper=np.arange(first,count)[None]>=np.arange(first,last)[:,None]
            weighted=block*intensity[first:last,None]*intensity[None,first:]
            quadratic+=2*weighted[upper].sum()-np.sum(np.diag(block[:,:last-first])*intensity[first:last]**2)
        self.value=owner.upload(host)
        self.indices=owner.upload(np.arange(count),integer=True)
        self.quadratic=float(quadratic)
        image=np.zeros((base.sampling.n,base.sampling.n))
        image[base.sampling.ys,base.sampling.xs]=intensity
        self.intensity_lags=fft.ifft2(abs(fft.fft2(image))**2).real
        self.ramp_mean=base.sampling.ramps@intensity
        self.ys=owner.upload(base.sampling.ys,integer=True)
        self.xs=owner.upload(base.sampling.xs,integer=True)
        self.ramps=owner.upload(base.sampling.ramps)

    def matches(self,base,digest):
        return (self.sampling is base.sampling and self.correction is base.correction
                and self.additional is base.additional and self.digest==digest)


class ResidentCovariance:
    """One field's full covariance, retaining every symmetric correction update.

    P3 owns the covariance and shared device allocation. Optical integration
    belongs to the caller. The usual host block API remains available on the
    original residual for diagnostics and existing consumers.
    """
    def __init__(self,owner,residual):
        started=time.perf_counter()
        self.owner=owner
        self.residual=residual
        self.terms=[]
        base=residual
        while isinstance(base,CoupledFiniteResidual):
            if base.device is not None:
                if base.device.owner is not owner:
                    raise ValueError('Covariance factors belong to a different device owner')
                self.terms.append(base.device)
            else:
                self.terms.append((owner.upload(base.operator),owner.upload(base.update)))
            base=base.base
        if not isinstance(base,FiniteResidualCovariance):
            raise TypeError('Resident covariance requires the native finite residual')
        # Two-stage device fields already include both common cross terms.
        self.base_cross = not any(getattr(term, 'includes_base_cross', False) for term in self.terms)
        if self.base_cross:
            operator=(owner.common_operator if base.operator is getattr(owner.backend, 'common_basis', None)
                      else owner.upload(base.operator))
            self.terms.append((operator, -owner.upload(base.cross)))
        digest=_background_digest(base)
        cached=owner._resident_background
        if cached is None or not cached.matches(base,digest):
            cached=_ResidentBackground(owner,base,digest)
            owner._resident_background=cached
            self.background_prepared=True
        else:
            self.background_prepared=False
        self.background=cached
        self.sampling=base.sampling
        self.diagonal=owner.upload(residual.diagonal)
        lags,points,tilt=base.independent_terms
        offset=float(lags[0,0])
        self.optical_offset=cached.optical_offset+offset
        self.optical_diagonal=owner.upload(residual.diagonal-self.optical_offset)
        self.independent=tuple(owner.upload(value) for value in (lags-offset,points,tilt))
        self.independent_quadratic=float(np.sum(cached.intensity_lags*lags)
            -2*cached.ramp_mean@(points@self.sampling.intensity)
            +cached.ramp_mean@tilt@cached.ramp_mean)
        self.preparation_seconds=time.perf_counter()-started

    def block_array(self,first,last):
        """Return covariance in the original physical gauge for diagnostics."""
        value=self.optical_block_array(first,last)+self.optical_offset
        index=self.background.indices[:last-first]
        value[index,index]=self.diagonal[first:last]
        return value

    def optical_block_array(self,first,last):
        """Return rebased covariance; only pair differences may consume this view."""
        if not 0<=first<last<=len(self.residual.diagonal):
            raise ValueError('Invalid covariance tile bounds')
        value=self.owner.copy(self.background.value[first:last,first:])
        background=self.background
        lags,points,tilt=self.independent
        n=self.sampling.n
        dy=(background.ys[first:last,None]-background.ys[None,first:])%n
        dx=(background.xs[first:last,None]-background.xs[None,first:])%n
        value+=lags[dy,dx]
        ramps=background.ramps
        value-=ramps[:,first:last].T@points[:,first:]
        value-=points[:,first:last].T@ramps[:,first:]
        value+=(ramps[:,first:last].T@tilt)@ramps[:,first:]
        for term in self.terms:
            if isinstance(term,tuple):
                operator,update=term
                value+=update[first:last]@operator[first:].T+operator[first:last]@update[first:].T
            else:
                value+=term.block_array(first,last)
        index=self.background.indices[:last-first]
        value[index,index]=self.optical_diagonal[first:last]
        return value

    def variance(self):
        """Pupil-weighted variance from complete factors, without dense downloads."""
        intensity=self.sampling.intensity
        quadratic=self.background.quadratic+self.independent_quadratic
        base=self.residual
        while isinstance(base,CoupledFiniteResidual):
            if base.device is None:
                quadratic+=2*np.dot(intensity@base.update,intensity@base.operator)
            else:
                device=base.device
                weight=self.owner.upload(intensity)
                extra=2*((weight@device.update)*(weight@device.operator)).sum()
                extra-=2*((weight@device.common)*(weight@self.owner.common_operator)).sum()
                quadratic+=float(self.owner.to_host(extra))
            base=base.base
        if self.base_cross:
            quadratic-=2*np.dot(intensity@base.cross, intensity@base.operator)
        return float(intensity@self.residual.diagonal-quadratic)

"""Batched Fourier queries and symmetric finite-covariance updates.

Shared by single-stage sensor projection and coupled science correction.
"""
import numpy as np
from scipy import signal


class CommandCovariance:
    """Complete pupil covariance in exact mirror-command factors.

    ``basis`` is shared read-only with the backend. Only ``basis @ covariance``
    is allocated here; blocks and the diagonal retain every command component.
    This is an algebraic factorization, with no rank or covariance truncation.
    """
    def __init__(self,basis,covariance):
        self.basis=basis
        self.product=basis@covariance
        self.shape=(len(basis),len(basis))

    @property
    def nbytes(self):
        """Owned factor bytes, excluding the shared optical basis."""
        return self.product.nbytes

    def diagonal(self):
        return np.einsum('ij,ij->i',self.product,self.basis)

    def __getitem__(self,key):
        rows,columns=key
        return self.product[rows]@self.basis[columns].T


def _fourier_sum_many(spectra, dk, start, step, size):
    """Finite Fourier sum; exact rectangular products for small output grids."""
    count = spectra.shape[-1]
    if count >= 4*size:
        # Exact finite sum at a small output grid; avoid padded CZT arrays.
        frequency=(np.arange(count)-count//2)*dk
        positions=np.asarray(start)[:,None]+np.arange(size)[None]*step
        transform=np.exp(2j*np.pi*positions[:,:,None]*frequency[None,None])
        return (transform[0]@spectra)@transform[1].T
    j = np.arange(count)
    z = spectra*np.exp(2j*np.pi*j*dk*start[0])[None, :, None]
    z = z*np.exp(2j*np.pi*j*dk*start[1])[None, None, :]
    transform = signal.CZT(count, m=size, w=np.exp(2j*np.pi*dk*step))
    z = transform(transform(z, axis=-2), axis=-1)
    x = start[0]+np.arange(size)*step
    y = start[1]+np.arange(size)*step
    return z*np.exp(-2j*np.pi*(count//2)*dk*x)[None, :, None]*np.exp(-2j*np.pi*(count//2)*dk*y)[None, None, :]


def residual_stream_moments(c11, c22, c1212, c12, c1_12, c2_12, feedback):
    """Joint moments of a=h1*y and b=h2*y-L*h1*h2*y, all real coordinates."""
    cross = c12-c1_12@feedback.T
    residual = c22-c2_12@feedback.T-feedback@c2_12.T
    residual += (feedback@c1212)@feedback.T
    return c11, (residual+residual.T)*.5, cross


class CoupledFiniteResidual:
    """Complete residual with a symmetric covariance update and bounded storage."""
    def __init__(self, base, operator, update, device=None):
        self.base, self.operator, self.update, self.device = base, operator, update, device
        extra = 2*np.einsum('ij,ij->i', update, operator) if device is None else device.diagonal()
        self.diagonal = base.diagonal+extra

    def blocks(self, rows=None):
        if self.device is None:
            for first, covariance in self.base.blocks(rows):
                last = first+len(covariance)
                covariance += self.update[first:last]@self.operator[first:].T
                covariance += self.operator[first:last]@self.update[first:].T
                index = np.arange(last-first)
                covariance[index,index] = self.diagonal[first:last]
                yield first,covariance
        else:
            if rows is None: rows = max(1,min(512,(64*1024**2)//(8*len(self.diagonal))))
            if not isinstance(rows,int) or rows <= 0:
                raise ValueError('Covariance block rows must be positive')
            for first in range(0,len(self.diagonal),rows):
                last = min(first+rows,len(self.diagonal))
                covariance = self.base.background_block(first,last)
                covariance += self.device.block(first,last)
                index = np.arange(last-first)
                covariance[index,index] = self.diagonal[first:last]
                yield first,covariance

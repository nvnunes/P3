"""Batched Fourier queries and symmetric finite-covariance updates.

Shared by single-stage sensor projection and coupled science correction.
"""
import numpy as np
from scipy import signal


def _fourier_sum_many(spectra, dk, start, step, size):
    """One existing finite Fourier sum with a leading moment batch."""
    count = spectra.shape[-1]
    j = np.arange(count)
    z = spectra*np.exp(2j*np.pi*j*dk*start[0])[None, :, None]
    z = z*np.exp(2j*np.pi*j*dk*start[1])[None, None, :]
    transform = signal.CZT(count, m=size, w=np.exp(2j*np.pi*dk*step))
    z = transform(transform(z, axis=-2), axis=-1)
    x = start[0]+np.arange(size)*step
    y = start[1]+np.arange(size)*step
    return z*np.exp(-2j*np.pi*(count//2)*dk*x)[None, :, None]*np.exp(-2j*np.pi*(count//2)*dk*y)[None, None, :]


class CoupledFiniteResidual:
    """Complete residual with a field-owned symmetric covariance update."""
    def __init__(self, base, operator, update):
        self.base, self.operator, self.update = base, operator, update
        self.diagonal = base.diagonal+2*np.einsum('ij,ij->i',update,operator)

    def blocks(self, rows=None):
        for first, covariance in self.base.blocks(rows):
            last = first+len(covariance)
            covariance += self.update[first:last]@self.operator[first:].T
            covariance += self.operator[first:last]@self.update[first:].T
            index = np.arange(last-first)
            covariance[index,index] = self.diagonal[first:last]
            yield first,covariance

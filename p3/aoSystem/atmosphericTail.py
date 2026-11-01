"""Unresolved von Karman source power on a finite optical sampling grid.

The reconstructor's spectral support is unchanged. Positive Fourier shells
outside the represented source are folded onto the optical grid. The remaining
continuous tail supplies a white covariance closure; doubling the shell grid
must converge in total absolute spectral power before the result is accepted.
No amplitude is calibrated against a reference PSF.
"""

import numpy as np
from scipy import fft, special


_COEFFICIENT = ((24*special.gamma(6/5)/5)**(5/6)
                *special.gamma(11/6)**2/(2*np.pi**(11/3)))


class AtmosphericTail:
    """Source-only completion, in phase radians at the supplied Fried parameter.

    ``source_count`` and ``optical_count`` share Fourier spacing ``dk`` in
    cycles/metre. ``symmetric_source`` selects an open symmetric source square
    (finite aperture); otherwise the source is the usual half-open FFT square.
    The unshifted ``power`` contains per-mode phase variance, not PSD density.
    ``tolerance`` bounds successive completions in absolute L1 phase power.
    Tilt moments, when requested, use original physical frequencies before
    folding. The final white closure is projected on the original fit lattice.
    """

    def __init__(self, r0, outer_scale, dk, source_count, optical_count,
                 symmetric_source=False, tolerance=1e-5):
        values = (r0, dk, tolerance)
        if (not all(np.isfinite(v) and v > 0 for v in values)
                or np.isnan(outer_scale) or outer_scale <= 0):
            raise ValueError('Atmospheric tail requires positive finite physical scales and tolerance')
        if any(int(n) != n or n < 2 or n % 2 for n in (source_count, optical_count)):
            raise ValueError('Atmospheric tail requires positive even source and optical sizes')
        self.dk = float(dk)
        self.source_count = int(source_count)
        self.n = int(optical_count)
        self.symmetric_source = bool(symmetric_source)
        self.coefficient = _COEFFICIENT*float(r0)**(-5/3)
        self.outer_scale = float(outer_scale)
        count = self.n*max(4, 2**int(np.ceil(np.log2(self.source_count/self.n+1))))
        previous = None
        for _ in range(7):
            power = self._folded_power(count)
            remainder = self._outside_square((count/2-.5)*self.dk)
            power += remainder/self.n**2
            if previous is not None:
                error = float(np.sum(abs(power-previous)))
                if error < tolerance and remainder < 1e-3:
                    break
            previous = power
            count *= 2
        else:
            raise ValueError('Atmospheric source tail did not converge on the bounded shell grid')
        self.count = count
        self.power = power
        self.remainder = float(remainder)
        self.convergence_error = error

    def _outside_square(self, cutoff):
        nodes, weights = np.polynomial.legendre.leggauss(32)
        theta = (nodes+1)*np.pi/8
        radius = cutoff/np.cos(theta)
        return float(8*np.pi/8*np.sum(weights*self.coefficient*.6
            *(radius**2+self.outer_scale**-2)**(-5/6)))

    def _strips(self, count):
        modes = np.arange(count)-count//2
        q = modes*self.dk
        half = self.source_count//2
        for first in range(0, count, 128):
            last = min(first+128, count)
            x = modes[first:last]
            radius = q[first:last,None]**2+q[None]**2+self.outer_scale**-2
            # The source DC is always inside the excluded band, including L0=inf.
            radius[radius == 0] = 1
            power = self.coefficient*radius**(-11/6)*self.dk**2
            if self.symmetric_source:
                inside_x = abs(x) < half
                inside_y = abs(modes) < half
            else:
                inside_x = (x >= -half) & (x < half)
                inside_y = (modes >= -half) & (modes < half)
            power[inside_x[:,None] & inside_y[None]] = 0
            power[:,0] = 0
            if first == 0:
                power[0] = 0
            indices = ((modes[None] % self.n)*self.n+(x[:,None] % self.n)).ravel()
            yield first, last, power, indices

    def _folded_power(self, count):
        folded = np.zeros(self.n**2)
        for _, _, power, indices in self._strips(count):
            folded += np.bincount(indices, power.ravel(), minlength=self.n**2)
        return folded.reshape(self.n,self.n)

    def covariance(self):
        """Return the real periodic lag covariance on the optical grid."""
        return fft.ifft2(self.power).real*self.n**2

    def fitted_moments(self, sampling):
        """Return lag, phase/fit and fit/fit covariance for finite-pupil optics.

        The sampling protocol supplies original-pupil fit coefficients, physical
        pupil span and optical pixel indices. Only 128 source rows persist at a
        time. This calculation is shared by all field directions.
        """
        if sampling.n != self.n or not np.isclose(sampling.period*self.dk, 1):
            raise ValueError('Atmospheric tail and pupil must share optical grid and period')
        fine = sampling.fit.shape[1]
        axis = (-sampling.native_side/2+np.arange(fine)*sampling.native_side/fine)*sampling.period/sampling.native_n
        q = (np.arange(self.count)-self.count//2)*self.dk
        waves = np.exp(2j*np.pi*np.outer(q,axis))
        cross = np.zeros((2,self.n**2),complex)
        tilt = np.zeros((2,2))
        for first,last,power,indices in self._strips(self.count):
            kernels = np.asarray([waves[first:last]@fit.T@waves.T for fit in sampling.fit])
            for i in range(2):
                values = power*kernels[i].conj()
                cross[i] += np.bincount(indices,values.real.ravel(),minlength=self.n**2)
                cross[i] += 1j*np.bincount(indices,values.imag.ravel(),minlength=self.n**2)
            tilt += np.einsum('xy,ixy,jxy->ij',power,kernels,kernels.conj(),optimize=True).real
        cross = fft.ifft2(cross.reshape(2,self.n,self.n),axes=(-2,-1)).real*self.n**2
        y,x = (sampling.ys-self.n//2)%self.n,(sampling.xs-self.n//2)%self.n
        points = cross[:,y,x]
        # White closure on the original fit lattice. A science pixel has a
        # nonzero cross covariance only when it coincides with a fit sample.
        coordinates = (sampling.xy*sampling.native_n/sampling.period
            +sampling.native_side/2)*fine/sampling.native_side
        nearest = np.rint(coordinates).astype(int)
        coincident = np.all(abs(coordinates-nearest)<1e-10,axis=1)
        coincident &= np.all((nearest>=0)&(nearest<fine),axis=1)
        points[:,coincident] += self.remainder*sampling.fit[:,
            nearest[coincident,1],nearest[coincident,0]]
        tilt += self.remainder*np.einsum('iyx,jyx->ij',sampling.fit,sampling.fit)
        return self.covariance(),points,tilt

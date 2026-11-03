"""Finite square-cell sensing and physical-cone Gaussian reconstruction.

The prior includes the configured alias band and its full correlated x/y
measurements. Forward covariance retains raw incoming phase uncertainty and
every science/correction cross term. This CPU candidate replaces atmospheric
PSD optics; a diagonal spectrum is not a representation of its residual.
"""

import hashlib
import json
import time

import numpy as np
from scipy import fft, ndimage, signal
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from .FourierUtils import pistonFilter
from .atmosphericTail import AtmosphericTail
from .finiteConfiguration import configuration


def input_signature(native):
    """Identify all configuration and realized physical operator inputs."""
    from .mirrorSurface import geometry_signature
    state = dict(surface_geometry=[geometry_signature(native.my_data_map['DM'],native.ao.resolve_path)] +
        ([geometry_signature(native.my_data_map['DM_science'],native.ao.resolve_path)] if native.twoStageEnabled else []),
        covariance_backend=native.finiteApertureBackend, configuration=native.my_data_map, controller=native.ao.rtc.holoop,
        numerical_floor=native.tomoRelRegFloor, exposure=native.exposureTime)
    h = hashlib.sha256(json.dumps(state, sort_keys=True,
        default=lambda x: np.asarray(x).tolist()).encode())
    for values in (native.ao.tel.pupil, native.ao.atm.heights, native.ao.atm.weights,
                   native.ao.atm.wSpeed, native.ao.atm.wDir, native.Popt,
                   native.freq.kxAO_, native.freq.kyAO_, native.gs.height,
                   native.gs.direction, native.ao.src.direction, native.ao.src.wvl,
                   native.ao.atm.r0, native.ao.atm.L0, native.ao.tel.D,
                   native.ao.dms.nActu1D, native.ao.dms.pitch,
                   [native.ao.tel.extraErrorNm, native.ao.tel.extraErrorExp,
                    native.ao.tel.extraErrorMin, native.ao.tel.extraErrorMax],
                   native.strechFactor, native.ao.dms.opt_dir, native.ao.dms.opt_weights,
                   [w.dsub for w in native.ao.wfs.optics],
                   [native.freq.nOtf, native.freq.resAO, native.freq.kcMax_,
                    native.freq.nTimes, native.freq.PSDstep, native.freq.dk_]):
        a = np.ascontiguousarray(values)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.view(np.uint8))
    return h.hexdigest()


def _fourier_sum(spectrum, dk, start, step, size):
    count = len(spectrum); j = np.arange(count)
    z = spectrum*np.exp(2j*np.pi*j*dk*start[0])[:, None]*np.exp(2j*np.pi*j*dk*start[1])[None]
    for axis in (0, 1):
        z = signal.CZT(count, m=size, w=np.exp(2j*np.pi*dk*step))(z, axis=axis)
    x = start[0]+np.arange(size)*step; y = start[1]+np.arange(size)*step
    return z*np.exp(-2j*np.pi*(count//2)*dk*x)[:, None]*np.exp(-2j*np.pi*(count//2)*dk*y)[None]


def _fold(spectrum, n):
    count = len(spectrum); index = (np.arange(count)-count//2) % n
    result = np.zeros((n, n), complex)
    np.add.at(result.ravel(), (index[None]*n+index[:, None]).ravel(), spectrum.ravel())
    return result


def _stationary_terms(power, kernels, sampling, size):
    c = fft.ifft2(_fold(power, size)).real*size**2
    cross = np.array([fft.ifft2(_fold(power*k.conj(), size)).real*size**2 for k in kernels])
    tilt = np.einsum('xy,ixy,jxy->ij', power, kernels, kernels.conj()).real
    y, x = (sampling.ys-sampling.n//2) % size, (sampling.xs-sampling.n//2) % size
    points = cross[:, y, x]
    return c, points, tilt


def _stationary(power, kernels, sampling, size):
    c, points, tilt = _stationary_terms(power, kernels, sampling, size)
    return _stationary_matrix(c, points, tilt, sampling)


def _stationary_matrix(c, points, tilt, sampling):
    result = sampling.covariance_from_lags(c)
    result -= sampling.ramps.T@points
    result -= points.T@sampling.ramps
    result += sampling.ramps.T@tilt@sampling.ramps
    return result


def _periodic_cubic_matrix(coordinates, size):
    """Evaluate existing cubic B-spline coefficients with grid-wrap boundaries."""
    coordinates = np.remainder(coordinates.ravel(), size)
    start = np.floor(coordinates).astype(np.intp)
    t = coordinates-start
    weights = np.stack(((1-t)**3, 3*t**3-6*t*t+4,
        -3*t**3+3*t*t+3*t+1, t**3), axis=1)/6
    columns = (start[:, None]+np.arange(-1, 3)) % size
    return csr_matrix((weights.ravel(), columns.ravel(),
        np.arange(0, 4*len(start)+1, 4)), shape=(len(start), size))


class _LayerInterpolation:
    """Separately owned phase and float64 fitted-tilt covariance coefficients.

    A device owner may release ``phase`` after pinning the complete coefficient
    set. Fit moments always stay on the host; CPU and cache-evicting paths retain
    phase too. Separate allocations make releasing phase reclaim actual storage.
    """
    def __init__(self,grid):
        self.phase=np.empty((2,grid,grid))
        self.fit=np.empty((2,2,grid,grid))

    def host_phase(self):
        if self.phase is None:
            raise RuntimeError('Phase coefficients belong to the pinned device cache; rebuild the backend for CPU execution')
        return self.phase


class _ScienceMeasurementQuery:
    """Separable cubic queries at pupil-minus-contracted-lenslet positions.

    This geometry belongs to one optical-cache key. Only coordinate axes and
    output indices persist; direction-dependent stencils and intermediates do
    not. Pupil rows are blocked to budget 32 MiB for the two dense interpolation
    passes, including their transpose copy (or one row if it exceeds that).
    The caller owns the reusable double-precision output for both slope axes.
    """

    def __init__(self, xy, cells, grid, period):
        self.x, px = np.unique(xy[:, 0], return_inverse=True)
        self.y, py = np.unique(xy[:, 1], return_inverse=True)
        self.cx, ix = np.unique(cells[:, 0], return_inverse=True)
        self.cy, iy = np.unique(cells[:, 1], return_inverse=True)
        self.grid, self.scale = grid, grid/period
        self.grouped_query_stats=dict(calls=0,requested_y_products=0,computed_y_products=0)
        nx = len(self.x)*len(self.cx)
        rows = max(1, (32*1024**2)//(8*len(self.cy)*(2*grid+nx)))
        self.blocks = []
        for first in range(0, len(self.y), rows):
            last = min(first+rows, len(self.y))
            points = np.flatnonzero((py >= first) & (py < last))
            ny = (last-first)*len(self.cy)
            indices = (px[points, None]*len(self.cx)+ix[None])*ny
            indices += (py[points, None]-first)*len(self.cy)+iy[None]
            self.blocks.append((self.y[first:last], points, indices))

    def evaluate(self, kernels, origin, contraction, output):
        """Fill both slope outputs from the unchanged 2-D spline coefficients."""
        x = (self.x[:, None]+(origin[0]-contraction*self.cx[None]))*self.scale
        mx = _periodic_cubic_matrix(x, self.grid)
        for y, points, indices in self.blocks:
            coordinates = (y[:, None]+(origin[1]-contraction*self.cy[None]))*self.scale
            my = _periodic_cubic_matrix(coordinates, self.grid)
            for axis in range(2):
                intermediate = my@kernels[axis]
                values = mx@intermediate.T
                output[axis, points] = values.ravel()[indices]

    def evaluate_group(self,kernels,origins,contraction,output):
        """Reuse exactly identical y operators across a bounded guide group.

        Output is ``(slope axis, guide, pupil point, lenslet)``. Grouping compares
        complete sparse coefficients; close but unequal operators stay separate.
        No directions or covariance weights are rounded or averaged.
        """
        matrices=[]
        for origin in origins:
            x=(self.x[:,None]+(origin[0]-contraction*self.cx[None]))*self.scale
            matrices.append(_periodic_cubic_matrix(x,self.grid))
        stats=self.grouped_query_stats
        stats['calls']+=1
        for y,points,indices in self.blocks:
            groups={}
            for guide,origin in enumerate(origins):
                coordinates=(y[:,None]+(origin[1]-contraction*self.cy[None]))*self.scale
                matrix=_periodic_cubic_matrix(coordinates,self.grid)
                key=(matrix.indices.tobytes(),matrix.data.tobytes())
                groups.setdefault(key,(matrix,[]))[1].append(guide)
            stats['requested_y_products']+=len(kernels)*len(origins)
            stats['computed_y_products']+=len(kernels)*len(groups)
            for matrix,guides in groups.values():
                for axis in range(len(kernels)):
                    intermediate=np.ascontiguousarray((matrix@kernels[axis]).T)
                    for guide in guides:
                        values=matrices[guide]@intermediate
                        output[axis,guide,points]=values.ravel()[indices]
                    del intermediate


class FiniteResidualCovariance:
    """One field's complete residual, evaluated in bounded upper row strips.

    Arrays from the reconstruction/optics cache are held by reference and
    never changed. Science--measurement cross covariance and independent
    stationary moments belong to this field. The residual retains incoming,
    correction and both science--correction terms. It is not a low-rank or
    diagonal approximation. ``diagonal`` and every block use radians squared.
    """

    def __init__(self, sampling, source, correction, additional, cross, operator, independent):
        self.sampling = sampling
        self.correction = correction; self.additional = additional
        self.cross = cross; self.operator = operator
        # Pupil fitting is linear. Sum stationary moments before projection,
        # retaining the complete incoming source without a dense source matrix.
        self.source_terms=source
        self.independent_terms=independent
        self.lags, self.points, self.tilt = (a+b for a,b in zip(source,independent))
        ramps = sampling.ramps
        diagonal = np.full(len(sampling.xs), self.lags[0,0])
        diagonal -= np.sum(ramps*self.points,axis=0)
        diagonal -= np.sum(self.points*ramps,axis=0)
        diagonal += np.sum((ramps.T@self.tilt)*ramps.T,axis=1)
        if additional is not None: diagonal += np.diag(additional)
        diagonal += correction.diagonal()
        sc = np.einsum('ij,ij->i',cross,operator)
        diagonal -= sc; diagonal -= sc
        self.diagonal = diagonal

    def background_block(self, first, last):
        """Stationary/source and common-correction block before phase cross terms."""
        ramps = self.sampling.ramps
        r, c = slice(first,last), slice(first,None)
        independent = self.sampling.covariance_from_lags(self.lags,r,c)
        independent -= ramps[:,r].T@self.points[:,c]
        independent -= self.points[:,r].T@ramps[:,c]
        independent += (ramps[:,r].T@self.tilt)@ramps[:,c]
        covariance = independent
        if self.additional is not None: covariance += self.additional[r,c]
        covariance += self.correction[r,c]
        return covariance

    def blocks(self, rows=None):
        """Yield ``(first, covariance[first:last, first:])`` without full intermediates.

        Rows bound temporary storage; all columns at or above each row are
        evaluated. The small lower triangle within a strip is also supplied.
        New reconstruction/optical inputs require a new residual object.
        """
        if rows is None:
            # At most 64 MiB per dense strip, with a 512-row BLAS tile.
            rows = max(1,min(512,(64*1024**2)//(8*len(self.diagonal))))
        if not isinstance(rows,int) or rows <= 0:
            raise ValueError('Covariance block rows must be a positive integer')
        ramps = self.sampling.ramps
        for first in range(0,len(self.diagonal),rows):
            last = min(first+rows,len(self.diagonal)); r = slice(first,last); c = slice(first,None)
            covariance = self.background_block(first,last)
            covariance -= self.cross[r]@self.operator[c].T
            covariance -= self.operator[r]@self.cross[c].T
            # Dot-product ordering differs between diagonal and blocked GEMM.
            # One shared diagonal enforces the exact zero-separation identity.
            index = np.arange(last-first)
            covariance[index,index] = self.diagonal[r]
            yield first,covariance


def validate_native(native):
    """Reject combinations outside this candidate's tested physical contract."""
    if not isinstance(native.freq.kxAO_, np.ndarray):
        raise ValueError('Finite-aperture reconstruction requires CPU P3 arrays; set P3_DISABLE_GPU=1 before importing P3. Use finiteApertureBackend for optional GPU covariance products.')
    sensor = native.my_data_map.get('sensor_HO', {})
    native._validate_tomographic_aliasing()
    if native.temporalModel != 'polc' or native.exposureTime != 0:
        raise ValueError('ReconstructionModel=finite_aperture requires its model-owned POLC and instantaneous measurements')
    if np.any(native.ao.wfs.processing.noiseVar):
        raise ValueError('ReconstructionModel=finite_aperture has not been validated with nonzero HO noise')
    if sensor.get('TipTiltRemoval') != 'pupil_fit' or not native.computeOpticalPSD:
        raise ValueError("ReconstructionModel=finite_aperture requires TipTiltRemoval='pupil_fit' and optical power")
    if len(np.unique(native.ao.src.wvl)) != 1:
        raise ValueError('ReconstructionModel=finite_aperture currently requires one science wavelength')
    heights = np.asarray(native.gs.height)
    if not np.all(np.isfinite(heights)) or np.any(heights <= max(native.ao.atm.heights)) or not np.allclose(heights, heights[0]):
        raise ValueError('ReconstructionModel=finite_aperture requires equal-height LGSs above every layer')
    if len(native.atm_mod.heights) != len(native.ao.atm.heights):
        raise ValueError('ReconstructionModel=finite_aperture currently requires all physical reconstruction layers')
    if native.ao.windPsdFile != 0:
        raise ValueError('ReconstructionModel=finite_aperture has not been validated with vibration')


class FiniteApertureReconstructor:
    """Exact global estimator with reusable physical residual operators.

    The supplied native model owns configuration, spectra and controller
    policy. Geometry is sampled from its physical pupil and lenslet pitch.
    Sampling is an explicit optics protocol: coordinates in metres, original
    ramp-fit moments, pupil amplitudes and ``covariance_from_lags`` for periodic
    lag sampling at pupil pairs. No optics package is imported here.
    Cache lifetime is this object; callers compare ``signature`` before reuse.
    """

    def __init__(self, native):
        started = time.perf_counter(); validate_native(native)
        self.native = native
        self.regularization = configuration(native.my_data_map, native.reconstructionModel)
        self.signature = input_signature(native)
        self.timings = {}
        self.pitch = float(native.ao.wfs.optics[0].dsub)
        self.native_n = int(native.freq.nOtf)
        self.native_side = int(2*np.round(float(native.ao.tel.D)/2*self.native_n*native.freq.PSDstep))
        dk = float(native.freq.kxAO_[1, 0]-native.freq.kxAO_[0, 0])
        self.period = 1/dk
        self.nbase = round(1/(self.pitch*dk))
        if not np.isclose(self.nbase*self.pitch, self.period, rtol=1e-10):
            raise ValueError('Finite-aperture Fourier period must contain an integral lenslet lattice')
        self.dk = 1/(self.nbase*self.pitch)
        # Internal integration coverage, not an independently selectable model.
        self.order = max(3, int(native.freq.nTimes))
        self.count = self.nbase*(2*self.order+1)
        axis = (np.arange(self.count)-self.count//2)*self.dk
        self.qx, self.qy = np.meshgrid(axis, axis, indexing='ij')
        spectrum = native.ao.atm.spectrum(np.hypot(self.qx, self.qy))
        spectrum[0] = 0; spectrum[:, 0] = 0; spectrum[self.count//2, self.count//2] = 0
        self.raw = np.asarray(native.ao.atm.weights)[:, None, None]*spectrum
        self.prior = self.raw.copy()
        self.heights = np.asarray(native.ao.atm.heights)
        self.contraction = 1/np.asarray(native.strechFactor)
        self.guide = np.asarray(native.gs.direction.T)
        self.ng = len(self.guide); self.nc = 2*self.ng
        self.offsets = np.asarray(native.my_data_map['sensor_HO'].get('GridOffsets', [[0., 0.]]*self.ng))*self.pitch
        ncell = round(float(native.ao.tel.D)/self.pitch)
        if not np.isclose(ncell*self.pitch, native.ao.tel.D, rtol=1e-10):
            raise ValueError('Finite-aperture lenslet grid must span the configured telescope diameter')
        cell_axis = (np.arange(ncell)-(ncell-1)/2)*self.pitch
        x, y = np.meshgrid(cell_axis, cell_axis, indexing='ij')
        pupil = np.asarray(native.ao.tel.pupil)
        coordinates = np.array([(y/native.ao.tel.D+.5)*len(pupil), (x/native.ao.tel.D+.5)*len(pupil)])
        valid = ndimage.map_coordinates(pupil, coordinates, order=0, mode='constant', cval=0) > .5
        self.cells = np.column_stack((x[valid], y[valid])); self.nv = len(self.cells)
        indices = np.rint((self.cells-cell_axis[0])/self.pitch).astype(int)
        delta = indices[:, None]-indices[None]
        self.ncell = ncell; self.span = 2*ncell-1
        self.didx = (delta[..., 0]+ncell-1)*self.span+delta[..., 1]+ncell-1
        gradients = []; temporal = []
        for layer, a in enumerate(self.contraction):
            u, v = a*self.qx, a*self.qy
            cell = np.sinc(self.pitch*u)*np.sinc(self.pitch*v)
            gradients.append(np.array([2j*np.pi*u*cell, 2j*np.pi*v*cell]))
            direction = np.deg2rad(native.ao.atm.wDir[layer])
            temporal.append(native._polc_response(-(self.qx*np.cos(direction)+self.qy*np.sin(direction))*native.ao.atm.wSpeed[layer]))
        self.gradients = np.asarray(gradients); self.temporal = np.asarray(temporal)
        limit = int(np.floor(float(native.freq.kcMax_)/self.dk))
        modes = np.arange(-limit, limit+1)
        mx, my = np.meshgrid(modes, modes, indexing='ij')
        supported = ((mx**2+my**2)*self.dk**2 <= float(native.freq.kcMax_)**2) & ((mx != 0) | (my != 0))
        self.qi = np.column_stack((mx[supported], my[supported])); self.q = self.qi*self.dk
        radius = np.asarray(native.ao.dms.opt_dir[0])/206264.8
        angle = np.deg2rad(native.ao.dms.opt_dir[1])
        self.targets = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
        self.target_weights = np.asarray(native.ao.dms.opt_weights)
        h = self.heights[-1]*native.strechFactor[-1]
        old_average = sum(w*np.exp(2j*np.pi*h*(native.freq.kxAO_*b[0]+native.freq.kyAO_*b[1]))
            for w, b in zip(self.target_weights, self.targets))
        mask = native.freq.mskInAO_.astype(bool)
        self.gain = float(np.vdot(old_average[mask], native.Popt[:, :, 0, -1][mask]).real/np.vdot(old_average[mask], old_average[mask]).real)
        self.phase = np.array([self.gain*(np.exp(2j*np.pi*h*(self.q@self.targets.T))@self.target_weights) for h in self.heights]).T
        self._covariances()
        self._reconstruct()
        self._sampling = None
        self.timings['construction_seconds'] = time.perf_counter()-started

    def _covariances(self):
        started = time.perf_counter(); s = self.span; p = self.pitch
        prior = np.zeros((self.nc, self.nc, s, s)); raw = np.zeros_like(prior)
        for layer, (a, h) in enumerate(zip(self.contraction, self.heights)):
            for g in range(self.ng):
                for k in range(g, self.ng):
                    shift = a*(self.offsets[g]-self.offsets[k])+h*(self.guide[g]-self.guide[k])
                    for i, j in ((0, 0), (0, 1), (1, 1)):
                        gradient = self.gradients[layer, i]*self.gradients[layer, j].conj()
                        values = [_fourier_sum(density*gradient*self.dk**2, self.dk,
                            shift-a*p*(self.ncell-1), a*p, s).real
                            for density in (self.prior[layer], self.raw[layer]*abs(self.temporal[layer])**2)]
                        pairs = [(2*g+i, 2*k+j)]
                        if i != j and g != k: pairs.append((2*g+j, 2*k+i))
                        for row, col in pairs:
                            for result, value in zip((prior, raw), values):
                                result[row, col] += value
                                if row != col: result[col, row] += value[::-1, ::-1]
        u, v = self.qx, self.qy; density = np.zeros_like(u)
        maximum = (self.order+.5)/p
        for layer, a in enumerate(self.contraction):
            q = np.hypot(u/a, v/a)
            value = self.native.ao.atm.spectrum(q)*self.native.ao.atm.weights[layer]/a**2
            value *= pistonFilter(float(self.native.ao.tel.D), q, dtype=self.native.dtype)
            value *= (abs(u/a) < maximum) & (abs(v/a) < maximum)
            density += value
        cell = np.sinc(p*u)*np.sinc(p*v)
        for axis in (u, v):
            near = abs(p*axis-np.rint(p*axis)) < 1e-12
            cell[near & (np.rint(p*axis) != 0)] = 0
        tiles = 2*self.order+1
        axes = [(density*(2*np.pi*axis*cell)**2).reshape(tiles, self.nbase, tiles, self.nbase).sum((0, 2)) for axis in (u, v)]
        self.floor_spectrum = np.maximum(*axes)*float(self.native.tomoRelRegFloor)/p**2
        self.floor_spectrum[self.nbase//2, self.nbase//2] = 0
        r = fft.ifft2(fft.ifftshift(self.floor_spectrum)).real
        lag = np.arange(-(self.ncell-1), self.ncell)
        regular = prior.copy()
        for channel in range(self.nc): regular[channel, channel] += r[lag[:, None] % self.nbase, lag[None] % self.nbase]
        def matrix(values):
            return values.reshape(self.nc, self.nc, -1)[:, :, self.didx].transpose(0, 2, 1, 3).reshape(self.nc*self.nv, self.nc*self.nv)
        self.cov_prior, self.cov_regular, self.cov_raw = map(matrix, (prior, regular, raw))
        if self.regularization:
            self.cov_regular.flat[::len(self.cov_regular)+1] += (
                self.regularization*np.diag(self.cov_prior).mean())
        self.factor = cho_factor(self.cov_regular, lower=True, check_finite=False)
        self.timings['measurement_covariance_seconds'] = time.perf_counter()-started

    def _target(self, layer, sl):
        q = self.q[sl]; a = self.contraction[layer]; h = self.heights[layer]
        index = self.qi[sl]+self.count//2
        density = self.prior[layer, index[:, 0], index[:, 1]]*self.dk**2
        density = density*self.phase[sl, layer]
        u = q*a
        cell = np.sinc(self.pitch*u[:, 0])*np.sinc(self.pitch*u[:, 1])
        gradient = 2j*np.pi*u*cell[:, None]
        guide = np.exp(2j*np.pi*(h*q@self.guide.T+u@self.offsets.T))
        measurement = (guide[:, :, None]*gradient[:, None]).reshape(len(q), self.nc)
        cells = np.exp(-2j*np.pi*(u@self.cells.T))
        return (density[:, None, None]*measurement.conj()[:, :, None]*cells[:, None]).reshape(len(q), self.nc*self.nv)

    def _reconstruct(self):
        started = time.perf_counter()
        self.weights = np.empty((len(self.q), self.nc*self.nv), complex)
        for first in range(0, len(self.q), 512):
            sl = slice(first, min(first+512, len(self.q)))
            target = sum(self._target(layer, sl) for layer in range(len(self.heights)))
            w = (cho_solve(self.factor, target.real.T, check_finite=False)+1j*cho_solve(self.factor, target.imag.T, check_finite=False)).T
            self.weights[sl] = w
            if first == 0:
                self.normal_equation_error = float(np.linalg.norm(w@self.cov_regular-target)/np.linalg.norm(target))
        self.timings['reconstruction_seconds'] = time.perf_counter()-started

    @property
    def source(self):
        """Materialize the source only for callers of the dense covariance API."""
        if self._source_covariance is None:
            self._source_covariance = _stationary_matrix(*self.source_terms,self._source_sampling)
        return self._source_covariance

    def _prepare_common_operator(self,sampling):
        """Synthesize the current common Fourier command on the optical pupil."""
        index = self.qi+self.count//2
        moments = self.kernels[:, index[:, 0], index[:, 1]]
        self.operator = np.empty((len(sampling.xs), self.weights.shape[1]))
        ys = (sampling.ys-sampling.n//2) % sampling.n
        xs = (sampling.xs-sampling.n//2) % sampling.n
        for first in range(0, self.weights.shape[1], 16):
            sl = slice(first, min(first+16, self.weights.shape[1])); w = self.weights[:, sl]
            power = np.zeros((sampling.n, sampling.n, w.shape[1]), complex)
            power[self.qi[:, 1] % sampling.n, self.qi[:, 0] % sampling.n] = w
            values = (fft.ifft2(power, axes=(0, 1))*sampling.n**2)[ys, xs]
            if abs(values.imag).max() > 2e-7: raise ValueError('Non-real finite-aperture command surface')
            self.operator[:, sl] = values.real-sampling.ramps.T@(moments@w).real

    def _common_correction_covariance(self):
        """Return complete common covariance for this backend's optical cache."""
        return (self.operator@self.cov_raw)@self.operator.T

    def prepare_optics(self, sampling, dense_source=True):
        """Prepare/cache correction and science covariance for one optical grid."""
        key = (sampling.n, sampling.native_n, sampling.native_side, sampling.period, sampling.signature)
        if self._sampling == key:
            if dense_source: _ = self.source
            return
        started = time.perf_counter()
        if sampling.native_n != self.native_n or not np.isclose(sampling.period, self.period):
            raise ValueError('Finite-aperture optics and reconstruction must share the physical Fourier period')
        self.kernels = sampling.tilt_moments(self.count, self.period)
        self._prepare_common_operator(sampling)
        self.correction = self._common_correction_covariance()
        self.source_terms = _stationary_terms(self.raw.sum(0)*self.dk**2, self.kernels, sampling, sampling.n)
        tail_started = time.perf_counter()
        self.atmospheric_tail = AtmosphericTail(self.native.ao.atm.r0,
            self.native.ao.atm.L0, self.dk, self.count, sampling.n, symmetric_source=True)
        tail_terms = self.atmospheric_tail.fitted_moments(sampling)
        self.source_terms = tuple(a+b for a,b in zip(self.source_terms,tail_terms))
        self.timings['atmospheric_tail_seconds'] = time.perf_counter()-tail_started
        self._source_sampling = sampling
        self._source_covariance = None
        if dense_source: _ = self.source
        # Factor only this known stationary high-frequency science term from
        # optical quadrature. All of its fitted-tilt and correction cross terms
        # remain in the full covariance. This is an algebraic factorization,
        # not an assumption that the high input is independent of correction.
        high = self.raw.sum(0)*self.dk**2*(np.hypot(self.qx, self.qy) > self.native.freq.kcMax_)
        self.stationary_high_lags = tuple(fft.ifft2(_fold(high, n)).real*n**2
            for n in (sampling.n, sampling.native_n))
        # The native blocked optics consume source_terms. The reduced-grid
        # diagnostic path also factors this same stationary source completion.
        if sampling.n == sampling.native_n:
            self.stationary_high_lags = tuple(c+tail_terms[0] for c in self.stationary_high_lags)
        else:
            native_tail = AtmosphericTail(self.native.ao.atm.r0,self.native.ao.atm.L0,
                self.dk,self.count,sampling.native_n,symmetric_source=True)
            self.stationary_high_lags = (self.stationary_high_lags[0]+tail_terms[0],
                self.stationary_high_lags[1]+native_tail.covariance())
        self.kernel_grid = 2**int(np.ceil(np.log2(max(self.count, 4*self.native_n))))
        self.interpolation = []
        for layer in range(len(self.heights)):
            values = _LayerInterpolation(self.kernel_grid)
            for j in range(2):
                base = self.raw[layer]*self.dk**2*self.gradients[layer, j].conj()*self.temporal[layer].conj()
                for i, multiplier in enumerate((1, self.kernels[0], self.kernels[1])):
                    c = fft.ifft2(_fold(base*multiplier, self.kernel_grid)).real*self.kernel_grid**2
                    plane=values.phase[j] if i==0 else values.fit[i-1,j]
                    plane[:] = ndimage.spline_filter(c, order=3, mode='grid-wrap')
            self.interpolation.append(values)
        self.aux_kernels = sampling.tilt_moments(self.native_n, self.period)
        auxiliary = np.real(self.native.differentialRefractionPSD()+self.native.chromatismPSD())
        auxiliary = np.broadcast_to(auxiliary, (self.native.freq.resAO, self.native.freq.resAO, self.native.ao.src.nSrc))
        self.auxiliary = auxiliary
        m = self.native.freq.resAO
        self.ao_start = int(np.ceil(self.native_n/2-m/2))
        norm = (2*self.native.freq.kcMax_/m*self.native.freq.wvlRef*1e9/(2*np.pi))**2
        wavelength = float(self.native.ao.src.wvl[0])
        conversion = (np.pi/wavelength*self.native_n*self.native.freq.PSDstep/self.native.freq.dk_)**2
        self.aux_conversion = norm*conversion/self.native_n**2
        self.additional_covariance = None
        if self.native.ao.tel.extraErrorNm > 0:
            # Reuse ordinary P3's independent additional-error spectrum and
            # normalization, with the same original-pupil tilt projection.
            power = self.native.extraErrorPSD()*self.aux_conversion
            self.additional_covariance = _stationary(power, self.aux_kernels, sampling, sampling.n)
            self.stationary_high_lags = tuple(previous+fft.ifft2(_fold(power, n)).real*n**2
                for previous, n in zip(self.stationary_high_lags, (sampling.n, self.native_n)))
        query_started = time.perf_counter()
        self._science_query = _ScienceMeasurementQuery(
            sampling.xy, self.cells, self.kernel_grid, self.period)
        self.timings['science_query_preparation_seconds'] = time.perf_counter()-query_started
        if self.native.finiteApertureBackend == "cuda" and not self.native.twoStageEnabled:
            from .finiteTwoStageCupy import CupyFiniteProducts
            self.device = CupyFiniteProducts(self, sampling)
        self._sampling = key
        self.timings['optical_cache_preparation_seconds'] = time.perf_counter()-started

    def residual_covariance(self, sampling, field):
        """Raw residual covariance after fitted-pupil tilt, in radians squared."""
        self.prepare_optics(sampling)
        started = time.perf_counter()
        cross,query_seconds = self._science_measurement_cross(sampling,field)
        independent = _stationary(self._independent_power(field), self.aux_kernels, sampling, sampling.n)
        sc = cross@self.operator.T
        covariance = self.source.copy()
        covariance += independent
        if self.additional_covariance is not None:
            covariance += self.additional_covariance
        covariance += self.correction
        covariance -= sc
        covariance -= sc.T
        return covariance, dict(science_measurement_seconds=query_seconds,
            residual_covariance_seconds=time.perf_counter()-started)

    def _science_measurement_cross(self, sampling, field):
        if getattr(self, "device", None) is not None:
            started=time.perf_counter()
            return self.device.to_host(self.device.crosses(field)[0]), time.perf_counter()-started
        started = time.perf_counter(); beta = self.native.ao.src.direction[:, field]
        cross = np.zeros((len(sampling.xs), self.nc*self.nv))
        values = np.empty((2, len(sampling.xs), self.nv))
        for layer, (a, h) in enumerate(zip(self.contraction, self.heights)):
            kernel = self.interpolation[layer]
            for guide in range(self.ng):
                origin = h*(beta-self.guide[guide])-a*self.offsets[guide]
                cx = origin[0]-a*self.cells[:, 0]; cy = origin[1]-a*self.cells[:, 1]
                base = np.array([cy, cx])*self.kernel_grid/self.period
                self._science_query.evaluate(kernel.host_phase(), origin, a, values)
                for j in range(2):
                    fit = np.array([ndimage.map_coordinates(kernel.fit[i,j], base, order=3, mode='grid-wrap', prefilter=False) for i in range(2)])
                    values[j] -= sampling.ramps.T@fit
                    cross[:, (2*guide+j)*self.nv:(2*guide+j+1)*self.nv] += values[j]
        return cross,time.perf_counter()-started

    def _independent_power(self, field):
        extra = np.zeros((self.native_n, self.native_n)); first = self.ao_start
        extra[first:first+self.native.freq.resAO, first:first+self.native.freq.resAO] = self.auxiliary[:, :, field]*self.aux_conversion
        return extra

    def residual_blocks(self, sampling, field):
        """Return the same physical residual as factors and bounded covariance blocks."""
        self.prepare_optics(sampling,dense_source=False)
        started = time.perf_counter()
        device=getattr(self, 'device', None)
        if device is not None and hasattr(self, 'common_basis'):
            before=time.perf_counter()
            cross=device.common_context(field)
            query_seconds=time.perf_counter()-before
            operator=self.common_basis
        else:
            cross,query_seconds = self._science_measurement_cross(sampling,field)
            operator=self.operator
        independent = _stationary_terms(self._independent_power(field),self.aux_kernels,sampling,sampling.n)
        residual = FiniteResidualCovariance(sampling,self.source_terms,self.correction,
            self.additional_covariance,cross,operator,independent)
        return residual,dict(science_measurement_seconds=query_seconds,
            residual_covariance_seconds=time.perf_counter()-started)

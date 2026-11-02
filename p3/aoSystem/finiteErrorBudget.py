"""TIPTOP-style error accounting through the actual finite correction operators.

Fitting is incoming power outside the native correction band; spatio-temporal
error is the in-band residual. Aliasing is correction power driven by outside
inputs (the GIRMOS input-band convention, not Fourier replica bookkeeping).
Their signed covariance is retained separately. Reporting never changes optics.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
import time

import numpy as np
from scipy import fft, linalg

from .finiteAperture import _stationary_terms
from .finiteBudgetInputs import band_view


@dataclass(frozen=True)
class FiniteErrorBudgetResult:
    """One field's HO budget in nm², with static NCPA kept separate.

    ``components_nm2`` contains nonnegative dynamic HO variances. Their sum
    plus ``fitting_aliasing_covariance_nm2`` is ``dynamic_ho_nm2``. The signed
    covariance includes the factor two. ``residual_variance_nm2`` is the native
    dynamic residual, used for closure. NCPA enters ``total_nm2`` only;
    LO remains outside this HO budget, following TIPTOP reporting.
    ``first_stage`` is the same-field common-correction budget for MOAO,
    otherwise None. It retains the actual common estimator and controller.
    ``seconds`` excludes shared preparation and covers paired field accounting
    when both stages are reported; it is shared by those records, not additive.
    ``as_dict`` is JSON compatible.
    """
    field_index: int
    stage: str
    cutoff_per_m: float
    components_nm2: dict
    fitting_aliasing_covariance_nm2: float
    residual_variance_nm2: float
    seconds: float
    ncpa_nm2: float = 0.
    first_stage: 'FiniteErrorBudgetResult | None' = None
    definition: str = 'tiptop_input_band_v1'

    def __post_init__(self):
        object.__setattr__(self, 'components_nm2', MappingProxyType(dict(self.components_nm2)))

    @property
    def rms_nm(self):
        return {name: float(np.sqrt(value)) for name, value in self.components_nm2.items()}

    @property
    def dynamic_ho_nm2(self):
        return sum(self.components_nm2.values()) + self.fitting_aliasing_covariance_nm2

    @property
    def quadrature_nm2(self):
        return sum(self.components_nm2.values()) + self.ncpa_nm2

    @property
    def total_nm2(self):
        return self.dynamic_ho_nm2 + self.ncpa_nm2

    @property
    def total_rms_nm(self):
        return float(np.sqrt(self.total_nm2))

    @property
    def closure_nm2(self):
        return self.dynamic_ho_nm2 - self.residual_variance_nm2

    def as_dict(self):
        return dict(field_index=self.field_index, stage=self.stage,
                    definition=self.definition, cutoff_per_m=self.cutoff_per_m,
                    components_nm2=dict(self.components_nm2),
                    fitting_aliasing_covariance_nm2=self.fitting_aliasing_covariance_nm2,
                    dynamic_ho_nm2=self.dynamic_ho_nm2, ncpa_nm2=self.ncpa_nm2,
                    quadrature_nm2=self.quadrature_nm2,
                    total_nm2=self.total_nm2, total_rms_nm=self.total_rms_nm,
                    residual_variance_nm2=self.residual_variance_nm2,
                    closure_nm2=self.closure_nm2, seconds=self.seconds,
                    first_stage=None if self.first_stage is None else self.first_stage.as_dict())

    def with_optics(self, *, ncpa_nm2):
        """Attach identical science-plane NCPA to both correction paths."""
        first = None if self.first_stage is None else self.first_stage.with_optics(
            ncpa_nm2=ncpa_nm2)
        return replace(self, ncpa_nm2=ncpa_nm2, first_stage=first)

    def print(self, wavelength):
        """Print the first-stage comparison when available, at the same field."""
        records = (self,) if self.first_stage is None else (self.first_stage, self)
        headings = (self.stage.upper(),) if self.first_stage is None else (
            'LTAO: first stage', 'MOAO: both stages')
        print('\n_____ ERROR BREAKDOWN _____')
        print(f'Field {self.field_index}; errors in nm RMS')
        print(f'{"":36}'+''.join(f'{heading:>24}' for heading in headings))
        print('-'*(36+24*len(records)))

        def row(label, value):
            print(f'{"."+label+":":36}'+''.join(f'{value(record):24.2f}' for record in records))

        row(f'Maréchal Strehl at {wavelength*1e6:.2f} µm [%]',
            lambda r: 100*np.exp(-r.total_nm2*(2*np.pi*1e-9/wavelength)**2))
        row('Residual wavefront error', lambda r: r.total_rms_nm)
        row('NCPA residual', lambda r: np.sqrt(r.ncpa_nm2))
        labels = dict(fitting='Fitting error', differential_refraction='Differential refraction',
                      chromatism='Chromatic error', aliasing='Aliasing error',
                      measurement_noise='Noise error', spatio_temporal='Spatio-temporal error',
                      extra_error='Extra error')
        for key, label in labels.items():
            row(label, lambda r: r.rms_nm[key])
        print('-'*(36+24*len(records)))
        row('Quadrature total', lambda r: np.sqrt(r.quadrature_nm2))
        row('Fitting–aliasing covariance [nm²]', lambda r: r.fitting_aliasing_covariance_nm2)


class FiniteErrorBudget:
    """Cached forward input-band accounting for one backend and optical pupil."""
    def __init__(self, backend, sampling):
        started = time.perf_counter()
        self.backend, self.sampling = backend, sampling
        self.key = backend._sampling
        self.weight = sampling.intensity
        self.records = {}
        self.low = band_view(backend, sampling)
        self._base = {id(view): self._shared_moments(view) for view in (backend, self.low)}
        self.auxiliary = dict(differential_refraction=backend.native.differentialRefractionPSD(),
                              chromatism=backend.native.chromatismPSD())
        self.preparation_seconds = time.perf_counter()-started

    def _product(self, left, right):
        """Piston-removed intensity-weighted trace of L R.T."""
        return float(np.einsum('i,ij,ij->', self.weight, left, right)
                     - (self.weight@left) @ (self.weight@right))

    def _stationary(self, terms):
        lags, points, tilt = terms
        sample = self.sampling
        weights = np.zeros((sample.n, sample.n))
        weights[sample.ys, sample.xs] = self.weight
        pairs = fft.ifft2(abs(fft.fft2(weights))**2).real
        ramps = sample.ramps.T
        return (float(lags[0, 0] - np.sum(lags*pairs))
                - 2*self._product(ramps, points.T) + self._product(ramps@tilt, ramps))

    def _shared_moments(self, b):
        raw = b.weights@b.cov_raw@b.weights.T
        common = raw.copy()
        lc = xc = None
        if b.native.removeMeanSlopes:
            p = b.projected_feedback
            lc = np.zeros((len(b.weights), 3*b.nc))
            lc[:, :b.nc] = b.weights@p.means
            xc = b.weights@p.cross[0]
            common += xc@lc.T + lc@xc.T + lc@p.covariance@lc.T
        extra = 0.
        if b.additional_covariance is not None:
            matrix = b.additional_covariance
            extra = float(self.weight@np.diag(matrix)-self.weight@matrix@self.weight)
        source = self._stationary(b.source_terms)
        base = source+self._product(b.common_basis@raw, b.common_basis)+extra
        return dict(base=base, source=source, extra=extra, common=common, lc=lc, xc=xc,
                    common_power=self._product(b.common_basis@common, b.common_basis))

    def _correction(self, b, field, residual):
        """Joint command power and first-stage residual from existing field factors."""
        shared = self._base[id(b)]
        value = shared['common_power']
        if not b.native.twoStageEnabled:
            return value, None
        target = b.science_surface.fit@b._prior_cross(field)
        rows = linalg.cho_solve(b.factor, target.T, check_finite=False).T
        science = rows@b.cbb@rows.T
        cross = b.weights@b.cab@rows.T
        if b.native.removeMeanSlopes:
            p = b.projected_feedback
            feedback = p.feedback_means-p.means@(p.means.T@p.feedback_means)
            ls = np.concatenate((np.zeros((len(rows), b.nc)), rows@p.means, -rows@feedback), axis=1)
            xs = rows@(p.cross[1]-b.feedback@p.cross[2])
            science += xs@ls.T + ls@xs.T + ls@p.covariance@ls.T
            lc, xc = shared['lc'], shared['xc']
            cross += xc@ls.T + lc@xs.T + lc@p.covariance@ls.T
        # The science update wraps the original common residual. Recover its
        # source/common cross moment before the joint mean-slope update.
        core = residual.base.base if b.native.removeMeanSlopes else residual.base
        common_residual = shared['source']+value-2*self._product(core.cross, core.operator)
        if b.native.removeMeanSlopes:
            atmospheric_z = (b.common_basis@xc + b.science_basis@xs
                             + .5*residual.operator@p.covariance-residual.update)
            common_residual -= 2*self._product(atmospheric_z, b.common_basis@lc)
        joint = (value+self._product(b.science_basis@science, b.science_basis)
                 + 2*self._product(b.common_basis@cross, b.science_basis))
        return joint, common_residual

    def _auxiliary(self, field):
        b = self.backend
        values = {}
        for name, spectrum in self.auxiliary.items():
            power = np.zeros((b.native_n, b.native_n))
            first, size = b.ao_start, b.native.freq.resAO
            power[first:first+size, first:first+size] = spectrum[:, :, field]*b.aux_conversion
            values[name] = self._stationary(_stationary_terms(power, b.aux_kernels,
                                                              self.sampling, self.sampling.n))
        return values

    def _residual(self, residual, base):
        if not hasattr(residual, 'base'):
            return base - 2*self._product(residual.cross, residual.operator)
        if residual.device is not None:
            operator, update, common_operator, common_cross = residual.device.reporting_factors()
            return base + 2*self._product(update, operator) - 2*self._product(common_cross, common_operator)
        return self._residual(residual.base, base) + 2*self._product(residual.update, residual.operator)

    def evaluate(self, field, residual):
        """Use fixed native estimators/controllers; partition forward inputs only."""
        if field in self.records:
            return self.records[field]
        started = time.perf_counter()
        b = self.backend
        shared, low_shared = self._base[id(b)], self._base[id(self.low)]
        auxiliary = self._auxiliary(field)
        full = self._residual(residual, shared['base']+sum(auxiliary.values()))
        low_residual, _ = self.low.residual_blocks(self.sampling, field)
        inside = self._residual(low_residual, low_shared['base'])
        fitting = shared['source']-low_shared['source']
        correction, common = self._correction(b, field, residual)
        low_correction, low_common = self._correction(self.low, field, low_residual)
        seconds = time.perf_counter()-started
        first = None
        if b.native.twoStageEnabled:
            first = self._record(field, 'ltao', common+sum(auxiliary.values())+shared['extra'],
                low_common, fitting, shared['common_power']-low_shared['common_power'],
                auxiliary, shared['extra'], seconds)
        result = self._record(field, 'moao' if b.native.twoStageEnabled else 'ltao',
            full, inside, fitting, correction-low_correction, auxiliary, shared['extra'], seconds)
        if first is not None:
            result = replace(result, first_stage=first)
        self.records[field] = result
        return result

    def _record(self, field, stage, full, inside, fitting, aliasing, auxiliary, extra, seconds):
        """Apply the same component definitions and closure to either stage."""
        b = self.backend
        cross = full-sum(auxiliary.values())-extra-inside-fitting-aliasing
        values = dict(fitting=fitting, spatio_temporal=inside, aliasing=aliasing,
                      measurement_noise=0., **auxiliary, extra_error=extra)
        tolerance = (2e-5 if getattr(b, 'array_backend', 'numpy') == 'mps' else 1e-9)*max(1., abs(full))
        if not np.isfinite(cross) or any(not np.isfinite(v) or v < -tolerance for v in values.values()):
            raise ValueError('Finite error-budget component is negative beyond numerical tolerance')
        scale = (b.native.freq.wvlRef*1e9/(2*np.pi))**2
        result = FiniteErrorBudgetResult(int(field), stage, float(b.native.freq.kcMax_),
            {name: float(max(0., value)*scale) for name, value in values.items()},
            float(cross*scale), float(full*scale), seconds)
        if abs(result.closure_nm2) > tolerance*scale:
            raise ValueError('Finite error-budget components do not close')
        return result

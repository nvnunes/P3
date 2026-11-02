# Finite-aperture error reporting

`fourierModel(..., getErrorBreakDown=True)` enables TIPTOP-style dynamic HO
accounting for `ReconstructionModel='finite_aperture'`, with or without science
correction. The calculation uses the actual finite estimators, mirror surfaces,
controller responses, slope projection and optical pupil. It changes no command,
residual covariance or PSF.

Reporting is deferred until `finite_aperture_residual(sampling, field)` receives
the optical pupil. `finiteErrorBreakDown[field]` then holds an immutable
`FiniteErrorBudgetResult`. `errorBreakDown(verbose=False)` returns the field
mapping; `verbose=True` additionally prints the evaluated direction closest to
on-axis using familiar TIPTOP headings. Calling before optical evaluation raises
an error. The finite convention does not overwrite legacy Fourier `wfe*` fields.

For MOAO, `result.first_stage` contains the matching LTAO budget at the same
science direction, using the common correction already computed for MOAO.
Printing shows **LTAO: first stage** and **MOAO: both stages** side by side.
The first column removes only the science-mirror correction; it does not refit
the common reconstructor or change its optimization directions. Both columns
use the same correction band and component definitions. Standalone LTAO has
`first_stage=None`. `as_dict()` includes the nested first-stage result.

## Component definitions

The correction band is the existing circular/square mask at `kcMax_`, the finest
configured common-DM cutoff. The definition identifier is `tiptop_input_band_v1`.
It is deliberately a band-based budget, not an orthogonal mirror-space budget.

| `components_nm2` key | Definition |
| --- | --- |
| `fitting` | Incoming atmospheric variance outside the correction band, including the native high-frequency source tail. This follows TIPTOP's cutoff convention. |
| `spatio_temporal` | Full residual driven by in-band atmospheric inputs, with both actual controllers and reconstructors retained. It includes reconstruction, tomography, timing and in-band projection/fitting limitations of the science mirror. |
| `aliasing` | Variance of the joint mirror correction driven by above-band atmospheric inputs, including covariance between the two mirrors. This is the GIRMOS input-band aliasing convention; it is not numerically identical to P3's Fourier shifted-replica convention. |
| `measurement_noise` | Zero in the supported finite scope. Nonzero physical HO measurement noise remains rejected; regularization is not noise. |
| `differential_refraction`, `chromatism` | Existing P3 auxiliary contributions, separately measured through the native pupil treatment. |
| `extra_error` | Independent dynamic `extraErrorNm` contribution after pupil treatment. |

The above-band atmosphere and induced correction are correlated. Their signed
cross term, including the factor two, is retained as
`fitting_aliasing_covariance_nm2`; it is not assigned an RMS or folded into
another named error. Positive components plus this term give `dynamic_ho_nm2`.
The independently contracted native residual is `residual_variance_nm2`, and
`closure_nm2` measures their numerical agreement. A separate sole-tomography or
servo-lag number is not inferred from unrelated Fourier formulas.

## Optical terms and totals

`ncpa_nm2` is the variance of the static phase actually applied by the supplied
optical sampling, after native RMS compensation, tilt removal and piston
subtraction. It is refreshed on every optical call, independently of the dynamic
cache. NCPA is excluded from `dynamic_ho_nm2` but included in `total_nm2`.

`total_nm2` includes NCPA but excludes LO, matching TIPTOP's HO breakdown.
`quadrature_nm2` excludes only the signed cross term. TIPTOP retains its
existing separate `LO_res [nm]` output.
`total_rms_nm` and `rms_nm` provide RMS values; `as_dict()` is JSON compatible.
The printed Maréchal Strehl is an approximation from total variance, not a
replacement for the predicted PSF or its measured Strehl. Unsupported physical
terms such as mirror go-to error are not assigned invented values.

`with_optics(ncpa_nm2=...)` attaches the same science-plane NCPA variance
to both columns. The comparison isolates the HO science-mirror correction.

## Cost and validation

Shared input-band moments and field results are cached per backend and optical
pupil. Configuration/geometry/pupil changes invalidate the appropriate cache.
No additional dense pupil covariance or physical-mode SVD is required.
The first-stage column reuses the common-command and field cross moments;
it requires no second simulation.
`t_errorBreakDown` records reporting milliseconds; field profiles expose
`error_budget_seconds`, including shared preparation on the first field.
Result `seconds` excludes shared preparation. Both stage records carry the
same paired accounting time; their times must not be added. Reporting remains
opt-in.

CPU/CUDA use float64; MPS products use their native float32 tolerance. Tests
compare aliasing with independent plane-wave propagation, check closure and
PSF parity, and exercise static-map changes, optical/configuration cache
invalidation, disabled science correction and separate extra-error accounting.
Conventional Fourier reporting remains unchanged.

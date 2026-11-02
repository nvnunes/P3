# Residual-driven science correction

Paired `[DM_science]` and `[RTC_science]` sections add one ground science mirror
per science direction. The common `[DM]` retains its shared optimization field
and `[RTC]` controller. Omit both science sections for LTAO. Supplying only one,
or the retired `[DM_Stage2]` section, raises an error.

```ini
[DM_science]
NumberActuators = [17]
DmPitchs = [0.49375]
DmHeights = [0.0]
InfModel = 'gaussian'
InfCoupling = [0.30]
PathInfGeometry = 'science-mirror.fits'

[RTC_science]
LoopGain_HO = 1.0
LoopDelaySteps_HO = 1
```

These are the accepted comparison's science controller values. The science RTC
is a direct residual-command controller on the common HO measurement clock:
`LoopGain_HO` multiplies the estimate and `LoopDelaySteps_HO` is an integer
number of those frames. It is not another integrator. Both values are required.
The common loop independently uses `[RTC] LoopGain_HO`, `SensorFrameRate_HO`
and `LoopDelaySteps_HO` (0.3, 500 Hz and 2 frames in that comparison).
`LoopGain_HO=0` recovers the corresponding single-stage calculation without
loading science geometry. There is no separate `Enabled` flag.

## Reconstruction methods

With `ReconstructionModel='alias_aware'`, the science correction uses the same
native layer estimator and ideal Fourier mirror bandwidth set by `DmPitchs`.
No finite geometry file is required. Both commands act on the same atmospheric
wave; the science command uses measurements residual to the common correction.
The command is `h1 A + h2 B - h1 h2 B D1 K1`; interference is retained before
squaring. Actual-alias propagation uses physical temporal frequencies and the
same shifted square-cell sensing. Source completion and existing additive
chromatic/refraction terms remain in normal P3 assembly.

With `ReconstructionModel='finite_aperture'`, both mirrors use the physical
Gaussian geometry described in [mirror surfaces](mirror-surfaces.md).
`PathInfGeometry` optionally overrides each mirror's regular layout;
omission derives it from the nominal mirror settings and telescope pupil.
The fixed directional atmospheric estimator acts on residual measurements.
Finite sensing, physical phase prior, LGS slope-mean projection with complete
feedback, and all atmospheric/inter-stage correlations are retained. Shared
moments are reused across the field. TIPTOP consumes bounded covariance strips
through its native full-flux pupil-fit/static-map optics; no atmospheric diagonal
PSD is added to that covariance.

Current science-stage scope: equal-pitch multiple SH WFSs, one ground mirror
in each stage, all physical reconstruction layers, an active common POLC, zero
injected HO noise, science pitch >= common pitch, and HO-only TIPTOP. Unsupported
combinations raise errors. These limitations do not restrict single-stage LTAO's
validated LO handoff.

## Numerical execution

`fourierModel(..., finiteApertureBackend='numpy')` and
`baseSimulation(..., finiteApertureBackend='numpy')` default to CPU float64 products.
For active two-stage finite aperture, `'cuda'` selects optional CuPy float64
products on the current CUDA device; `'mps'` selects optional PyTorch Metal
float32 products. Both use the same periodic cubic queries and covariance
factorization. Reconstruction and shared solves remain CPU float64.

Set `P3_DISABLE_GPU=1` before importing P3/TIPTOP: its global GPU array mode is
separate and incompatible with finite reconstruction. This does not disable the
explicit finite-aperture device products. Keep the selected CUDA device current
while using the model and its residuals. CuPy must be installed for the host's
CUDA runtime; PyTorch is needed only for MPS.

An unavailable requested device raises an error; there is no silent CPU fallback.
No rank, source band, kernel resolution or covariance term is discarded. Changing
the API value rebuilds the cached reconstructor. Device products are not supported
for single-stage finite or alias-aware reconstruction. There is no INI backend
option. Hardware parity tests in `test_finite_two_stage.py` and
`test_gaussian_mirrors.py` compare queries, covariance and PSFs against CPU; they
skip explicitly when their device is unavailable.

## Verification

Run `tests/test_two_stage_fourier.py`, `tests/test_finite_two_stage.py`,
`tests/test_gaussian_mirrors.py`, `tests/test_finite_mean_slopes.py` and
`tests/test_science_configuration.py` with the sibling TIPTOP/MASTSEL packages
available and `P3_DISABLE_GPU=1`, `MASTSEL_DISABLE_GPU=1`.

`tests/test_finite_device_products.py` additionally checks device-state lifetime,
cache selection and explicit unavailable-device failures. Hardware parity tests
in `test_finite_two_stage.py` and `test_gaussian_mirrors.py` require the selected
CUDA or Metal device; unavailable hardware is reported as a skip.

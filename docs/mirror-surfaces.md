# Finite mirror surfaces

`[sensor_HO] ReconstructionModel='finite_aperture'` uses physical Gaussian
actuator surfaces, a physical phase prior and full residual covariance. LGS
measurements have each sensor's mean x/y slopes removed, with complete POLC
feedback. This projection is not applied to genuine NGS HO measurements.
These are properties of the model, not independent switches.

Each active mirror (`DM`, and optionally `DM_science`) uses existing
`InfModel='gaussian'` and `InfCoupling=[...]`, with optional `PathInfGeometry`.
The optional path resolves through the normal P3 external-file resolver.
Another influence model is rejected. Without the path (or with an empty string),
a centred regular grid uses the resolved `NumberActuators` and `DmPitchs`.
Actuator centres are retained where the nearest telescope-pupil sample is
illuminated. Calibration coordinates span `[-D/2, D/2]` at the pupil's own
resolution; no padding is added. Each mirror chooses its default or override
independently. A supplied FITS file replaces this layout, allowing a padded grid
and an explicit valid-actuator mask. Invalid supplied files raise errors.

## Geometry FITS contract

The primary HDU has `GEOMVER=1`. These four image extensions are required;
additional extensions are rejected:

| Extension | Contents |
| --- | --- |
| `ACTUATOR_AXIS` | Centred, increasing, uniformly spaced 1D coordinates shared by the actuator grid's x/y axes; `BUNIT='m'`. |
| `VALID_ACTUATORS` | Nonempty binary y/x mask on that square grid; commands follow row-major mask order. |
| `PUPIL_X`, `PUPIL_Y` | Centred, increasing, uniformly spaced 1D coordinates for actuator-tilt calibration; `BUNIT='m'`. |

The calibration grid must contain the configured telescope pupil with equal
integer zero padding on opposite sides. Pupil samples are placed on these
coordinates without resampling. The telescope owns the pupil weights; the INI
owns coupling. Neither is duplicated in the FITS file. Downstream owns actuator
padding, selection and calibration coordinates. P3 contains no instrument-specific
sizes or masks. Geometry content participates in cache invalidation.

The surface fit estimates raw phase on the union of nominal valid square WFS
cells' closed 3x3 nodes, fits actuator amplitudes, and removes fitted actual-pupil
actuator tilt. The unit-phase singular-value cutoff is fixed at `5e-4`. Physical
weighted optimization directions are used directly, without a scalar gain from
the Fourier projector. Gaussian gradient averages supply common-mirror WFS
feedback. Final optical tilt removal is a separate step.

## Estimator regularization

`[sensor_HO] ReconstructionRegularization` defaults to `0.0`. It adds this
fraction of the mean prior slope variance to the reconstruction covariance's
diagonal. It is separate from `TomographicRegularizationFloor` and does not
inject measurement noise. The accepted comparison uses `0.1` and a numerical
floor of `1e-6`. The setting requires finite-aperture reconstruction.

The prior, geometry, regularization, atmospheric source completion and all
changed atmospheric/command cross terms are shared by single-stage LTAO and
[science correction](two-stage-fourier.md). Existing finite-model restrictions
on geometry, noise and optics still apply. Reconstruction runs on CPU in float64. Optional CUDA/Metal covariance products
are selected through the `finiteApertureBackend` Python API, not an INI setting.

The experimental `SurfaceModel`, `SurfaceGeometry`, `RemoveMeanSlopes`,
`ReconstructionPrior`, `ReconstructionNoiseFraction` and `CovarianceArrayBackend`
keys are rejected, including when false or inactive. Replaying earlier experiments
requires their retained sources and inputs.

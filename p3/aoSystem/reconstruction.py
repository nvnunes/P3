"""Public reconstruction-model selection shared by P3 and TIPTOP."""

RECONSTRUCTION_MODELS = ('standard', 'alias_aware', 'finite_aperture')
RETIRED_RECONSTRUCTION_KEYS = (
    'AliasAwareReconstruction', 'FiniteApertureReconstruction',
    'TomographicAliasing', 'SymmetricAliasing', 'SuperResolution',
)


def reconstruction_model(sensor):
    """Validate sensor_HO settings and return the explicit model or standard.

    Retired switches are rejected even when false. Historical combinations
    are not equivalent to a unique model; callers must choose deliberately.
    Physical sampling, optics, regularization and controller settings remain
    independent of this selector.
    """
    retired = [key for key in RETIRED_RECONSTRUCTION_KEYS if key in sensor]
    if retired:
        raise ValueError(
            'Retired sensor_HO reconstruction keys: ' + ', '.join(retired) +
            ". Remove them and set ReconstructionModel to 'standard', "
            "'alias_aware' or 'finite_aperture'. Historical experimental "
            'combinations require their frozen source/configuration; no automatic migration is defined.')
    model = sensor.get('ReconstructionModel', 'standard')
    if not isinstance(model, str) or model not in RECONSTRUCTION_MODELS:
        raise ValueError("sensor_HO.ReconstructionModel must be 'standard', 'alias_aware' or 'finite_aperture'")
    return model

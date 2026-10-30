"""Public reconstruction-model selection shared by P3 and TIPTOP."""

# None retains the existing detector-derived measurement duration.
MODEL_TEMPORAL_BEHAVIOUR = {
    'standard': ('integrator', None),
    'alias_aware': ('polc', 0.0),
    'finite_aperture': ('polc', 0.0),
}
RECONSTRUCTION_MODELS = tuple(MODEL_TEMPORAL_BEHAVIOUR)
RETIRED_RECONSTRUCTION_KEYS = (
    'AliasAwareReconstruction', 'FiniteApertureReconstruction',
    'TomographicAliasing', 'SymmetricAliasing', 'SuperResolution',
)


def reconstruction_model(sensor, rtc=None):
    """Validate sensor_HO settings and return the explicit model or standard.

    Retired switches are rejected even when false. Historical combinations
    are not equivalent to a unique model; callers must choose deliberately.
    Physical sampling, optics, numerical regularization, loop gain, frame rate
    and delay remain independent. Controller response and measurement averaging
    belong to the model; retired timing keys are not accepted as overrides.
    """
    retired = [key for key in RETIRED_RECONSTRUCTION_KEYS if key in sensor]
    if retired:
        raise ValueError(
            'Retired sensor_HO reconstruction keys: ' + ', '.join(retired) +
            ". Remove them and set ReconstructionModel to 'standard', "
            "'alias_aware' or 'finite_aperture'. Historical experimental "
            'combinations require their frozen source/configuration; no automatic migration is defined.')
    retired_timing = [section+'.'+key
        for section, settings in (('sensor_HO', sensor), ('RTC', rtc or {}))
        for key in ('TemporalModel_HO', 'ExposureTime') if key in settings]
    if retired_timing:
        raise ValueError(
            'Retired temporal configuration keys: ' + ', '.join(retired_timing) +
            '. Remove these keys. ReconstructionModel owns controller response '
            'and WFS measurement averaging: standard uses the existing integrator '
            'and detector duration; alias_aware and finite_aperture use POLC and '
            'instantaneous measurements. Historical timing combinations require '
            'their frozen source/configuration.')
    model = sensor.get('ReconstructionModel', 'standard')
    if not isinstance(model, str) or model not in RECONSTRUCTION_MODELS:
        raise ValueError("sensor_HO.ReconstructionModel must be 'standard', 'alias_aware' or 'finite_aperture'")
    return model

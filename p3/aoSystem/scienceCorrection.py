"""Configuration contract shared by P3 reconstruction and TIPTOP orchestration.

The optional science mirror/controller pair uses the common HO measurement
clock. Its controller is a direct residual command, not a second integrator.
"""
from dataclasses import dataclass
from .finiteConfiguration import _number, configuration as finite_configuration


@dataclass(frozen=True)
class ScienceCorrection:
    """Ground science mirror and its feed-forward response.

    Pitch is nominal metres, actuators the nominal count across the pupil,
    gain a dimensionless command multiplier, and delay integer HO frames.
    """
    pitch: float
    actuators: int
    gain: float
    delay: int

    @property
    def active(self):
        return self.gain > 0


def configuration(config, model):
    """Return (science correction or None, reconstruction regularization).

    Reject retired switches even when false. This validates scientific inputs
    without reading geometry files or selecting a numerical execution device.
    Both orchestration and reconstruction call this single contract owner.
    """
    if 'DM_Stage2' in config:
        raise ValueError('DM_Stage2 is retired; use DM_science and RTC_science')
    retired = {
        'DM_science': ('SurfaceModel', 'SurfaceGeometry', 'PathInfluenceGeometry', 'Enabled',
                       'CommandGain', 'CommandDelaySteps', 'LoopGain_HO', 'LoopDelaySteps_HO'),
        'RTC_science': ('CommandGain', 'CommandDelaySteps'),
    }
    invalid = [f'{section}.{key}' for section, keys in retired.items()
               for key in keys if key in config.get(section, {})]
    if invalid:
        raise ValueError('Retired configuration keys: ' + ', '.join(invalid)
                         + '. Use ReconstructionRegularization, InfModel/InfCoupling,'
                         ' PathInfGeometry and RTC_science.LoopGain_HO/LoopDelaySteps_HO.')
    regularization = finite_configuration(config, model)
    mirrors = 'DM_science' in config
    controller = 'RTC_science' in config
    if mirrors != controller:
        raise ValueError('DM_science and RTC_science must be supplied together')
    if not mirrors:
        return None, regularization
    dm, rtc = config['DM_science'], config['RTC_science']
    allowed = {'NumberActuators', 'DmPitchs', 'DmHeights', 'InfModel',
               'InfCoupling', 'PathInfGeometry'}
    for name, values, keys in (('DM_science', dm, allowed),
                              ('RTC_science', rtc, {'LoopGain_HO', 'LoopDelaySteps_HO'})):
        extra = set(values) - keys
        if extra:
            raise ValueError(f'Unknown {name} settings: ' + ', '.join(sorted(extra)))
    def single(key):
        value = dm.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise ValueError(f'DM_science.{key} requires one value')
        return value[0]
    pitch = _number(single('DmPitchs'), 'DM_science.DmPitchs')
    if pitch == 0:
        raise ValueError('DM_science.DmPitchs must be positive')
    actuators = _number(single('NumberActuators'), 'DM_science.NumberActuators', 2, True)
    if dm.get('DmHeights', [0.]) != [0.]:
        raise ValueError('DM_science currently requires ground conjugation')
    if not {'LoopGain_HO', 'LoopDelaySteps_HO'} <= set(rtc):
        raise ValueError('RTC_science requires explicit LoopGain_HO and LoopDelaySteps_HO')
    gain = _number(rtc['LoopGain_HO'], 'RTC_science.LoopGain_HO', maximum=1)
    delay = _number(rtc['LoopDelaySteps_HO'], 'RTC_science.LoopDelaySteps_HO', integer=True)
    if gain > 0 and model not in ('alias_aware', 'finite_aperture'):
        raise ValueError('Science correction requires alias_aware or finite_aperture')
    return ScienceCorrection(pitch, actuators, gain, delay), regularization


def covariance_backend(value, model, science):
    """Validate execution policy; GPU products currently accelerate two-stage finite optics."""
    if not isinstance(value, str) or value not in ('numpy', 'cuda', 'mps'):
        raise ValueError("finiteApertureBackend must be 'numpy', 'cuda' or 'mps'")
    if value != 'numpy' and (model != 'finite_aperture' or science is None or not science.active):
        raise ValueError(f'finiteApertureBackend={value} requires active finite-aperture science correction')
    return value

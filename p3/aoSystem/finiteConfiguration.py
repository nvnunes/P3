"""Validation of common finite-aperture reconstruction inputs."""
import math


def _number(value, name, minimum=0, integer=False, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)
            or (integer and value != int(value))):
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"}'
                         f' >= {minimum}' + (f' and <= {maximum}' if maximum is not None else ''))
    return int(value) if integer else float(value)



def configuration(config, model):
    """Validate shared model settings and return the regularization fraction."""
    retired = {
        'sensor_HO': ('RemoveMeanSlopes', 'ReconstructionPrior',
                      'ReconstructionNoiseFraction', 'CovarianceArrayBackend'),
        'DM': ('SurfaceModel', 'SurfaceGeometry', 'PathInfluenceGeometry'),
    }
    invalid = [f'{section}.{key}' for section, keys in retired.items()
               for key in keys if key in config.get(section, {})]
    if invalid:
        raise ValueError('Retired configuration keys: ' + ', '.join(invalid)
                         + '. Use ReconstructionRegularization, InfModel/InfCoupling'
                         ' and PathInfGeometry.')
    sensor = config.get('sensor_HO', {})
    if 'ReconstructionRegularization' in sensor and model != 'finite_aperture':
        raise ValueError('ReconstructionRegularization requires finite_aperture')
    return _number(sensor.get('ReconstructionRegularization', 0.),
                   'ReconstructionRegularization')

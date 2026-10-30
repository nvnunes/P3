"""Model selection, migration errors and per-instance scientific state."""

import numpy as np
import pytest
from unittest.mock import patch

from p3.aoSystem.reconstruction import reconstruction_model, RETIRED_RECONSTRUCTION_KEYS
from p3.aoSystem.fourierModel import fourierModel
from test_tomographic_aliasing import _case
from test_finite_aperture import small_model


@pytest.mark.parametrize('key', RETIRED_RECONSTRUCTION_KEYS)
@pytest.mark.parametrize('value', [False, True])
def test_retired_keys_are_never_interpreted(key, value):
    with pytest.raises(ValueError, match='Remove them and set ReconstructionModel'):
        reconstruction_model({key: value, 'ReconstructionModel': 'standard'})


@pytest.mark.parametrize('value', [False, True, None, 1, '', 'Standard', 'unknown'])
def test_invalid_model(value):
    with pytest.raises(ValueError, match='ReconstructionModel must be'):
        reconstruction_model({'ReconstructionModel': value})


def test_omitted_and_explicit_standard_and_instance_isolation(tmp_path):
    first = _case(tmp_path/'first', '')
    snapshot = {k: getattr(first, k).copy() for k in ('W', 'PSD')}
    alias = _case(tmp_path/'alias', "ReconstructionModel = 'alias_aware'")
    finite = small_model(tmp_path/'finite', True)
    second = _case(tmp_path/'second', "ReconstructionModel = 'standard'")
    assert finite.PSD is None and not hasattr(finite, 'W')
    assert alias.W.shape[-1] == 2*alias.nGs
    assert second.W.shape[-1] == second.nGs
    for key, expected in snapshot.items():
        np.testing.assert_array_equal(getattr(first, key), expected)
        np.testing.assert_array_equal(getattr(second, key), expected)
    assert reconstruction_model({}) == 'standard'


def test_finite_model_does_not_execute_the_alternative(tmp_path):
    with patch.object(fourierModel, 'spatialReconstructor', side_effect=AssertionError('ordinary estimator executed')), \
         patch.object(fourierModel, 'powerSpectrumDensity', side_effect=AssertionError('ordinary PSD executed')), \
         patch.object(fourierModel, 'controller', side_effect=AssertionError('ordinary controller propagation executed')):
        value = small_model(tmp_path/'finite', True)
    assert value.reconstructionModel == 'finite_aperture'
    assert value.Popt is not None
    assert value.PSD is None and value.opticalPSD is None


@pytest.mark.parametrize('setting,value,message', [
    ('noise', .01, 'noise'), ('temporal', 'integrator', 'POLC'),
    ('exposure', .001, 'instantaneous'), ('ncpa', 1., 'NCPA'),
    ('layers', None, 'all physical reconstruction layers'),
])
def test_finite_scope_does_not_broaden(tmp_path, setting, value, message):
    from p3.aoSystem.finiteAperture import validate_native
    model = small_model(tmp_path/'finite', True)
    if setting == 'noise': model.ao.wfs.processing.noiseVar[0] = value
    elif setting == 'temporal': model.temporalModel = value
    elif setting == 'exposure': model.exposureTime = value
    elif setting == 'ncpa': model.ao.tel.opdMap_on = np.ones((4, 4))*value
    elif setting == 'layers': model.atm_mod.heights = model.atm_mod.heights[:1]
    with pytest.raises(ValueError, match=message): validate_native(model)


@pytest.mark.parametrize('model,response,exposure', [
    ('standard', 'integrator', None), ('alias_aware', 'polc', 0.0),
    ('finite_aperture', 'polc', 0.0),
])
def test_model_owns_temporal_response(tmp_path, model, response, exposure):
    if model == 'finite_aperture':
        value = small_model(tmp_path/'finite', True)
    else:
        value = _case(tmp_path/'case', f'ReconstructionModel = {model!r}')
    assert value.temporalModel == response
    assert value.exposureTime == exposure
    for key, runtime in [('LoopGain_HO', 'gain'), ('SensorFrameRate_HO', 'rate'), ('LoopDelaySteps_HO', 'delay')]:
        assert value.my_data_map['RTC'][key] == value.ao.rtc.holoop[runtime]
    durations = [value._wfs_exposure_time(g) for g in range(value.nGs)]
    expected = ([d.clock_rate/value.ao.rtc.holoop['rate'] for d in value.ao.wfs.detector]
                if exposure is None else [0.0]*value.nGs)
    np.testing.assert_array_equal(durations, expected)


@pytest.mark.parametrize('model', [None, 'standard', 'alias_aware', 'finite_aperture'])
@pytest.mark.parametrize('section,key', [('RTC','TemporalModel_HO'), ('sensor_HO','ExposureTime')])
@pytest.mark.parametrize('value', [None, False, 0.0, 'polc'])
def test_retired_timing_is_rejected_even_if_it_matches_the_model(model, section, key, value):
    sensor = {} if model is None else {'ReconstructionModel': model}
    rtc = {}
    (sensor if section == 'sensor_HO' else rtc)[key] = value
    with pytest.raises(ValueError, match='Remove these keys. ReconstructionModel owns'):
        reconstruction_model(sensor, rtc)

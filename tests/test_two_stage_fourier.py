"""Shared-input cascade, native preservation and bounded scientific scope."""
from pathlib import Path
import numpy as np
import pytest

from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.twoStageFourier import applied_science_response, coupled_command
from p3.aoSystem.scienceCorrection import configuration
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE


_STAGE = '''
[DM_science]
NumberActuators = [17]
DmPitchs = [0.5]
DmHeights = [0.0]
[RTC_science]
LoopGain_HO = {gain}
LoopDelaySteps_HO = 2
'''


def build(folder, stage=None, noise=0.):
    folder.mkdir(parents=True, exist_ok=True)
    text = _INI_TEMPLATE.format(nph=[10000]*3, noisevar=repr([noise]*3))
    text = text.replace('[sensor_HO]', "[sensor_HO]\nReconstructionModel = 'alias_aware'\nTomographicRegularizationFloor = 1e-6")
    text = text.replace('Zenith = [0.0]\nAzimuth = [0.0]',
                        'Zenith = [0.0, 20.0, 40.0]\nAzimuth = [0.0, 45.0, 180.0]')
    if stage is not None:
        text += stage
    path = folder/'case.ini'
    path.write_text(text)
    return fourierModel(str(path), calcPSF=False, verbose=False, display=False,
                        reduce_memory=False, computeFocalAnisoCov=False, computeOpticalPSD=True)


@pytest.mark.parametrize('frequency', [0., 13., -47.])
def test_transfer_against_independent_frame_queue(frequency):
    rate, gain, delay = 500., .3, 2
    # Complex scalar coefficients retain nontrivial angular/alignment phases.
    k1, k2, dm = .7+.13j, .9-.21j, .8+.11j
    y = np.exp(2j*np.pi*frequency*np.arange(3000)/rate)
    common = np.zeros(y.size+delay, complex)
    science = np.zeros_like(common)
    memory = 0j
    for t in range(y.size):
        memory = (1-gain)*memory+gain*k1*y[t]
        common[t+delay] = memory
        science[t+delay] = k2*(y[t]-dm*common[t])
    p = np.exp(-2j*np.pi*frequency/rate)
    h1 = gain*p**delay/(1-(1-gain)*p)
    h2 = applied_science_response(frequency, rate, 1., delay)
    expected = coupled_command(k1, k2, k2*dm, h1, h2)
    np.testing.assert_allclose((common[1000:3000]+science[1000:3000])/y[1000:],
                               expected, rtol=2e-12, atol=2e-12)


def test_exact_shared_wave_limits():
    assert coupled_command(1., 1., 1., 1., 1.) == 1.
    assert coupled_command(.4, 1., 1., 1., 1.) == 1.
    assert coupled_command(.4, .8, .8, 0., 1.) == .8
    assert coupled_command(.4, .8, .8, 1., 0.) == .4


@pytest.mark.parametrize('stage', [_STAGE.format(gain=0.)])
def test_disabled_stage_reproduces_native_spectra_and_weights(tmp_path, stage):
    native = build(tmp_path/'native')
    control = build(tmp_path/'control', stage)
    assert not control.twoStageEnabled
    for key in ('W', 'Wtomo', 'PSD', 'opticalPSD'):
        np.testing.assert_array_equal(getattr(native, key), getattr(control, key))


def test_coupled_main_and_alias_are_positive_and_optically_consistent(tmp_path):
    value = build(tmp_path, _STAGE.format(gain=1.))
    assert value.twoStageEnabled
    for calculate in (value.twoStage.main_psd, value.twoStage.alias_psd):
        variance, optical = calculate(optical_pair=True)
        assert variance.shape[-1] == 3
        assert np.isfinite(variance).all() and np.isfinite(optical).all()
        assert variance.min() >= 0 and optical.min() >= 0
        assert np.all(optical >= variance-1e-12)
    # Independent direct matrix construction checks the same-wave interference.
    stage = value.twoStage
    prepared = stage._prepare_wave(stage.kx, stage.ky)
    measured, common, h1, h2 = prepared
    for source in range(3):
        row = stage.science_rows[..., source, :]
        residual_measured = measured-stage.dm_measurement[..., :, None]*h1[..., 0, None, :]*common[..., None, :]
        expected = h1[..., 0, :]*common+h2[..., 0, :]*np.einsum('...m,...ml->...l',row,residual_measured)
        actual = stage.command(stage.kx, stage.ky, slice(source,source+1))[..., 0, :]
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_nonzero_noise_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match='zero injected HO noise'):
        build(tmp_path, _STAGE.format(gain=1.), noise=.01)


@pytest.mark.parametrize('key,value', [('LoopDelaySteps_HO', -.5), ('LoopDelaySteps_HO', True),
    ('LoopGain_HO', 'bad'), ('LoopGain_HO', 2.), ('DmPitchs', [0]),
    ('NumberActuators', [2.5]), ('Enabled', 1), ('DmHeights', [1000.])])
def test_invalid_science_settings(key, value):
    setting = {'DmPitchs': [.5], 'NumberActuators': [17]}
    rtc={'LoopGain_HO':1.,'LoopDelaySteps_HO':2}
    (rtc if key.startswith('Loop') else setting)[key] = value
    with pytest.raises(ValueError):
        configuration({'DM_science':setting,'RTC_science':rtc},'alias_aware')

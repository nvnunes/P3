"""Native queued POLC timing, exposure and white measurement-noise contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from p3.aoSystem.fourierModel import fourierModel
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _build


def _case(tmp_path, reconstruction='alias_aware', noise='[0., 0., 0.]', extra=''):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = _INI_TEMPLATE.format(nph=[10000]*3, noisevar=noise)
    options = 'GridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]\n'+extra
    if reconstruction is not None:
        options += f'\nReconstructionModel = {reconstruction!r}'
    text = text.replace('[sensor_HO]', '[sensor_HO]\n'+options)
    path = tmp_path/'case.ini'
    path.write_text(text)
    return _build(str(path))


@pytest.mark.parametrize('gain', [0., .1, .3, .9, 1.])
@pytest.mark.parametrize('delay', [0, 1, 2, 4])
def test_response_matches_discrete_command_queue(gain, delay):
    model = fourierModel.__new__(fourierModel)
    model.ao = SimpleNamespace(rtc=SimpleNamespace(holoop=dict(rate=500., gain=gain, delay=delay)))
    frequency = np.array([0., .1, 5., 100., 245., -100.])
    phase = np.exp(2j*np.pi*np.arange(12000)[:, None]*frequency/500.)
    command = np.zeros(len(frequency), complex)
    commands = []
    for wave in phase:
        command = (1-gain)*command+gain*wave
        commands.append(command.copy())
    commands = np.array(commands)
    applied = np.roll(commands, delay, axis=0)
    measured = np.mean(applied[4000:]/phase[4000:], axis=0)
    np.testing.assert_allclose(model._polc_response(frequency), measured, atol=2e-12)


def test_omitted_and_explicit_standard_keep_existing_timing(tmp_path):
    default = _case(tmp_path/'default', reconstruction=None)
    explicit = _case(tmp_path/'explicit', reconstruction='standard')
    np.testing.assert_array_equal(default.PSD, explicit.PSD)
    for name in ('W', 'Wtomo', 'Popt', 'Cphi', 'Cphi_mod', 'psdAlias', 'psdNoise'):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    assert default.temporalModel == 'integrator' and default.exposureTime is None


def test_polc_gain_and_zero_wind_limits(tmp_path):
    model = _case(tmp_path)
    frozen_weights = model.W.copy()
    model.ao.atm.wSpeed[:] = 0
    powers = []
    for gain in (.1, .3, 1.):
        model.ao.rtc.holoop['gain'] = gain
        model.controller()
        powers.append(model.spatioTemporalPSD())
    np.testing.assert_allclose(powers[0], powers[1], rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(powers[0], powers[2], rtol=2e-13, atol=1e-12)
    np.testing.assert_array_equal(model.W, frozen_weights)
    model.ao.rtc.holoop['gain'] = 0
    model.controller()
    np.testing.assert_array_equal(model.aliasingPSD(), 0.)
    np.testing.assert_allclose(model.spatioTemporalPSD(),
                              model.Cphi.sum(axis=-1)[:, :, None]*model.freq.mskInAO_[:, :, None])


@pytest.mark.parametrize('noise', ['[0.1, 0.2, 0.3]', '[0., 0.2, 0.3]'])
def test_noise_gain_matches_independent_impulse_energy(tmp_path, noise):
    model = _case(tmp_path, noise=noise)
    # Sum the squared impulse response of u[n]=(1-g)u[n-1]+g*noise[n].
    gains = (.1, .3, .9, 1.)
    powers = []
    for gain in gains:
        model.ao.rtc.holoop['gain'] = gain
        model.controller()
        expected = np.sum((gain*(1-gain)**np.arange(10000))**2)
        np.testing.assert_allclose(model.noiseGain, expected, rtol=2e-14)
        powers.append(model.noisePSD()/expected)
    for power in powers[1:]:
        np.testing.assert_allclose(power, powers[0], rtol=3e-14)
    model.ao.rtc.holoop['gain'] = 0
    model.controller()
    np.testing.assert_array_equal(model.noisePSD(), 0.)


def test_alias_power_matches_independent_cell_and_time_integration(tmp_path):
    from p3.aoSystem.FourierUtils import pistonFilter

    model = _case(tmp_path)
    exposure = 0.0
    row, col = model.freq.resAO//2+2, model.freq.resAO//2+1
    k = np.array([model.freq.kxAO_[row, col], model.freq.kyAO_[row, col]])
    unit = k/np.linalg.norm(k)
    pitch = model.ao.wfs.optics[0].dsub
    offsets = np.array(model.my_data_map['sensor_HO']['GridOffsets'])*pitch
    nodes, weights = np.polynomial.legendre.leggauss(24)
    nodes, weights = nodes*pitch/2, weights/2
    wind = np.deg2rad(model.ao.atm.wDir)
    velocity = np.stack((model.ao.atm.wSpeed*np.cos(wind), model.ao.atm.wSpeed*np.sin(wind)), axis=1)
    heights = model.ao.atm.heights*model.strechFactor
    rtc = model.ao.rtc.holoop
    total = 0.
    for m in model._alias_shift_grid():
        for n in model._alias_shift_grid():
            if m == n == 0:
                continue
            q = k-np.array([m, n])/pitch
            frequency = -velocity @ q
            # Numerically average the input phasor over the measurement window.
            window = np.sum(weights*np.exp(2j*np.pi*frequency[:, None]*
                                          (nodes/pitch*exposure)), axis=-1)
            # Transform the sampled impulse response, independently of H(f).
            steps = np.arange(2000)
            impulse = rtc['gain']*(1-rtc['gain'])**steps
            timing = np.sum(impulse*np.exp(-2j*np.pi*frequency[:, None]*
                                          (steps+rtc['delay'])/rtc['rate']), axis=-1)
            for layer, height in enumerate(heights):
                measured = []
                for guide, offset in enumerate(offsets):
                    center = offset+height*model.gs.direction[:, guide]
                    phase = np.exp(2j*np.pi*(q[0]*(nodes[:, None]+center[0])+
                                             q[1]*(nodes[None, :]+center[1])))
                    cell = np.einsum('i,ij,j->', weights, phase, weights)
                    measured.extend(2j*np.pi*q*cell)
                command = model.W[row, col, 0] @ np.array(measured)*window[layer]*timing[layer]
                spectrum = (q @ q+1/model.ao.atm.L0**2)**(-11/6)
                piston = pistonFilter(model.ao.tel.D, np.array([[np.linalg.norm(q)]]))[0, 0]
                total += model.ao.atm.weights[layer]*abs(command)**2*spectrum*piston
    total *= .0229*model.ao.atm.r0**(-5/3)*model.freq.mskInAO_[row, col]
    np.testing.assert_allclose(model.psdAlias[row, col], total, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize('section,key,value', [
    ('RTC', 'TemporalModel_HO', "'polc'"),
    ('RTC', 'TemporalModel_HO', "'integrator'"),
    ('sensor_HO', 'ExposureTime', '0.0'),
    ('sensor_HO', 'ExposureTime', 'None'),
])
def test_actual_config_rejects_retired_temporal_options(tmp_path, section, key, value):
    text = _INI_TEMPLATE.format(nph=[10000]*3, noisevar='[0.,0.,0.]')
    text = text.replace('['+section+']', '['+section+']\n'+key+' = '+value)
    path = tmp_path/'invalid.ini'; path.write_text(text)
    with pytest.raises(ValueError, match='Remove these keys. ReconstructionModel owns'):
        _build(str(path))

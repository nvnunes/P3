"""Optical covariance power is distinct from unchanged variance reporting."""

from configparser import ConfigParser
from unittest.mock import patch

import numpy as np
import pytest

from p3.aoSystem import FourierUtils
from p3.aoSystem.fourierModel import fourierModel
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _p3_path


def _model(path, optical=False, mode='tomo', reduced=False):
    path.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config.optionxform = str
    config.read_string(_INI_TEMPLATE.format(nph=[10000]*3, noisevar='[0., 0., 0.]'))
    if mode == 'scao':
        config['sources_HO'].update(Zenith='[0.]', Azimuth='[0.]', Height='0.')
        config['sensor_HO'].update(NumberPhotons='[10000]', NumberLenslets='[20]', NoiseVariance='[0.]')
    if mode in ('tomographic_alias', 'alias_aware'):
        config['sensor_HO']['GridOffsets'] = '[[-.23,-.23],[.23,-.23],[.23,.23]]'
        config['sensor_HO']['ReconstructionModel'] = "'alias_aware'"
    if mode == 'alias_aware':
        config['sensor_HO']['ReconstructionModel'] = "'alias_aware'"
    if mode == 'polc':
        config['sensor_HO']['ReconstructionModel'] = "'alias_aware'"
        config['sensor_HO']['ExposureTime'] = '0.0'
        config['RTC']['TemporalModel_HO'] = "'polc'"
    if mode == 'open':
        config['RTC']['LoopGain_HO'] = '0.0'
    if mode == 'noisy':
        config['sensor_HO']['NoiseVariance'] = '[.01, .02, .03]'
    with (path/'model.ini').open('w') as stream:
        config.write(stream)
    return fourierModel(str(path/'model.ini'), path_root=_p3_path(), calcPSF=False,
        display=False, computeFocalAnisoCov=False, getErrorBreakDown=not reduced,
        reduce_memory=reduced, computeOpticalPSD=optical)


@pytest.mark.parametrize('mode', ['tomo', 'scao', 'tomographic_alias', 'alias_aware', 'polc', 'open', 'noisy'])
def test_optical_request_preserves_variance_and_reconstruction(tmp_path, mode):
    legacy = _model(tmp_path/'legacy', mode=mode)
    optical = _model(tmp_path/'optical', optical=True, mode=mode)
    assert legacy.opticalPSD is None
    np.testing.assert_array_equal(optical.PSD, legacy.PSD)
    assert optical.opticalPSD.shape == optical.PSD.shape
    assert np.isfinite(optical.opticalPSD).all()
    assert optical.opticalPSD.min() >= -1e-12
    for key in ('W', 'Wtomo', 'Walpha', 'Popt', 'Cphi_mod', 'h1', 'h2', 'hn',
                'wfeST', 'wfeAl', 'wfeN', 'wfeFit'):
        if hasattr(legacy, key):
            np.testing.assert_array_equal(getattr(optical, key), getattr(legacy, key))
    if mode != 'open':
        for key in ('psdSpatioTemporal', 'psdAlias', 'psdNoise', 'psdFit', 'psdDiffRef', 'psdChromatism'):
            np.testing.assert_array_equal(getattr(optical, key), getattr(legacy, key))


@pytest.mark.parametrize('method,precompute', [('chunked', True), ('streaming', True),
                                               ('streaming', False), ('limited', True)])
def test_stock_optical_alias_matches_independent_input_factor_control(tmp_path, method, precompute):
    model = _model(tmp_path)
    actual = model.aliasingPSD(method=method, use_precompute=precompute, piston_filter=False)
    model.clearAliasingPrecompute()

    def unit(diameter, frequency, fm=0, fn=0, dtype=np.float64):
        return np.asarray(frequency != 0, dtype=dtype)

    with patch.object(FourierUtils, 'pistonFilter', unit):
        expected = model.aliasingPSD(method=method, use_precompute=precompute)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


def test_optical_delta_matches_fixed_covariance_control(tmp_path):
    model = _model(tmp_path, optical=True)
    legacy = model.psdSpatioTemporal.copy()
    factor = model.freq.pistonFilterAO_[..., None]
    cphi = model.Cphi.copy()
    model.Cphi = np.divide(cphi, factor, out=np.zeros_like(cphi), where=factor > 0)
    expected_main = model.spatioTemporalPSD()
    model.Cphi = cphi
    np.testing.assert_allclose(model.spatioTemporalPSD(piston_filter=False), expected_main,
                               rtol=2e-13, atol=1e-12)
    units = ((2*model.freq.kcMax_/model.freq.resAO)*model.freq.wvlRef*1e9/(2*np.pi))**2
    delta = (expected_main-legacy
             +(model.aliasingPSD(piston_filter=False)-model.psdAlias)[..., None])*units
    first = int(np.ceil((model.freq.nOtf-model.freq.resAO)/2))
    expected = model.PSD.copy()
    expected[first:first+model.freq.resAO, first:first+model.freq.resAO] += delta
    np.testing.assert_allclose(model.opticalPSD, expected, rtol=2e-13, atol=1e-12)


def test_optical_power_survives_reduced_memory(tmp_path):
    full = _model(tmp_path/'full', optical=True)
    reduced = _model(tmp_path/'reduced', optical=True, reduced=True)
    np.testing.assert_array_equal(reduced.opticalPSD, full.opticalPSD)
    np.testing.assert_array_equal(reduced.PSD, full.PSD)


def test_piston_variance_integral_differs_from_covariance_projection():
    n = 16
    rng = np.random.default_rng(81)
    power = rng.uniform(.1, 1., (n, n))
    reverse = (-np.arange(n)) % n
    power = .5*(power+power[reverse][:, reverse])
    covariance = np.fft.ifft2(power).real
    y, x = np.mgrid[:6, :6]
    amplitude = (((x-2.5)**2+(y-2.5)**2 < 9)
                 & ((x-2.5)**2+(y-2.5)**2 > 1)).astype(float)
    amplitude[1, 2] = .6
    ys, xs = np.nonzero(amplitude)
    weights = amplitude[ys, xs]**2
    weights /= weights.sum()
    matrix = covariance[(ys[:, None]-ys[None]) % n, (xs[:, None]-xs[None]) % n]
    projection = np.eye(len(xs))-np.ones((len(xs), 1))*weights
    removed = projection@matrix@projection.T
    window = np.zeros((n, n)); window[ys, xs] = weights
    variance_factor = 1-abs(np.fft.fft2(window))**2
    exact_variance = weights@np.diag(removed)
    np.testing.assert_allclose(exact_variance, np.sum(power*variance_factor)/n**2,
                               rtol=1e-13, atol=1e-15)

    def structure(c):
        return np.diag(c)[:, None]+np.diag(c)[None]-2*c

    np.testing.assert_allclose(structure(removed), structure(matrix), atol=1e-15)
    filtered = np.fft.ifft2(power*variance_factor).real
    filtered_matrix = filtered[(ys[:, None]-ys[None]) % n, (xs[:, None]-xs[None]) % n]
    assert np.max(abs(structure(filtered_matrix)-structure(removed))) > 1e-4

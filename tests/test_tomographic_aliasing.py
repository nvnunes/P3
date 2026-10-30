"""Alias-only propagation must leave every reconstruction coefficient fixed."""

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import pistonFilter
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _build


def _case(tmp_path, option):
    text = _INI_TEMPLATE.format(nph=[10000]*3, noisevar='[0., 0., 0.]')
    text = text.replace('[sensor_HO]', '[sensor_HO]\n'+option)
    path = tmp_path/'case.ini'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _build(str(path))


@pytest.mark.parametrize('sampling', ['', 'GridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]'])
def test_only_alias_power_changes(tmp_path, sampling):
    native = _case(tmp_path/'native', sampling)
    disabled = _case(tmp_path/'disabled', sampling+'\nTomographicAliasing = False')
    changed = _case(tmp_path/'changed', sampling+'\nTomographicAliasing = True')
    np.testing.assert_array_equal(native.PSD, disabled.PSD)
    for field in ('Wtomo', 'Popt', 'W', 'Walpha', 'Cphi', 'Cphi_mod', 'h1', 'h2', 'hn',
                  'psdSpatioTemporal', 'psdNoise', 'psdDiffRef', 'psdChromatism'):
        np.testing.assert_array_equal(getattr(native, field), getattr(changed, field))
    assert np.isfinite(changed.PSD).all()
    assert not np.array_equal(native.psdAlias, changed.psdAlias)


@pytest.mark.parametrize('offsets', [None, [[.23,-.23]]*3,
                                    [[-.23,-.23],[.23,-.23],[.23,.23]]])
@pytest.mark.parametrize('symmetric', [False, True])
def test_alias_power_matches_independent_cell_integration(tmp_path, offsets, symmetric):
    option = 'TomographicAliasing = True'
    option += f'\nSymmetricAliasing = {symmetric}'
    if offsets is not None:
        option += f'\nGridOffsets = {offsets}'
    fao = _case(tmp_path, option)
    offsets = np.zeros((fao.nGs, 2)) if offsets is None else np.array(offsets)
    ix, iy = fao.freq.resAO//2+2, fao.freq.resAO//2+1
    k = np.array([fao.freq.kxAO_[ix, iy], fao.freq.kyAO_[ix, iy]])
    u = k/np.linalg.norm(k)
    d = fao.ao.wfs.optics[0].dsub
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes, weights = nodes*d/2, weights/2
    wind = np.deg2rad(fao.ao.atm.wDir)
    vx, vy = fao.ao.atm.wSpeed*np.cos(wind), fao.ao.atm.wSpeed*np.sin(wind)
    T = np.mean([w.clock_rate for w in fao.ao.wfs.detector])/fao.ao.rtc.holoop['rate']
    total = 0.
    for m in range(-fao.freq.nTimes, fao.freq.nTimes+int(symmetric)):
        for n in range(-fao.freq.nTimes, fao.freq.nTimes+int(symmetric)):
            if m == n == 0:
                continue
            q = k - np.array([m, n])/d
            gradient = []
            for offset in offsets:
                dx, dy = d*offset
                gx = np.sum(weights*(np.exp(2j*np.pi*(q[0]*(dx+d/2)+q[1]*(dy+nodes)))
                                      - np.exp(2j*np.pi*(q[0]*(dx-d/2)+q[1]*(dy+nodes)))))/d
                gy = np.sum(weights*(np.exp(2j*np.pi*(q[0]*(dx+nodes)+q[1]*(dy+d/2)))
                                      - np.exp(2j*np.pi*(q[0]*(dx+nodes)+q[1]*(dy-d/2)))))/d
                gradient.append(u @ [gx, gy])
            M = np.array(gradient)[:,None]*np.exp(2j*np.pi*(q @ fao.gs.direction)[:, None]
                                      *(fao.ao.atm.heights*fao.strechFactor)[None])
            covariance = (M*fao.ao.atm.weights) @ M.conj().T
            W = fao.W[ix, iy, 0]
            spatial_power = np.real(W @ covariance @ W.conj())
            temporal = np.sum(fao.ao.atm.weights*np.sinc(q[0]*vx*T)*np.sinc(q[1]*vy*T)
                              *np.exp(2j*np.pi*(q[0]*vx+q[1]*vy)*T*fao.ao.rtc.holoop['delay']))
            temporal *= fao.h1[ix, iy]
            spectrum = .0229*fao.ao.atm.r0**(-5/3)*(q @ q+1/fao.ao.atm.L0**2)**(-11/6)
            piston = pistonFilter(fao.ao.tel.D, np.array([[np.linalg.norm(q)]]))[0, 0]
            total += spatial_power*spectrum*piston*abs(temporal)**2
    np.testing.assert_allclose(fao.psdAlias[ix, iy], total, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize('option', ['TomographicAliasing = 1'])
def test_ambiguous_option_is_rejected(tmp_path, option):
    with pytest.raises(ValueError, match='TomographicAliasing'):
        _case(tmp_path, option)

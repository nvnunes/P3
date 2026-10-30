"""Alias-aware tomography retains correlated x/y channels without changing defaults."""

import numpy as np
import pytest

from test_tomographic_aliasing import _case


@pytest.mark.parametrize('sampling', ['', 'GridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]'])
def test_disabled_option_and_prior_are_unchanged(tmp_path, sampling):
    native = _case(tmp_path/'native',sampling+'\nTomographicAliasing = True')
    disabled = _case(tmp_path/'disabled',
        sampling+'\nTomographicAliasing = True\nAliasAwareReconstruction = False')
    aware = _case(tmp_path/'aware',
        sampling+'\nTomographicAliasing = True\nAliasAwareReconstruction = True')
    np.testing.assert_array_equal(native.PSD,disabled.PSD)
    for field in ('Popt','Cphi','Cphi_mod','h1','h2','hn'):
        np.testing.assert_array_equal(getattr(native,field),getattr(aware,field))
    assert aware.W.shape[-1] == 2*native.W.shape[-1]
    assert not np.array_equal(native.W,aware.W)
    assert np.isfinite(aware.PSD).all()
    assert aware.PSD.min() >= 0


@pytest.mark.parametrize('offsets', [None, [[.23,-.23]]*3,
                                    [[-.23,-.23],[.23,-.23],[.23,.23]]])
@pytest.mark.parametrize('symmetric', [False, True])
@pytest.mark.parametrize('noise', [[0.,0.,0.],[.01,.02,.03],[0.,.02,.03],[None]])
def test_covariance_and_estimator_match_direct_sampled_cells(tmp_path, offsets, symmetric, noise):
    option = 'TomographicAliasing = True\nAliasAwareReconstruction = True'
    option += f'\nSymmetricAliasing = {symmetric}'
    if offsets is not None:
        option += f'\nGridOffsets = {offsets}'
    from configparser import ConfigParser
    from test_p3_bugfixes_noise_tomo import _build
    model = _case(tmp_path,option)
    config = ConfigParser();config.optionxform = str
    config.read(tmp_path/'case.ini')
    config['sensor_HO']['NoiseVariance'] = repr(noise)
    with (tmp_path/'noise.ini').open('w') as stream:config.write(stream)
    model = _build(str(tmp_path/'noise.ini'))
    offsets = np.zeros((model.nGs, 2)) if offsets is None else np.array(offsets)
    ij = (model.freq.resAO//2+2,model.freq.resAO//2+1)
    k = np.array([model.freq.kxAO_[ij],model.freq.kyAO_[ij]])
    d = model.ao.wfs.optics[0].dsub
    nodes,weights = np.polynomial.legendre.leggauss(48)
    nodes,weights = nodes*d/2,weights/2
    h = model.atm_mod.heights*model.strechFactor_mod
    channels = 2*model.nGs
    ca = np.zeros((channels,channels),complex)
    from scipy.special import j1
    M0 = None
    for m in range(-model.freq.nTimes,model.freq.nTimes+int(symmetric)):
        for n in range(-model.freq.nTimes,model.freq.nTimes+int(symmetric)):
            q = k-np.array([m,n])/d
            gradient = []
            for offset in offsets:
                dx, dy = d*offset
                gx = (weights @ (np.exp(2j*np.pi*(q[0]*(dx+d/2)+q[1]*(dy+nodes)))
                               -np.exp(2j*np.pi*(q[0]*(dx-d/2)+q[1]*(dy+nodes)))))/d
                gy = (weights @ (np.exp(2j*np.pi*(q[0]*(dx+nodes)+q[1]*(dy+d/2)))
                               -np.exp(2j*np.pi*(q[0]*(dx+nodes)+q[1]*(dy-d/2)))))/d
                gradient.extend([gx,gy])
            M = np.array(gradient)[:,None]*np.exp(2j*np.pi*np.repeat(q @ model.gs.direction,2)[:,None]*h[None])
            if m == 0 and n == 0:
                M0 = M
                continue
            x = np.pi*model.ao.tel.D*np.linalg.norm(q)
            spectral = (.0229*model.ao.atm.r0**(-5/3)
                *(q @ q+1/model.ao.atm.L0**2)**(-11/6)*(1-(2*j1(x)/x)**2))
            ca += (M*(spectral*model.atm_mod.weights)) @ M.conj().T
    actual = model._alias_measurement_covariance(h,model.atm_mod.weights,np.ones(len(h),bool))[ij]
    np.testing.assert_allclose(actual,ca,rtol=3e-12,atol=1e-13)
    np.testing.assert_allclose(actual,actual.conj().T,atol=1e-13)
    assert np.linalg.eigvalsh(actual).min() > -1e-13
    # Target and alias responses are independently integrated in the same axes.
    assert np.linalg.norm(actual[::2,1::2]) > 1e-12
    prior = np.diag(model.Cphi_mod[ij])
    C = M0 @ prior @ M0.conj().T + ca
    floor = model.tomoRelRegFloor*abs(C.diagonal()).max()
    variance = np.repeat(model.ao.wfs.processing.noiseVar,2)
    C += np.diag(np.maximum(variance,floor))
    rhs = model.Popt[ij] @ prior @ M0.conj().T
    expected = np.linalg.solve(C.T, rhs.T).T
    # Noise-free systems amplify cell-quadrature roundoff. Compare the matrix
    # norm and the independently formed normal equations, not near-zero entries.
    assert np.linalg.norm(model.W[ij]-expected)/np.linalg.norm(expected) < 3e-10
    residual = np.linalg.norm(model.W[ij] @ C-rhs)
    assert residual/(np.linalg.norm(model.W[ij])*np.linalg.norm(C)+np.linalg.norm(rhs)) < 1e-14


@pytest.mark.parametrize('sampling', ['', 'GridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]'])
def test_reconstruction_is_independent_of_alias_propagation(tmp_path, sampling):
    options = sampling+'\nAliasAwareReconstruction = True\nSymmetricAliasing = True'
    native = _case(tmp_path/'native',options)
    disabled = _case(tmp_path/'disabled',options+'\nTomographicAliasing = False')
    propagated = _case(tmp_path/'propagated',options+'\nTomographicAliasing = True')
    assert not native.tomographicAliasing and not disabled.tomographicAliasing
    np.testing.assert_array_equal(native.PSD,disabled.PSD)
    for field in ('Wtomo','W','Walpha','Popt','Cphi','Cphi_mod','MPalphaL',
                  'h1','h2','hn','psdSpatioTemporal','psdNoise'):
        np.testing.assert_array_equal(getattr(native,field),getattr(propagated,field))
    stock_alias = (native.freq.mskInAO_*native._aliasing_psd_chunked()
                   *native.ao.atm.r0**(-5/3)*0.0229)
    np.testing.assert_array_equal(native.psdAlias,stock_alias)
    assert not np.array_equal(native.psdAlias,propagated.psdAlias)


@pytest.mark.parametrize('option',[
    'AliasAwareReconstruction = 1\nTomographicAliasing = True',
])
def test_ambiguous_option_fails(tmp_path,option):
    with pytest.raises(ValueError,match='AliasAwareReconstruction'):
        _case(tmp_path,option)

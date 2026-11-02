"""Physical reporting must close without changing finite predictions."""
import numpy as np
import pytest

from test_finite_aperture import small_model
from test_finite_two_stage import coupled_model, sampling


def dense(residual):
    matrix = np.empty((len(residual.diagonal), len(residual.diagonal)))
    for first, block in residual.blocks():
        last = first+len(block)
        matrix[first:last, first:] = block
        matrix[first:, first:last] = block.T
    return matrix


@pytest.mark.parametrize('two_stage', [False, True])
@pytest.mark.parametrize('mean_slopes', [False, True])
def test_budget_closure_projection_and_prediction_parity(tmp_path, capsys, two_stage, mean_slopes):
    native = coupled_model(tmp_path/'case') if two_stage else small_model(tmp_path/'case', True)
    native.removeMeanSlopes = mean_slopes
    sample = sampling(native)
    before, _ = native.finite_aperture_residual(sample, 0)
    covariance = dense(before)
    backend = native._finite_aperture_backend()
    assert not hasattr(backend, '_error_budget')
    weights = backend.weights.copy()
    native.getErrorBreakDown = True
    after, _ = native.finite_aperture_residual(sample, 0)
    np.testing.assert_array_equal(dense(after), covariance)
    np.testing.assert_array_equal(backend.weights, weights)
    record = native.errorBreakDown(verbose=False)[0]
    budget = backend._error_budget
    w = sample.intensity
    scale = (native.freq.wvlRef*1e9/(2*np.pi))**2
    total = (w@np.diag(covariance)-w@covariance@w)*scale
    np.testing.assert_allclose(record.total_nm2, total, rtol=1e-9, atol=1e-7)
    low = dense(budget.low.residual_blocks(sample, 0)[0])
    low_variance = (w@np.diag(low)-w@low@w)*scale
    np.testing.assert_allclose(record.components_nm2['spatio_temporal'], low_variance, rtol=1e-9, atol=1e-7)
    assert record.total_nm2 == record.quadrature_nm2 + record.fitting_aliasing_covariance_nm2
    assert min(record.components_nm2.values()) >= 0
    native.finite_aperture_residual(sample, 0)
    assert native.finiteErrorBreakDown[0] is record
    assert not hasattr(native, 'wfeAl')
    assert record.as_dict()['definition'] == 'tiptop_input_band_v1'
    assert record.as_dict()['total_nm2'] == record.total_nm2
    native.errorBreakDown(verbose=True)
    assert '.Aliasing error:' in capsys.readouterr().out
    with pytest.raises(TypeError):
        record.components_nm2['fitting'] = 0


def test_configuration_change_discards_budget(tmp_path):
    native = coupled_model(tmp_path/'case')
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    old = native._finite_aperture_backend()
    native.my_data_map['RTC_science']['LoopGain_HO'] = .8
    native.finite_aperture_residual(sample, 0)
    assert native._finite_aperture_backend() is not old
    assert native.finiteErrorBreakDown[0] is not old._error_budget.records[0]


@pytest.mark.parametrize('backend_name', ['host', 'mps', 'cuda'])
def test_device_reporting_matches_cpu(tmp_path, backend_name):
    if backend_name == 'mps':
        torch = pytest.importorskip('torch')
        if not torch.backends.mps.is_available():
            pytest.skip('MPS device unavailable')
    elif backend_name == 'cuda':
        cp = pytest.importorskip('cupy')
        if not cp.is_available():
            pytest.skip('CUDA device unavailable')
    from test_finite_device_products import HostProducts
    from p3.aoSystem.finiteErrorBudget import FiniteErrorBudget
    native = coupled_model(tmp_path/'case')
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    expected = native.finiteErrorBreakDown[0]
    if backend_name == 'host':
        backend = native._finite_aperture_backend()
        backend.device = HostProducts(backend, sample)
        residual, _ = backend.residual_blocks(sample, 0)
        actual = FiniteErrorBudget(backend, sample).evaluate(0, residual)
    else:
        native.finiteApertureBackend = backend_name
        native.finite_aperture_residual(sample, 0)
        actual = native.finiteErrorBreakDown[0]
    np.testing.assert_allclose(list(actual.components_nm2.values()),
        list(expected.components_nm2.values()),
        rtol=2e-5 if backend_name == 'mps' else 1e-8, atol=1e-5)
    np.testing.assert_allclose(list(actual.first_stage.components_nm2.values()),
        list(expected.first_stage.components_nm2.values()),
        rtol=2e-5 if backend_name == 'mps' else 1e-8, atol=1e-5)


def test_optical_sampling_change_rebuilds_accounting(tmp_path):
    from tiptop.finite_pupil import FinitePupilSampling
    native = small_model(tmp_path/'case', True)
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    backend = native._finite_aperture_backend()
    old = backend._error_budget
    pupil = native.ao.tel.pupil.copy()
    pupil[:len(pupil)//2] *= .8
    changed = FinitePupilSampling(pupil, sample.n, sample.side, sample.period)
    native.finite_aperture_residual(changed, 0)
    assert backend._error_budget is not old
    assert not hasattr(backend._error_budget.low, '_error_budget')
    assert len(native.finiteErrorBreakDown) == 1


def test_extra_error_counted_once(tmp_path):
    native = small_model(tmp_path/'case', True)
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    first = native.finiteErrorBreakDown[0]
    native.ao.tel.extraErrorNm = 30.
    native.finite_aperture_residual(sample, 0)
    second = native.finiteErrorBreakDown[0]
    for name, value in first.components_nm2.items():
        if name != 'extra_error':
            np.testing.assert_allclose(second.components_nm2[name], value, atol=1e-7)
    assert second.components_nm2['extra_error'] > 0
    np.testing.assert_allclose(second.total_nm2-first.total_nm2,
                               second.components_nm2['extra_error'], atol=1e-7)


def test_disabled_science_matches_common_budget_and_noise_stays_rejected(tmp_path):
    single = small_model(tmp_path/'single', True)
    disabled = coupled_model(tmp_path/'disabled', enabled=False)
    results = []
    for native in (single, disabled):
        native.getErrorBreakDown = True
        with pytest.raises(ValueError, match='optical pupil fields'):
            native.errorBreakDown(verbose=False)
        native.finite_aperture_residual(sampling(native), 0)
        result = native.finiteErrorBreakDown[0]
        assert result.stage == 'ltao'
        assert result.first_stage is None
        results.append(list(result.components_nm2.values()))
    np.testing.assert_array_equal(*results)
    disabled.ao.wfs.processing.noiseVar[0] = .01
    with pytest.raises(ValueError, match='noise'):
        disabled.finite_aperture_residual(sampling(disabled), 0)


@pytest.mark.parametrize('mean_slopes', [False, True])
def test_aliasing_is_actual_above_band_command_power(tmp_path, mean_slopes):
    """Independent plane-wave propagation, including both mirrors and feedback."""
    from scipy.linalg import cho_solve
    native = coupled_model(tmp_path/'case')
    native.removeMeanSlopes = mean_slopes
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    b = native._finite_aperture_backend()
    rows = cho_solve(b.factor, (b.science_surface.fit@b._prior_cross(0)).T).T
    basis = np.concatenate((b.common_basis, b.science_basis), axis=1)
    mean = sample.intensity@basis
    gram = basis.T@(sample.intensity[:, None]*basis)-np.outer(mean, mean)
    mask = (b.qx**2+b.qy**2 >= native.freq.kcMax_**2 if native.ao.dms.AoArea == 'circle'
            else (abs(b.qx) >= native.freq.kcMax_) | (abs(b.qy) >= native.freq.kcMax_))
    ids = np.flatnonzero(mask)
    means = np.kron(np.eye(b.nc), np.ones((b.nv, 1))/np.sqrt(b.nv))
    small = means.T@b.feedback@means
    value = 0.
    for layer, (a, height) in enumerate(zip(b.contraction, b.heights)):
        wind = np.deg2rad(native.ao.atm.wDir[layer])
        for first in range(0, len(ids), 512):
            take = ids[first:first+512]
            q = np.column_stack((b.qx.ravel()[take], b.qy.ravel()[take]))
            u = a*q
            gradient = 2j*np.pi*u*np.prod(np.sinc(b.pitch*u), axis=1)[:, None]
            guide = np.exp(2j*np.pi*(height*q@b.guide.T+u@b.offsets.T))
            cells = np.exp(2j*np.pi*(u@b.cells.T))
            y = (guide[:, :, None, None]*gradient[:, None, :, None]*cells[:, None, None, :]).reshape(len(q), -1).T
            h1 = b.temporal[layer].ravel()[take]
            f = q@np.array([np.cos(wind), np.sin(wind)])*native.ao.atm.wSpeed[layer]
            h2 = b.stage.gain*np.exp(2j*np.pi*f*b.stage.delay/native.ao.rtc.holoop['rate'])
            if mean_slopes:
                filtered = y-means@(means.T@y)
                rhs = means.T@b.feedback@filtered
                reduced = np.linalg.solve(np.eye(b.nc)[None]-h1[:, None, None]*small,
                                          rhs.T[..., None])[..., 0].T
                common = h1*filtered+means@(reduced*h1**2)
            else:
                common = h1*y
            science = h2*(y-b.feedback@common)
            if mean_slopes:
                science -= means@(means.T@science)
            commands = np.concatenate((b.weights@common, rows@science), axis=0)
            power = b.raw[layer].ravel()[take]*b.dk**2
            value += np.sum(power*np.sum(commands.conj()*(gram@commands), axis=0).real)
    scale = (native.freq.wvlRef*1e9/(2*np.pi))**2
    np.testing.assert_allclose(native.finiteErrorBreakDown[0].components_nm2['aliasing'],
                               value*scale, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize('two_stage', [False, True])
def test_static_map_report_is_not_cached_with_dynamic_budget(tmp_path, two_stage):
    from tiptop.finite_pupil import FinitePupilSampling
    native = coupled_model(tmp_path/'case') if two_stage else small_model(tmp_path/'case', True)
    native.getErrorBreakDown = True
    sample = sampling(native)
    native.finite_aperture_residual(sample, 0)
    dynamic = native.finiteErrorBreakDown[0]
    y, x = np.indices(native.ao.tel.pupil.shape)
    phase = np.sin(x*.3)*np.cos(y*.2)
    static = FinitePupilSampling(native.ao.tel.pupil, sample.n, sample.side,
                                sample.period, static_phase=phase)
    native.finite_aperture_residual(static, 0)
    record = native.finiteErrorBreakDown[0]
    assert record.ncpa_nm2 > 0
    assert record.dynamic_ho_nm2 == dynamic.dynamic_ho_nm2
    assert record.total_nm2 == record.dynamic_ho_nm2+record.ncpa_nm2
    if two_stage:
        assert record.first_stage.ncpa_nm2 == record.ncpa_nm2
        assert record.first_stage.dynamic_ho_nm2 == dynamic.first_stage.dynamic_ho_nm2
    native.finite_aperture_residual(sample, 0)
    assert native.finiteErrorBreakDown[0].ncpa_nm2 == 0
    if two_stage:
        assert native.finiteErrorBreakDown[0].first_stage.ncpa_nm2 == 0


@pytest.mark.parametrize('mean_slopes', [False, True])
def test_first_stage_matches_independent_science_disabled_control(tmp_path, mean_slopes):
    """The LTAO column preserves actual common correction at every direction."""
    joint = coupled_model(tmp_path/'joint')
    single = coupled_model(tmp_path/'single', enabled=False)
    for model in (joint, single):
        model.removeMeanSlopes = mean_slopes
        model.getErrorBreakDown = True
        model.ao.tel.extraErrorNm = 20.
    for field in range(joint.ao.src.nSrc):
        sample = sampling(single)
        residual, _ = single.finite_aperture_residual(sample, field)
        covariance = dense(residual)
        joint.finite_aperture_residual(sampling(joint), field)
        expected = single.finiteErrorBreakDown[field]
        record = joint.finiteErrorBreakDown[field]
        actual = record.first_stage
        np.testing.assert_allclose(list(actual.components_nm2.values()),
                                   list(expected.components_nm2.values()), rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(actual.fitting_aliasing_covariance_nm2,
                                   expected.fitting_aliasing_covariance_nm2, rtol=1e-8, atol=1e-5)
        scale = (single.freq.wvlRef*1e9/(2*np.pi))**2
        w = sample.intensity
        direct = (w@np.diag(covariance)-w@covariance@w)*scale
        np.testing.assert_allclose(actual.dynamic_ho_nm2, direct, rtol=1e-8, atol=1e-6)
        assert record.as_dict()['first_stage']['total_nm2'] == actual.total_nm2

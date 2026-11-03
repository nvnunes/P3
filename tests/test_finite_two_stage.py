"""Coupled finite measurements checked against explicit shared-wave sums."""
from configparser import ConfigParser

import numpy as np
import pytest

from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.finiteAperture import _fourier_sum, _stationary
from p3.aoSystem.finiteTwoStage import FiniteTwoStageReconstructor
from p3.aoSystem.finiteCovariance import _fourier_sum_many
from test_finite_aperture import small_model
from test_p3_bugfixes_noise_tomo import _p3_path


def coupled_model(path, enabled=True, covariance_backend="numpy"):
    first=small_model(path, True)
    config = ConfigParser(); config.optionxform = str
    config.read(path/'system.ini')
    from _finite_geometry import write_geometry
    geometry=path/'science.fits';write_geometry(geometry,len(first.ao.tel.pupil),3)
    config['DM_science'] = dict(DmPitchs='[2.0]',NumberActuators='[5]',
        DmHeights='[0.]',InfModel="'gaussian'",InfCoupling='[0.25]',
        PathInfGeometry=repr(str(geometry)))
    config['RTC_science'] = dict(LoopGain_HO='1.0' if enabled else '0.0',LoopDelaySteps_HO='1')
    with (path/'system.ini').open('w') as stream: config.write(stream)
    return fourierModel(str(path/'system.ini'), path_root=_p3_path(), calcPSF=False,
        display=False, verbose=False, reduce_memory=False, computeOpticalPSD=True,
        computeFocalAnisoCov=False, finiteApertureBackend=covariance_backend)


def sampling(model):
    from tiptop.finite_pupil import FinitePupilSampling
    n = model.freq.nOtf
    side = int(2*np.round(float(model.ao.tel.D)/2*n*model.freq.PSDstep))
    period = 1/float(model.freq.kxAO_[1,0]-model.freq.kxAO_[0,0])
    return FinitePupilSampling(model.ao.tel.pupil, n, side, period)


def test_batched_transform_is_same_calculation():
    rng = np.random.default_rng(42)
    spectra = rng.normal(size=(4,16,16))+1j*rng.normal(size=(4,16,16))
    actual = _fourier_sum_many(spectra,.13,np.array([-.7,1.8]),.09,7)
    expected = np.array([_fourier_sum(s,.13,np.array([-.7,1.8]),.09,7) for s in spectra])
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)


def test_joint_moments_against_explicit_atmospheric_waves(tmp_path):
    model = coupled_model(tmp_path/'case')
    backend = FiniteTwoStageReconstructor(model)
    channels = backend.nc*backend.nv
    caa = np.zeros((channels,channels)); cbb = caa.copy(); cab = caa.copy()
    for layer,(a,h) in enumerate(zip(backend.contraction,backend.heights)):
        q = np.column_stack((backend.qx.ravel(),backend.qy.ravel()))
        u = a*q
        cell = np.sinc(backend.pitch*u[:,0])*np.sinc(backend.pitch*u[:,1])
        grad = 2j*np.pi*u*cell[:,None]
        guide = np.exp(2j*np.pi*(h*q@backend.guide.T+u@backend.offsets.T))
        lenslet = np.exp(2j*np.pi*(u@backend.cells.T))
        measurement = (guide[:,:,None,None]*grad[:,None,:,None]*lenslet[:,None,None,:]).reshape(len(q),channels).T
        h1 = backend.temporal[layer].ravel()
        wind = np.deg2rad(model.ao.atm.wDir[layer])
        frequency = -(q[:,0]*np.cos(wind)+q[:,1]*np.sin(wind))*model.ao.atm.wSpeed[layer]
        h2 = np.exp(-2j*np.pi*frequency/model.ao.rtc.holoop['rate'])
        common = measurement*h1
        residual = h2*(measurement-backend.feedback@common)
        density = backend.raw[layer].ravel()*backend.dk**2
        caa += ((common*density)@common.conj().T).real
        cbb += ((residual*density)@residual.conj().T).real
        cab += ((common*density)@residual.conj().T).real
    for actual,expected in ((backend.caa,caa),(backend.cbb,cbb),(backend.cab,cab)):
        np.testing.assert_allclose(actual,expected,rtol=5e-10,atol=2e-11)
    assert np.linalg.eigvalsh(np.block([[caa,cab],[cab.T,cbb]])).min() > -1e-10
    np.testing.assert_allclose(backend.feedback,
        backend.common_surface.slopes(backend.cells,backend.pitch,backend.offsets)@backend.weights,
        rtol=2e-12,atol=2e-12)


def test_real_synthesis_and_residual_factorization(tmp_path):
    model = coupled_model(tmp_path/'case'); sample = sampling(model)
    backend = FiniteTwoStageReconstructor(model)
    model.removeMeanSlopes=False  # Isolate the unprojected covariance identity.
    residual,_ = backend.residual_blocks(sample,0)
    common,science,product = backend._query_crosses(sample,0)
    o1,o2 = backend.operator,backend.science_operator(sample,0)
    expected = backend.source.copy()+backend.correction
    expected += _stationary(backend._independent_power(0),backend.aux_kernels,sample,sample.n)
    expected -= common@o1.T+o1@common.T
    expected += o2@backend.cbb@o2.T+o1@backend.cab@o2.T+o2@backend.cab.T@o1.T
    cross = science-product@backend.feedback.T
    expected -= cross@o2.T+o2@cross.T
    dense,_ = backend.residual_covariance(sample,0)
    np.testing.assert_allclose(dense,expected,rtol=3e-10,atol=3e-11)
    assert np.linalg.eigvalsh(dense).min() > -1e-7
    psf,_ = sample.psf_blocks(residual.diagonal,residual.blocks())
    np.testing.assert_allclose(psf.sum(),1,atol=2e-12)


def test_disabled_science_preserves_current_finite_ltao(tmp_path):
    model = coupled_model(tmp_path/'disabled',False)
    reference = small_model(tmp_path/'reference',True)
    sample = sampling(model)
    actual,_ = model.finite_aperture_covariance(sample,0)
    expected,_ = reference.finite_aperture_covariance(sample,0)
    from p3.aoSystem.finiteActuator import FiniteActuatorReconstructor
    assert type(model._finiteApertureReconstructor) is FiniteActuatorReconstructor
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(model._finiteApertureReconstructor.weights,reference._finiteApertureReconstructor.weights)


def require_device(name):
    if name == 'mps':
        torch = pytest.importorskip('torch')
        if not torch.backends.mps.is_available():
            pytest.skip('Metal is unavailable')
    else:
        cupy = pytest.importorskip('cupy')
        try:
            available = cupy.is_available()
        except cupy.cuda.runtime.CUDARuntimeError:
            available = False
        if not available:
            pytest.skip('CUDA is unavailable')


@pytest.mark.parametrize('device_name', ['cuda', 'mps'])
def test_device_matches_cpu_queries_covariance_and_psf(tmp_path, device_name):
    require_device(device_name)
    model = coupled_model(tmp_path/'case'); sample = sampling(model)
    backend = FiniteTwoStageReconstructor(model)
    cpu,_ = backend.residual_covariance(sample,0)
    cpu_psf,_ = sample.psf(cpu)
    expected = backend._query_crosses(sample,0)
    device_model = coupled_model(tmp_path/'device', covariance_backend=device_name)
    device = device_model._finite_aperture_backend()
    gpu,_ = device.residual_covariance(sample,0)
    actual = device.device.to_host(device._query_crosses(sample,0))
    if device_name == 'mps':
        np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=2e-5)
        np.testing.assert_allclose(gpu,cpu,rtol=3e-5,atol=3e-5)
    else:
        np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(gpu,cpu,rtol=2e-10,atol=2e-10)
    gpu_psf,_ = sample.psf(gpu)
    assert np.abs(gpu_psf-cpu_psf).sum() < (1e-4 if device_name == 'mps' else 1e-10)
    first,_ = device.residual_blocks(sample,0)
    snapshot = [(i,b.copy()) for i,b in first.blocks(19)]
    device.residual_blocks(sample,0)
    for (i,expected),(j,actual) in zip(snapshot,first.blocks(19)):
        assert i == j
        np.testing.assert_array_equal(actual,expected)


def test_field_residual_owns_directional_arrays_and_reuses_shared_state(tmp_path):
    model = coupled_model(tmp_path/'case'); sample = sampling(model)
    backend = FiniteTwoStageReconstructor(model)
    first,_ = backend.residual_blocks(sample,0)
    snapshot = [(i,b.copy()) for i,b in first.blocks(19)]
    common = backend.operator
    backend.residual_blocks(sample,0)
    assert common is backend.operator
    for (i,expected),(j,actual) in zip(snapshot,first.blocks(19)):
        assert i==j
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('count,size',[(31,7),(32,7),(32,16)])
def test_batched_fourier_preserves_complex_finite_sum(count,size):
    rng=np.random.default_rng(17)
    spectra=rng.normal(size=(4,count,count))+1j*rng.normal(size=(4,count,count))
    start=np.array([-.73,.84]);dk=.13;step=.19
    frequency=(np.arange(count)-count//2)*dk
    positions=start[:,None]+np.arange(size)[None]*step
    transforms=np.exp(2j*np.pi*positions[:,:,None]*frequency[None,None])
    expected=np.array([transforms[0]@s@transforms[1].T for s in spectra])
    actual=_fourier_sum_many(spectra,dk,start,step,size)
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-11)

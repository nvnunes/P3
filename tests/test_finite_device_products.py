"""Device arithmetic parity and explicit optional dependency failures."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from p3.aoSystem.finiteTwoStageProducts import DeviceFiniteProducts
from p3.aoSystem.finiteTwoStageCupy import CupyFiniteProducts
from test_finite_two_stage import coupled_model, sampling


class HostProducts(DeviceFiniteProducts):
    """Exercise shared device algebra on CPU; this does not validate CUDA."""
    def upload(self, array, integer=False):
        return np.array(array, dtype=np.int64 if integer else np.float64)

    def zeros(self, shape):
        return np.zeros(shape)

    def empty(self, shape):
        return np.empty(shape)

    def to_host(self, array):
        return np.array(array)


@pytest.mark.parametrize('project_means', [False, True])
def test_shared_device_arithmetic_matches_cpu(tmp_path, project_means):
    value = coupled_model(tmp_path/'case')
    value.removeMeanSlopes = project_means
    sample = sampling(value)
    backend = value._finite_aperture_backend()
    expected,_ = backend.residual_covariance(sample,0)
    crosses = backend._query_crosses(sample,0)
    expected_psf,_ = sample.psf(expected)
    backend.device = HostProducts(backend,sample)
    actual,_ = backend.residual_covariance(sample,0)
    np.testing.assert_allclose(backend._query_crosses(sample,0),crosses,rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)
    actual_psf,_ = sample.psf(actual)
    assert np.abs(actual_psf-expected_psf).sum() < 1e-10


@pytest.mark.parametrize('fits',[False,True])
def test_pinned_coefficients_release_only_redundant_host_planes(tmp_path,fits):
    value=coupled_model(tmp_path/'case');sample=sampling(value)
    backend=value._finite_aperture_backend()
    expected,_=backend.residual_covariance(sample,0)
    owner=HostProducts(backend,sample)
    layers=backend.interpolation+backend.raw_interpolation
    fit_arrays=[layer.fit for layer in layers]
    phase_bytes=sum(layer.phase.nbytes for layer in layers)
    assert all(not np.shares_memory(layer.phase,layer.fit) for layer in layers)
    owner.kernel_cache_info['limit_bytes']=phase_bytes if fits else 1
    assert owner.pin_kernel_pairs() is fits
    for layer,fit in zip(layers,fit_arrays):
        assert layer.fit is fit and fit.dtype==np.float64
        assert (layer.phase is None) is fits
    if fits:
        assert owner.kernel_cache_info['released_host_bytes']==phase_bytes
        with pytest.raises(RuntimeError,match='pinned device cache'):
            layers[0].host_phase()
        assert owner.pin_kernel_pairs()
    backend.device=owner
    actual,_=backend.residual_covariance(sample,0)
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)
    if fits:
        misses=owner.kernel_cache_info['misses']
        backend.residual_blocks(sample,0)
        assert owner.kernel_cache_info['misses']==misses


def test_cupy_missing_is_an_explicit_error(monkeypatch):
    monkeypatch.setitem(sys.modules,'cupy',None)
    with pytest.raises(RuntimeError,match='requires CuPy with CUDA support'):
        CupyFiniteProducts(None,None)


@pytest.mark.parametrize('driver_error',[False,True])
def test_cupy_without_device_is_an_explicit_error(monkeypatch, driver_error):
    class DriverError(Exception):
        pass

    def available():
        if driver_error:
            raise DriverError('driver unavailable')
        return False

    module = SimpleNamespace(is_available=available,
        cuda=SimpleNamespace(runtime=SimpleNamespace(CUDARuntimeError=DriverError)))
    monkeypatch.setitem(sys.modules,'cupy',module)
    with pytest.raises(RuntimeError,match='requires an available CUDA device'):
        CupyFiniteProducts(None,None)


def test_backend_choice_rebuilds_cached_reconstructor(tmp_path):
    value = coupled_model(tmp_path/'case')
    cpu = value._finite_aperture_backend()
    value.finiteApertureBackend = 'cuda'
    gpu = value._finite_aperture_backend()
    assert gpu is not cpu and gpu.array_backend == 'cuda'
    assert value._finite_aperture_backend() is gpu
    value.finiteApertureBackend = 'numpy'
    restored = value._finite_aperture_backend()
    assert restored is not gpu
    np.testing.assert_array_equal(restored.weights,cpu.weights)


def test_global_gpu_arrays_rejected_before_numpy_operations():
    from p3.aoSystem.finiteAperture import validate_native
    native = SimpleNamespace(freq=SimpleNamespace(kxAO_=object()))
    with pytest.raises(ValueError,match='P3_DISABLE_GPU=1'):
        validate_native(native)


def test_failed_single_stage_cuda_preparation_is_not_cached(tmp_path,monkeypatch):
    from p3.aoSystem import finiteTwoStageCupy
    native=coupled_model(tmp_path/'case',enabled=False,covariance_backend='cuda')
    sample=sampling(native);backend=native._finite_aperture_backend();attempts=[]
    def unavailable(*args):
        attempts.append(True)
        raise RuntimeError('unavailable test device')
    monkeypatch.setattr(finiteTwoStageCupy,'CupyFiniteProducts',unavailable)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='unavailable test device'):
            backend.prepare_optics(sample,dense_source=False)
    assert len(attempts)==2

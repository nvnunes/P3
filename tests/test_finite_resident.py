"""Complete device covariance and batched feedback retain the native algebra."""
import numpy as np
import pytest

from p3.aoSystem.finiteMeanSlopes import _fold_many
from p3.aoSystem.finiteAperture import _fold
from test_finite_two_stage import coupled_model,sampling
from test_finite_device_products import HostProducts


@pytest.mark.parametrize('count,size',[(5,8),(11,8)])
def test_bounded_fold_batch_matches_individual_physical_folds(count,size):
    rng=np.random.default_rng(19)
    values=rng.normal(size=(6,count,count))+1j*rng.normal(size=(6,count,count))
    expected=np.array([_fold(v,size) for v in values])
    np.testing.assert_array_equal(_fold_many(values,size),expected)


@pytest.mark.parametrize('means',[False,True])
def test_resident_covariance_retains_all_updates_and_reuses_background(tmp_path,means):
    native=coupled_model(tmp_path/'case')
    native.removeMeanSlopes=means
    sample=sampling(native)
    backend=native._finite_aperture_backend()
    backend.residual_blocks(sample,0)
    backend.device=HostProducts(backend,sample)
    residual,_=backend.residual_blocks(sample,0)
    resident=backend.device.resident_covariance(residual)
    quadratic=0.
    for first,expected in residual.blocks(rows=13):
        last=first+len(expected)
        actual=resident.block_array(first,last)
        np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-11)
        upper=np.arange(first,len(residual.diagonal))[None]>=np.arange(first,last)[:,None]
        weight=expected*sample.intensity[first:last,None]*sample.intensity[None,first:]
        quadratic+=2*weight[upper].sum()-np.sum(np.diag(expected[:,:last-first])*sample.intensity[first:last]**2)
    expected_variance=sample.intensity@residual.diagonal-quadratic
    np.testing.assert_allclose(resident.variance(),expected_variance,rtol=2e-11,atol=2e-11)
    assert resident.background_prepared
    again,_=backend.residual_blocks(sample,0)
    second=backend.device.resident_covariance(again)
    assert second.background is resident.background
    assert not second.background_prepared
    # A new moment array must invalidate the background even on the same owner.
    base=again
    while hasattr(base,'base'):base=base.base
    base.source_terms=(base.source_terms[0]+.01,*base.source_terms[1:])
    third=backend.device.resident_covariance(again)
    assert third.background is not second.background

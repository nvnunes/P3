"""Exact command-space covariance retains native reporting and cache contracts."""
import numpy as np
import pytest

from p3.aoSystem.finiteCovariance import CommandCovariance
from p3.aoSystem.finiteAperture import _ScienceMeasurementQuery
from test_finite_two_stage import coupled_model,sampling,require_device


def test_command_covariance_retains_complete_blocks_diagonal_and_quadratic():
    rng=np.random.default_rng(31)
    basis=rng.normal(size=(31,5));root=rng.normal(size=(5,5))
    covariance=root@root.T
    factored=CommandCovariance(basis,covariance)
    expected=(basis@covariance)@basis.T
    np.testing.assert_allclose(factored.diagonal(),np.diag(expected),rtol=2e-13,atol=2e-13)
    np.testing.assert_array_equal(factored[3:17,8:],expected[3:17,8:])
    weight=rng.random(31);weight/=weight.sum()
    actual=weight@factored.diagonal()-weight@factored[:,:]@weight
    np.testing.assert_allclose(actual,weight@np.diag(expected)-weight@expected@weight,rtol=2e-13)
    assert factored.basis is basis
    assert factored.nbytes==31*5*8<expected.nbytes


def test_cpu_common_covariance_keeps_dense_behavior(tmp_path):
    model=coupled_model(tmp_path/'case');sample=sampling(model)
    backend=model._finite_aperture_backend();backend.residual_blocks(sample,0)
    assert isinstance(backend.correction,np.ndarray)
    expected=(backend.operator@backend.cov_raw)@backend.operator.T
    np.testing.assert_array_equal(backend.correction,expected)
    backend.array_backend='cuda'
    np.testing.assert_array_equal(backend._common_correction_covariance(),expected)


def test_metal_pinned_kernels_preserve_native_common_covariance(tmp_path):
    require_device('mps')
    model=coupled_model(tmp_path/'case',covariance_backend='mps');sample=sampling(model)
    backend=model._finite_aperture_backend();backend.residual_blocks(sample,0)
    expected=(backend.operator@backend.cov_raw)@backend.operator.T
    np.testing.assert_allclose(backend.correction[:,:],expected,rtol=2e-11,atol=2e-11)
    assert backend.device.kernel_cache_info['released_host_bytes']>0
    assert all(layer.phase is None for family in (backend.interpolation,backend.raw_interpolation) for layer in family)


def test_grouped_cpu_queries_share_only_identical_operators():
    x,y=np.meshgrid(np.linspace(-3,3,7),np.linspace(-2,2,5))
    points=np.column_stack((x.ravel(),y.ravel()))
    points=points[np.linalg.norm(points,axis=1)>.6]
    cells=points[::3]
    query=_ScienceMeasurementQuery(points,cells,32,8.)
    kernels=np.random.default_rng(5).normal(size=(2,32,32))
    origins=np.array([[-.3,.271],[.4,.271],[.9,-.351],[1.2,-.351+2**-40]])
    expected=np.empty((2,4,len(points),len(cells)));actual=np.empty_like(expected)
    for guide,origin in enumerate(origins):query.evaluate(kernels,origin,.81,expected[:,guide])
    query.evaluate_group(kernels,origins,.81,actual)
    np.testing.assert_array_equal(actual,expected)
    assert query.grouped_query_stats['requested_y_products']==8
    assert query.grouped_query_stats['computed_y_products']==6


def test_directional_prior_cache_and_grouping_preserve_individual_queries(tmp_path):
    native=coupled_model(tmp_path/'case');backend=native._finite_aperture_backend()
    expected=np.zeros((len(backend.phase_fit_points),backend.nc*backend.nv))
    values=np.empty((2,len(backend.phase_fit_points),backend.nv))
    beta=native.ao.src.direction[:,0]
    for layer,(a,h) in enumerate(zip(backend.contraction,backend.heights)):
        for guide in range(backend.ng):
            origin=h*(beta-backend.guide[guide])-a*backend.offsets[guide]
            backend.prior_query.evaluate(backend.prior_interpolation[layer],origin,a,values)
            for axis in range(2):
                expected[:,(2*guide+axis)*backend.nv:(2*guide+axis+1)*backend.nv]+=values[axis]
    actual=backend._prior_cross(0)
    np.testing.assert_array_equal(actual,expected)
    assert backend._zero_height_prior
    before=dict(backend.prior_query.grouped_query_stats)
    np.testing.assert_array_equal(backend._prior_cross(0),expected)
    assert backend.prior_query.grouped_query_stats['calls']>before['calls']

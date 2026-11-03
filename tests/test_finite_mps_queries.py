"""Direct valid-pair Metal queries preserve the existing periodic cubic sum."""
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from p3.aoSystem.finiteAperture import _ScienceMeasurementQuery
from p3.aoSystem.finiteTwoStageMps import MpsFiniteProducts
from p3.aoSystem.finiteMpsQueries import FusedCubicQuery


@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('contraction',[1.,.83])
def test_fused_valid_pairs_match_torch_and_double_precision_queries(batch,contraction):
    if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
        pytest.skip('Optional Metal shader API unavailable')
    x,y=np.meshgrid(np.linspace(-4,4,11),np.linspace(-3,3,9))
    xy=np.column_stack((x.ravel(),y.ravel()))
    radius=np.linalg.norm(xy,axis=1);xy=xy[(radius<4)&(radius>.6)]
    x,y=np.meshgrid(np.linspace(-3,3,5),np.linspace(-2,2,4))
    cells=np.column_stack((x.ravel(),y.ravel()))
    cells=cells[np.linalg.norm(cells,axis=1)<3.5]
    owner=object.__new__(MpsFiniteProducts);owner.torch=torch
    owner.query=_ScienceMeasurementQuery(xy,cells,32,8.)
    owner.sampling=SimpleNamespace(xy=xy);owner.backend=SimpleNamespace(cells=cells)
    owner.blocks=[(y,owner.upload(points,True),owner.upload(indices,True))
                  for y,points,indices in owner.query.blocks]
    coefficients=np.random.default_rng(7).normal(size=(2,32,32))
    kernels=owner.upload(coefficients)
    origins=np.array([[-9.3,13.7],[.15,-.22],[60.,-61.]])[:batch]
    output=owner.empty((2,batch,len(xy),len(cells)))
    owner._fused_query=None
    owner.evaluate_many(kernels,origins,contraction,output)
    previous=output.cpu().numpy().copy()
    owner._fused_query=FusedCubicQuery(owner)
    owner.evaluate_many(kernels,origins,contraction,output)
    actual=output.cpu().numpy().copy()
    np.testing.assert_allclose(actual,previous,rtol=2e-5,atol=5e-7)
    expected=np.empty((2,len(xy),len(cells)))
    for field,origin in enumerate(origins):
        owner.query.evaluate(coefficients,origin,contraction,expected)
        np.testing.assert_allclose(actual[:,field],expected,rtol=2e-5,atol=5e-7)
    # Identical direction stencils are reused without coupling field outputs.
    stencils=owner._fused_query._stencils
    owner.evaluate_many(kernels,origins,contraction,output)
    assert owner._fused_query._stencils is stencils
    np.testing.assert_array_equal(output.cpu().numpy(),actual)

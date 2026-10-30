"""Sampling offsets preserve stock tomography's single-frequency contract."""

import numpy as np
import pytest

from p3.aoSystem.fourierModel import fourierModel
from test_tomographic_aliasing import _case


def test_zero_offsets_reproduce_stock_exactly(tmp_path):
    stock = _case(tmp_path/'stock', '')
    zero = _case(tmp_path/'zero', 'GridOffsets = [[0, 0]]*3')
    for field in ('W', 'Walpha', 'Cphi', 'Cphi_mod', 'psdAlias', 'PSD'):
        np.testing.assert_array_equal(getattr(stock, field), getattr(zero, field))


def test_relative_offsets_change_only_measurement_reference(tmp_path, monkeypatch):
    # Check phase covariance away from the native nearly singular solve limit.
    monkeypatch.setattr(fourierModel, 'tomoRelRegFloor', 1e-6)
    stock = _case(tmp_path/'stock', '')
    shifts = np.array([[-.23, -.23], [.23, -.23], [.23, .23]])
    shifted = _case(tmp_path/'shifted', f'GridOffsets = {shifts.tolist()}')
    pitch = np.array([w.dsub for w in shifted.ao.wfs.optics])
    phase = np.exp(2j*np.pi*pitch*(shifted.freq.kxAO_[..., None]*shifts[:, 0]
                                  + shifted.freq.kyAO_[..., None]*shifts[:, 1]))
    np.testing.assert_allclose(shifted.W, stock.W*phase.conj()[..., None, :],
                               rtol=3e-9, atol=2e-9)
    np.testing.assert_allclose(shifted.Walpha, stock.Walpha, rtol=3e-9, atol=2e-9)
    np.testing.assert_array_equal(shifted.psdAlias, stock.psdAlias)
    np.testing.assert_array_equal(shifted.Cphi, stock.Cphi)
    assert shifted.reconstructionModel == 'standard'
    assert not shifted.tomographicAliasing
    assert not shifted.aliasAwareReconstruction


@pytest.mark.parametrize('option', [
    'GridOffsets = [[0, 0]]',
    'GridOffsets = [[0, 0], [0, 0], [float("nan"), 0]]',
    'GridOffsets = [[float("inf"), 0]]*3',
])
def test_unsupported_offsets_fail_explicitly(tmp_path, option):
    with pytest.raises(ValueError, match='GridOffsets'):
        _case(tmp_path, option)

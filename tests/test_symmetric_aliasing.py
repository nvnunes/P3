"""Symmetric alias truncation is optional and shared by both error models."""

import numpy as np
import pytest

from test_tomographic_aliasing import _case


@pytest.mark.parametrize('options', ['', 'TomographicAliasing = True'])
def test_default_alias_cutoff_is_preserved(tmp_path, options):
    native = _case(tmp_path/'native', options)
    explicit = _case(tmp_path/'explicit', options+'\nSymmetricAliasing = False')
    np.testing.assert_array_equal(native.PSD, explicit.PSD)
    np.testing.assert_array_equal(native.W, explicit.W)


@pytest.mark.parametrize('options', ['', 'TomographicAliasing = True'])
def test_symmetric_cutoff_preserves_fixed_reconstruction(tmp_path, options):
    native = _case(tmp_path/'native', options)
    symmetric = _case(tmp_path/'symmetric', options+'\nSymmetricAliasing = True')
    np.testing.assert_array_equal(native.W, symmetric.W)
    np.testing.assert_array_equal(native.psdSpatioTemporal, symmetric.psdSpatioTemporal)
    np.testing.assert_allclose(symmetric._alias_shift_grid(), -symmetric._alias_shift_grid()[::-1])
    assert np.all(symmetric.psdAlias >= native.psdAlias-1e-12)
    # Changing the option must invalidate stock alias precomputations too.
    native.symmetricAliasing = True
    np.testing.assert_array_equal(native.aliasingPSD(), symmetric.psdAlias)


def test_invalid_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='SymmetricAliasing'):
        _case(tmp_path, 'SymmetricAliasing = 1')

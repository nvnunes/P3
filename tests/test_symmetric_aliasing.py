"""Symmetric alias truncation is optional and shared by both error models."""

import numpy as np
import pytest

from test_tomographic_aliasing import _case


@pytest.mark.parametrize('options', ['', 'TomographicAliasing = True',
                                     'TomographicAliasing = True\nAliasAwareReconstruction = True'])
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


def test_reconstruction_covariance_respects_conjugacy(tmp_path):
    model = _case(tmp_path, 'TomographicAliasing = True\nAliasAwareReconstruction = True\n'
                           'SymmetricAliasing = True\nGridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]')
    covariance = model._alias_measurement_covariance(model.atm_mod.heights*model.strechFactor_mod,
        model.atm_mod.weights, model._sensed_layers_mask(model.atm_mod.heights))
    # Native frequency arrays have a tiny nonzero origin. Use a nondegenerate
    # conjugate pair and allow only the corresponding phase-roundoff error.
    center = model.freq.resAO//2
    np.testing.assert_allclose(covariance[center+2,center+3],
                               covariance[center-2,center-3].conj(), rtol=2e-7, atol=1e-10)


def test_invalid_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='SymmetricAliasing'):
        _case(tmp_path, 'SymmetricAliasing = 1')

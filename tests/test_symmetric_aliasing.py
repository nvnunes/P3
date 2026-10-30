"""Alias-aware reconstruction owns symmetric alias coverage."""

import numpy as np
import pytest

from test_tomographic_aliasing import _case


def test_reconstruction_covariance_respects_conjugacy(tmp_path):
    model = _case(tmp_path, "ReconstructionModel = 'alias_aware'\n"
                           'GridOffsets = [[-.23,-.23],[.23,-.23],[.23,.23]]')
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

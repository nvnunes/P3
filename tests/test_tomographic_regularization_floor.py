"""A per-model INI floor preserves the default and legacy numerical rule."""

import numpy as np
import pytest

from p3.aoSystem.fourierModel import fourierModel
from test_tomographic_aliasing import _case


def test_explicit_default_matches_omitted_option(tmp_path):
    default = _case(tmp_path/'default', '')
    explicit = _case(tmp_path/'explicit', 'TomographicRegularizationFloor = 1e-12')
    assert default.tomoRelRegFloor == explicit.tomoRelRegFloor == 1e-12
    for field in ('W', 'Walpha', 'psdAlias', 'PSD'):
        np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field))


def test_ini_matches_process_override_without_leaking(tmp_path, monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setattr(fourierModel, 'tomoRelRegFloor', 1e-6)
        legacy = _case(tmp_path/'legacy', '')
    configured = _case(tmp_path/'configured', 'TomographicRegularizationFloor = 1e-6')
    assert configured.tomoRelRegFloor == 1e-6
    for field in ('W', 'Walpha', 'psdAlias', 'PSD'):
        np.testing.assert_array_equal(getattr(legacy, field), getattr(configured, field))
    subsequent = _case(tmp_path/'subsequent', '')
    assert subsequent.tomoRelRegFloor == fourierModel.tomoRelRegFloor == 1e-12


@pytest.mark.parametrize('value', ['-1', 'True', 'None', "'1e-6'", 'float("nan")', 'float("inf")'])
def test_invalid_floor_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='TomographicRegularizationFloor'):
        _case(tmp_path, f'TomographicRegularizationFloor = {value}')


def test_zero_disables_relative_floor(tmp_path):
    model = _case(tmp_path, 'TomographicRegularizationFloor = 0')
    assert model.tomoRelRegFloor == 0
    assert np.isfinite(model.W).all()

"""The scalar candidate rejects unsupported research settings explicitly."""

import pytest

from test_tomographic_aliasing import _case


@pytest.mark.parametrize('option,name', [
    ('SuperResolution = True', 'SuperResolution'),
    ('AliasAwareReconstruction = True\nSuperResolution = True', 'SuperResolution'),
    ("WfsSlopeMode = 'cartesian'", 'WfsSlopeMode'),
    ('PhaseSamplesPerSubaperture = 18', 'PhaseSamplesPerSubaperture'),
    ('AliasOrder = 2', 'AliasOrder'),
    ("DiagnosticAbsoluteFloor = '/tmp/floor.npy'", 'DiagnosticAbsoluteFloor'),
])
def test_unsupported_sampling_settings_fail(tmp_path, option, name):
    with pytest.raises(ValueError, match=name):
        _case(tmp_path, option)


def test_scalar_defaults_remain_available(tmp_path):
    model = _case(tmp_path, "SuperResolution = False\nWfsSlopeMode = 'radial'\n"
                            'PhaseSamplesPerSubaperture = None')
    assert not model.superResolution
    assert not hasattr(model, 'sampledWfs')
    assert model.W.shape[-1] == model.nGs
    assert not hasattr(model, 'wfsSlopeMode')
    assert not hasattr(model, 'diagnosticAppliedFloor')

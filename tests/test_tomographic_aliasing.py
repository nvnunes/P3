"""Alias-only propagation must leave every reconstruction coefficient fixed."""

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import pistonFilter
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _build


def _case(tmp_path, option):
    text = _INI_TEMPLATE.format(nph=[10000]*3, noisevar='[0., 0., 0.]')
    text = text.replace('[sensor_HO]', '[sensor_HO]\n'+option)
    path = tmp_path/'case.ini'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _build(str(path))


@pytest.mark.parametrize('model', ['standard', 'alias_aware'])
def test_model_owns_alias_details(tmp_path, model):
    value = _case(tmp_path, f"ReconstructionModel = {model!r}")
    aware = model == 'alias_aware'
    assert value.aliasAwareReconstruction == aware
    assert value.tomographicAliasing == aware
    assert value.symmetricAliasing == aware
    assert value.W.shape[-1] == value.nGs*(2 if aware else 1)

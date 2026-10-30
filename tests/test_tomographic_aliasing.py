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

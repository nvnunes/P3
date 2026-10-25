"""Atmospheric r0 spelling compatibility for file and dictionary inputs."""

import ast
import copy
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

from p3.aoSystem.aoSystem import aoSystem


@pytest.fixture
def config():
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(Path(__file__).with_name('scao_test_wvl1100nm.ini'))
    config = {section: {key: ast.literal_eval(value) for key, value in parser[section].items()}
              for section in parser.sections()}
    config['telescope']['Resolution'] = 32
    del config['atmosphere']['Seeing']
    return config


def _system(config, input_kind, tmp_path):
    if input_kind == 'dict':
        return aoSystem('', config_dict=config, verbose=False)
    parser = ConfigParser()
    parser.optionxform = str
    parser.read_dict({section: {key: repr(value) for key, value in values.items()}
                      for section, values in config.items()})
    path = tmp_path / 'r0.ini'
    with path.open('w') as stream:
        parser.write(stream)
    return aoSystem(str(path), verbose=False)


@pytest.mark.parametrize('input_kind', ['file', 'dict'])
@pytest.mark.parametrize('spelling', ['r0_value', 'r0_Value'])
def test_r0_spellings_give_the_same_atmosphere(config, input_kind, spelling, tmp_path):
    config['atmosphere'][spelling] = 0.18
    original = copy.deepcopy(config)
    system = _system(config, input_kind, tmp_path)
    expected = 0.18 * np.cos(np.deg2rad(config['telescope']['ZenithAngle']))**(3 / 5)
    assert system.atm.r0 == pytest.approx(expected)
    np.testing.assert_array_equal(system.atm.weights, config['atmosphere']['Cn2Weights'])
    assert config == original


@pytest.mark.parametrize('input_kind', ['file', 'dict'])
def test_lowercase_r0_keeps_precedence(config, input_kind, tmp_path):
    config['atmosphere'].update(r0_value=0.18, r0_Value=0.25)
    system = _system(config, input_kind, tmp_path)
    expected = 0.18 * np.cos(np.deg2rad(config['telescope']['ZenithAngle']))**(3 / 5)
    assert system.atm.r0 == pytest.approx(expected)


@pytest.mark.parametrize('input_kind', ['file', 'dict'])
def test_seeing_keeps_precedence_over_both_r0_spellings(config, input_kind, tmp_path):
    config['atmosphere'].update(Seeing=0.8, r0_value=0.18, r0_Value=0.25)
    system = _system(config, input_kind, tmp_path)
    expected = 0.976 * config['atmosphere']['Wavelength'] / 0.8 * 206264.80624709636
    expected *= np.cos(np.deg2rad(config['telescope']['ZenithAngle']))**(3 / 5)
    assert system.atm.r0 == pytest.approx(expected)

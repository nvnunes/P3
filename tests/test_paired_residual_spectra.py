"""Paired optical/reporting spectra retain independent propagation results."""
import numpy as np
import pytest
from test_alias_aware_xy_propagation import _model

@pytest.mark.parametrize('controller,aware', [('integrator', False), ('integrator', True), ('polc', True)])
@pytest.mark.parametrize('exposure', [0., .001])
def test_paired_spectra_equal_independent_calls(tmp_path, controller, aware, exposure):
    model = _model(tmp_path/'model', controller, [.01, .02, .03], exposure, aware)
    weights = model.W.copy()
    for method in (model._tomographic_aliasing_psd, model.spatioTemporalPSD):
        variance = method(piston_filter=True)
        optical = method(piston_filter=False)
        paired = method(optical_pair=True)
        np.testing.assert_array_equal(paired[0], variance)
        np.testing.assert_array_equal(paired[1], optical)
    np.testing.assert_array_equal(model.W, weights)

@pytest.mark.parametrize('method', ['_tomographic_aliasing_psd', 'spatioTemporalPSD'])
def test_paired_input_requires_reporting_spectrum(tmp_path, method):
    model = _model(tmp_path/'model', 'polc', [0., 0., 0.])
    with pytest.raises(ValueError, match='piston_filter=True'):
        getattr(model, method)(piston_filter=False, optical_pair=True)

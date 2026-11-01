"""Unresolved atmospheric power against continuous covariance and grid controls."""

import numpy as np
import pytest
from scipy import fft, special

from p3.aoSystem.atmosphericTail import AtmosphericTail, _COEFFICIENT


def test_completed_source_matches_continuous_von_karman_covariance():
    # Periodic images are negligible for this independently chosen L0/period.
    n, dk, r0, outer = 32, .125, .2, 1.
    tail = AtmosphericTail(r0,outer,dk,n,n)
    q = fft.fftfreq(n)*n*dk
    source = _COEFFICIENT*r0**(-5/3)*(q[:,None]**2+q[None]**2+outer**-2)**(-11/6)*dk**2
    covariance = fft.ifft2(source+tail.power).real*n**2
    distance = np.arange(8)/(n*dk)
    zero = np.pi*_COEFFICIENT*(outer/r0)**(5/3)/(5/6)
    x = 2*np.pi*distance[1:]/outer
    expected = np.r_[zero,zero*2**(1/6)/special.gamma(5/6)*x**(5/6)*special.kv(5/6,x)]
    np.testing.assert_allclose(covariance[0,:8],expected,atol=2e-6,rtol=0)
    assert np.all(tail.power >= 0)
    assert tail.convergence_error < 1e-5
    tighter = AtmosphericTail(r0,outer,dk,n,n,tolerance=1e-7)
    assert abs(tail.power-tighter.power).sum() < 2e-6


def test_completing_different_source_supports_gives_same_optical_covariance():
    n, dk = 32,.125
    completed = []
    for count in (32,48):
        tail = AtmosphericTail(.2,2.,dk,count,n,symmetric_source=True)
        q = (np.arange(count)-count//2)*dk
        source = _COEFFICIENT*.2**(-5/3)*(q[:,None]**2+q[None]**2+.25)**(-11/6)*dk**2
        source[0] = 0;source[:,0] = 0
        # Include DC here to compare to the complete covariance, not SF only.
        index = (np.arange(count)-count//2)%n
        folded = np.zeros(n*n)
        np.add.at(folded,(index[:,None]*n+index[None]).ravel(),source.ravel())
        completed.append(folded.reshape(n,n)+tail.power)
    np.testing.assert_allclose(completed[0],completed[1],atol=1e-12,rtol=0)


def test_tail_uses_physical_strength_and_requires_valid_grids():
    a = AtmosphericTail(.2,2.,.125,32,32)
    b = AtmosphericTail(.4,2.,.125,32,32)
    scale = 2**(5/3)
    # An absolute phase tolerance may select different quadrature grids.
    assert abs(a.power-b.power*scale).sum() < (1+scale)*1e-5
    matched = b._folded_power(a.count)
    matched += b._outside_square((a.count/2-.5)*b.dk)/b.n**2
    np.testing.assert_allclose(a.power,matched*scale,rtol=2e-14,atol=1e-16)
    for args in [(0,2,.125,32,32),(.2,2,.125,31,32),(.2,2,0,32,32)]:
        with pytest.raises(ValueError):
            AtmosphericTail(*args)
    assert np.isfinite(AtmosphericTail(.2,np.inf,.125,32,32).power).all()


def test_original_pupil_projection_matches_direct_complete_covariance():
    optics = pytest.importorskip('tiptop.finite_pupil')
    from p3.aoSystem.finiteAperture import _stationary_matrix
    from mastsel import pupilTiltFit
    pupil = np.ones((8,8))
    sampling = optics.FinitePupilSampling(pupil,32,8,8.)
    tail = AtmosphericTail(.2,1.,.125,32,32)
    moments = tail.fitted_moments(sampling)
    actual = _stationary_matrix(*moments,sampling)
    raw = sampling.covariance_from_lags(tail.covariance())
    fit = pupilTiltFit(pupil).reshape(2,-1)
    projection = np.eye(64)-sampling.ramps.T@fit
    expected = projection@raw@projection.T
    np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=1e-10)


def test_fractional_pupil_sampling_preserves_original_frequency_moments():
    optics = pytest.importorskip('tiptop.finite_pupil')
    from p3.aoSystem.finiteAperture import _stationary_terms
    sampling = optics.FinitePupilSampling(np.ones((17,17)),32,8,8.)
    tail = AtmosphericTail(.2,1.,.125,32,32,symmetric_source=True)
    count = 128
    q = (np.arange(count)-count//2)*tail.dk
    power = _COEFFICIENT*.2**(-5/3)*(q[:,None]**2+q[None]**2+1)**(-11/6)*tail.dk**2
    inside = abs(np.arange(count)-count//2)<16
    power[inside[:,None]&inside[None]] = 0
    power[0] = 0;power[:,0] = 0
    expected = _stationary_terms(power,sampling.tilt_moments(count,8.),sampling,32)
    tail.count = count
    tail.remainder = 0.
    tail.power = tail._folded_power(count)
    actual = tail.fitted_moments(sampling)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,atol=2e-13,rtol=2e-11)


def test_alias_completion_changes_fitting_power_without_changing_inverse(tmp_path, monkeypatch):
    from test_optical_residual_power import _model
    completed = _model(tmp_path/'complete',optical=True,mode='alias_aware')
    from p3.aoSystem.fourierModel import fourierModel
    monkeypatch.setattr(fourierModel,'_unresolved_atmospheric_psd',
                        lambda self: np.zeros((self.freq.nOtf,self.freq.nOtf)))
    truncated = _model(tmp_path/'truncated',optical=True,mode='alias_aware')
    np.testing.assert_array_equal(completed.W,truncated.W)
    np.testing.assert_array_equal(completed.psdSpatioTemporal,truncated.psdSpatioTemporal)
    np.testing.assert_array_equal(completed.psdAlias,truncated.psdAlias)
    assert completed.opticalPSD.sum() > truncated.opticalPSD.sum()
    assert completed.psdFit.sum() > truncated.psdFit.sum()


def test_standard_path_never_requests_completion(tmp_path, monkeypatch):
    from p3.aoSystem.fourierModel import fourierModel
    from test_optical_residual_power import _model
    def forbidden(self):
        raise AssertionError('Standard reconstruction requested atmospheric completion')
    monkeypatch.setattr(fourierModel,'_unresolved_atmospheric_psd',forbidden)
    model = _model(tmp_path/'standard',optical=True)
    assert model.reconstructionModel == 'standard'

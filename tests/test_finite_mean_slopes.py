"""Projected POLC covariance against independently composed physical waves."""
import numpy as np
import pytest

from test_gaussian_mirrors import model
from test_finite_two_stage import sampling


def test_projection_matches_direct_coupled_waves(tmp_path):
    native=model(tmp_path/'case');sample=sampling(native)
    native.removeMeanSlopes=False
    backend=native._finite_aperture_backend()
    baseline,_=backend.residual_covariance(sample,0)
    native.removeMeanSlopes=True
    actual,_=backend.residual_covariance(sample,0)
    take=np.linspace(0,len(sample.xy)-1,5,dtype=int)
    expected=np.zeros((5,5))
    o1=backend.operator[take];o2=backend.science_operator(sample,0)[take]
    means=backend.projected_feedback.means
    m=backend.nc*backend.nv
    q=np.column_stack((backend.qx.ravel(),backend.qy.ravel()))
    small=means.T@backend.feedback@means
    for layer,(a,h) in enumerate(zip(backend.contraction,backend.heights)):
        u=a*q;gradient=2j*np.pi*u*np.prod(np.sinc(backend.pitch*u),axis=1)[:,None]
        guide=np.exp(2j*np.pi*(h*q@backend.guide.T+u@backend.offsets.T))
        cell=np.exp(2j*np.pi*u@backend.cells.T)
        y=(guide[:,:,None,None]*gradient[:,None,:,None]*cell[:,None,None,:]).reshape(len(q),m).T
        h1=backend.temporal[layer].ravel()
        angle=np.deg2rad(native.ao.atm.wDir[layer])
        f=(q@np.array([np.cos(angle),np.sin(angle)]))*native.ao.atm.wSpeed[layer]
        h2=backend.stage.gain*np.exp(2j*np.pi*f*backend.stage.delay/native.ao.rtc.holoop['rate'])
        filtered=y-means@(means.T@y)
        # Solve the projected feedback equation directly at each frequency.
        rhs=means.T@backend.feedback@filtered
        reduced=np.linalg.solve(np.eye(backend.nc)[None]-h1[:,None,None]*small,
                                rhs.T[...,None])[...,0].T
        first=h1*filtered+means@reduced*h1**2
        second=h2*(y-backend.feedback@first)
        second-=means@(means.T@second)
        command=o1@first+o2@second
        old=o1@(h1*y)+o2@(h2*(y-backend.feedback@(h1*y)))
        phase=np.exp(2j*np.pi*(sample.xy[take]@q.T+h*q@native.ao.src.direction[:,0]))
        phase-=sample.ramps[:,take].T@backend.kernels.reshape(2,-1)*np.exp(2j*np.pi*h*q@native.ao.src.direction[:,0])
        power=backend.raw[layer].ravel()*backend.dk**2
        new_residual=phase-command;old_residual=phase-old
        expected+=((new_residual*power)@new_residual.conj().T-(old_residual*power)@old_residual.conj().T).real
    np.testing.assert_allclose((actual-baseline)[np.ix_(take,take)],expected,rtol=2e-8,atol=2e-9)
    assert np.linalg.eigvalsh(actual).min()>-1e-7
    image,checks=sample.psf(actual)
    np.testing.assert_allclose(image.sum(),1,atol=2e-12)


def test_regularization_changes_training_without_injected_noise(tmp_path):
    native=model(tmp_path/'case');native.my_data_map['sensor_HO']['ReconstructionRegularization']=.1
    b=native._finite_aperture_backend()
    np.testing.assert_array_equal(b.prior,b.raw)
    delta=b.cov_regular-b.cov_prior
    assert np.diag(delta).min()>=.1*np.diag(b.cov_prior).mean()
    assert np.all(native.ao.wfs.processing.noiseVar==0)


def test_projected_disabled_science_is_the_single_stage_limit(tmp_path):
    from p3.aoSystem.finiteActuator import FiniteActuatorReconstructor
    native=model(tmp_path/'case')
    native.removeMeanSlopes=True;native.my_data_map['RTC_science']['LoopGain_HO']=0.
    sample=sampling(native)
    actual,_=native.finite_aperture_covariance(sample,0)
    assert isinstance(native._finiteApertureReconstructor,FiniteActuatorReconstructor)
    single=FiniteActuatorReconstructor(native)
    expected,_=single.residual_covariance(sample,0)
    np.testing.assert_array_equal(actual,expected)

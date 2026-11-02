"""Physical Gaussian fitting, selector isolation and shared-wave identities."""
import numpy as np
import pytest
from scipy import integrate,linalg

from p3.aoSystem.mirrorSurface import geometry_signature
from p3.aoSystem.finiteActuator import FiniteActuatorReconstructor
from p3.aoSystem.finiteAperture import _stationary
from test_finite_two_stage import coupled_model,sampling,require_device


def model(path):
    return coupled_model(path)


def test_optional_geometry():
    assert geometry_signature({}, str) is None
    assert geometry_signature({'PathInfGeometry': ''}, str) is None
    with pytest.raises(ValueError, match='PathInfGeometry'):
        geometry_signature({'PathInfGeometry': False}, str)


def test_surface_fitting_projection_and_exact_cell_gradient(tmp_path):
    value=model(tmp_path/'case');b=value._finite_aperture_backend();g=b.common_surface
    response=g.basis(g.fit_points)
    np.testing.assert_allclose(response@g.raw_fit@response,response,rtol=1e-9,atol=1e-9)
    y,x=np.nonzero(g.pupil);points=np.column_stack((g.pupil_x[x],g.pupil_y[y]));raw=g.basis(points)
    tilt=points-points.mean(0)
    expected=np.eye(len(g.ix))-linalg.lstsq(raw,tilt,cond=1e-12)[0]@linalg.lstsq(tilt,raw,cond=1e-12)[0]
    np.testing.assert_allclose(g.projection,expected,rtol=5e-9,atol=1e-10)
    np.testing.assert_allclose(g.fit,g.projection@g.raw_fit,rtol=0,atol=0)
    cell=np.array([[.17,-.31]]);pitch=.6;offset=np.array([[.1,.07]])
    actual=g.slopes(cell,pitch,offset)
    centre=g.centres[2];beta=np.log(g.coupling)/g.pitch**2
    for axis in range(2):
        def f(y,x):
            p=np.array([x,y]);return 2*beta*(p[axis]-centre[axis])*np.exp(beta*np.sum((p-centre)**2))
        point=cell[0]+offset[0]
        expected=integrate.dblquad(f,point[0]-pitch/2,point[0]+pitch/2,
            lambda _:point[1]-pitch/2,lambda _:point[1]+pitch/2)[0]/pitch**2
        np.testing.assert_allclose(actual[axis,2],expected,rtol=2e-12,atol=1e-13)


def test_complete_coupled_covariance_and_psf(tmp_path):
    value=model(tmp_path/'case');value.removeMeanSlopes=False;s=sampling(value);b=value._finite_aperture_backend()
    residual,_=b.residual_blocks(s,0);x1,x2,x12=b._query_crosses(s,0)
    o1=b.operator;o2=b.science_operator(s,0)
    expected=b.source.copy()+b.correction+_stationary(b._independent_power(0),b.aux_kernels,s,s.n)
    expected-=x1@o1.T+o1@x1.T
    expected+=o2@b.cbb@o2.T+o1@b.cab@o2.T+o2@b.cab.T@o1.T
    cross=x2-x12@b.feedback.T;expected-=cross@o2.T+o2@cross.T
    actual,_=b.residual_covariance(s,0)
    np.testing.assert_allclose(actual,expected,rtol=3e-9,atol=3e-10)
    image,checks=s.psf_blocks(residual.diagonal,residual.blocks())
    np.testing.assert_allclose(image.sum(),1,atol=2e-12);assert checks['minimum_structure_function']>=-2e-7
    np.testing.assert_allclose(b.feedback,b.common_surface.slopes(b.cells,b.pitch,b.offsets)@b.weights,rtol=0,atol=0)


def test_gaussian_disabled_science_uses_same_common_calculation(tmp_path):
    value=model(tmp_path/'case');s=sampling(value)
    b=value._finite_aperture_backend();b.prepare_optics(s,False)
    # A matched single-stage owner has exactly the same common estimator and
    # applied surface. This is a limiting identity, not an LTAO campaign.
    single=FiniteActuatorReconstructor(value);single.prepare_optics(s,False)
    np.testing.assert_array_equal(b.weights,single.weights)
    np.testing.assert_array_equal(b.operator,single.operator)


def test_gaussian_geometry_content_invalidates_cache(tmp_path):
    value=model(tmp_path/'case');old=value._finite_aperture_backend()
    from astropy.io import fits
    path=tmp_path/'case/common.fits'
    with fits.open(path,mode='update') as h:
        h['ACTUATOR_AXIS'].data *= 1.01
    new=value._finite_aperture_backend();assert old is not new
    assert not np.array_equal(old.weights,new.weights)


def test_gaussian_joint_moments_from_independent_shared_waves(tmp_path):
    value=model(tmp_path/'case');b=value._finite_aperture_backend()
    m=b.nc*b.nv;aa=np.zeros((m,m));bb=aa.copy();ab=aa.copy()
    q=np.column_stack((b.qx.ravel(),b.qy.ravel()))
    for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
        u=a*q;gradient=2j*np.pi*u*(np.sinc(b.pitch*u[:,0])*np.sinc(b.pitch*u[:,1]))[:,None]
        guide=np.exp(2j*np.pi*(h*q@b.guide.T+u@b.offsets.T));cell=np.exp(2j*np.pi*(u@b.cells.T))
        measured=(guide[:,:,None,None]*gradient[:,None,:,None]*cell[:,None,None,:]).reshape(len(q),m).T
        angle=np.deg2rad(value.ao.atm.wDir[layer]);f=-(q[:,0]*np.cos(angle)+q[:,1]*np.sin(angle))*value.ao.atm.wSpeed[layer]
        h2=np.exp(-2j*np.pi*f/value.ao.rtc.holoop['rate']);first=measured*b.temporal[layer].ravel()
        second=h2*(measured-b.feedback@first);p=b.raw[layer].ravel()*b.dk**2
        aa+=((first*p)@first.conj().T).real;bb+=((second*p)@second.conj().T).real;ab+=((first*p)@second.conj().T).real
    for actual,expected in ((b.caa,aa),(b.cbb,bb),(b.cab,ab)):
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('device_name', ['cuda', 'mps'])
def test_command_factored_device_preserves_complete_psf(tmp_path, device_name):
    require_device(device_name)
    value=model(tmp_path/'case');value.removeMeanSlopes=False;s=sampling(value);b=value._finite_aperture_backend()
    cpu,_=b.residual_covariance(s,0);cpu_psf,_=s.psf(cpu)
    if device_name == 'mps':
        from p3.aoSystem.finiteTwoStageMps import MpsFiniteProducts
        b.device=MpsFiniteProducts(b,s)
    else:
        from p3.aoSystem.finiteTwoStageCupy import CupyFiniteProducts
        b.device=CupyFiniteProducts(b,s)
    actual,_=b.residual_covariance(s,0);image,_=s.psf(actual)
    tolerance = 2e-4 if device_name == 'mps' else 2e-10
    np.testing.assert_allclose(actual,cpu,rtol=tolerance,atol=tolerance)
    assert np.abs(image-cpu_psf).sum() < (1e-4 if device_name == 'mps' else 1e-10)


def test_phase_target_covariance_matches_direct_wave_sum(tmp_path):
    value=model(tmp_path/'case');b=value._finite_aperture_backend()
    q=np.column_stack((b.qx.ravel(),b.qy.ravel()));points=b.phase_fit_points
    for average in (False,True):
        expected=np.zeros((len(points),b.nc*b.nv))
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            u=a*q;gradient=2j*np.pi*u*(np.sinc(b.pitch*u[:,0])*np.sinc(b.pitch*u[:,1]))[:,None]
            guide=np.exp(2j*np.pi*(h*q@b.guide.T+u@b.offsets.T));cell=np.exp(2j*np.pi*(u@b.cells.T))
            measured=(guide[:,:,None,None]*gradient[:,None,:,None]*cell[:,None,None,:]).reshape(len(q),-1).T
            if average:
                phase=sum(w*np.exp(2j*np.pi*h*(q@beta)) for beta,w in zip(b.targets,b.target_weights))*b.gain
            else:phase=np.exp(2j*np.pi*h*(q@value.ao.src.direction[:,0]))
            density=b.prior[layer].ravel()*b.dk**2
            expected+=((np.exp(2j*np.pi*(points@q.T))*phase*density)@measured.conj().T).real
        actual=b._prior_cross(average=average,field=None if average else 0)
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-4


def test_unsupported_influence_model_rejected(tmp_path):
    value=model(tmp_path/'case')
    value.my_data_map['DM']['InfModel']='unsupported'
    with pytest.raises(ValueError,match='InfModel'):
        value._finite_aperture_backend()

"""Shared single-stage finite physics independent of science-stage support."""
from copy import deepcopy
import numpy as np
import pytest
from scipy import integrate, linalg

from p3.aoSystem.finiteConfiguration import configuration
from p3.aoSystem.mirrorSurface import geometry_signature
from test_finite_aperture import small_model


def model(path):
    return small_model(path, True)


def sampling(value):
    from tiptop.finite_pupil import FinitePupilSampling
    n = value.freq.nOtf
    side = int(2*np.round(float(value.ao.tel.D)/2*n*value.freq.PSDstep))
    period = 1/float(value.freq.kxAO_[1,0]-value.freq.kxAO_[0,0])
    return FinitePupilSampling(value.ao.tel.pupil,n,side,period)


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


def test_gaussian_geometry_content_invalidates_cache(tmp_path):
    value=model(tmp_path/'case');old=value._finite_aperture_backend()
    from astropy.io import fits
    path=tmp_path/'case/common.fits'
    with fits.open(path,mode='update') as h:
        h['ACTUATOR_AXIS'].data *= 1.01
    new=value._finite_aperture_backend();assert old is not new
    assert not np.array_equal(old.weights,new.weights)


def test_phase_target_covariance_matches_direct_wave_sum(tmp_path):
    value=model(tmp_path/'case');b=value._finite_aperture_backend()
    q=np.column_stack((b.qx.ravel(),b.qy.ravel()));points=b.phase_fit_points
    for average in (True,):
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


def test_projection_matches_direct_single_stage_waves(tmp_path):
    native=model(tmp_path/'case');sample=sampling(native)
    native.removeMeanSlopes=False
    backend=native._finite_aperture_backend()
    baseline,_=backend.residual_covariance(sample,0)
    native.removeMeanSlopes=True
    actual,_=backend.residual_covariance(sample,0)
    take=np.linspace(0,len(sample.xy)-1,5,dtype=int)
    expected=np.zeros((5,5))
    o1=backend.operator[take];o2=np.zeros_like(o1)
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
        h2=np.zeros_like(h1)
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


@pytest.mark.parametrize('value',[True,-1.,float('nan'),'0.1'])
def test_regularization_is_an_explicit_nonnegative_number(value):
    with pytest.raises(ValueError):
        configuration({'sensor_HO':{'ReconstructionRegularization':value}},'finite_aperture')


def test_regularization_scope_and_cache(tmp_path):
    with pytest.raises(ValueError,match='requires finite_aperture'):
        configuration({'sensor_HO':{'ReconstructionRegularization':.1}},'alias_aware')
    value=model(tmp_path/'case')
    old=value._finite_aperture_backend()
    value.my_data_map['sensor_HO']['ReconstructionRegularization']=.1
    new=value._finite_aperture_backend()
    assert new is not old and new.regularization==.1
    assert not np.array_equal(old.weights,new.weights)


@pytest.mark.parametrize('section,key',[('sensor_HO','ReconstructionPrior'),
    ('sensor_HO','RemoveMeanSlopes'),('DM','SurfaceModel'),('DM','PathInfluenceGeometry')])
def test_retired_shared_settings_are_rejected(section,key):
    with pytest.raises(ValueError,match='Retired'):
        configuration({section:{key:False}},'finite_aperture')


def test_default_geometry_matches_explicit_fits(tmp_path):
    from astropy.io import fits
    value=model(tmp_path/'case')
    value.my_data_map['DM'].pop('PathInfGeometry')
    sample=sampling(value)
    expected,_=value.finite_aperture_covariance(sample,0)
    backend=value._finite_aperture_backend()
    g=backend.common_surface
    mask=np.zeros((len(g.axis),len(g.axis)),dtype=np.uint8);mask[g.iy,g.ix]=1
    hdus=[fits.PrimaryHDU()];hdus[0].header['GEOMVER']=1
    for name,data in [('ACTUATOR_AXIS',g.axis),('VALID_ACTUATORS',mask),('PUPIL_X',g.pupil_x),('PUPIL_Y',g.pupil_y)]:
        hdu=fits.ImageHDU(data,name=name)
        if name!='VALID_ACTUATORS':hdu.header['BUNIT']='m'
        hdus.append(hdu)
    path=tmp_path/'equivalent.fits';fits.HDUList(hdus).writeto(path)
    value.my_data_map['DM']['PathInfGeometry']=str(path)
    actual,_=value.finite_aperture_covariance(sample,0)
    assert value._finite_aperture_backend() is not backend
    np.testing.assert_array_equal(actual,expected)


def test_mean_projection_is_specific_to_lgs(tmp_path):
    from p3.aoSystem.fourierModel import fourierModel
    native=model(tmp_path/'case')
    assert native.removeMeanSlopes
    config=deepcopy(native.my_data_map);config['sources_HO']['Height']=0.
    ngs=fourierModel(None,config_dict=config,calcPSF=False,display=False,
        verbose=False,doComputations=False,computeFocalAnisoCov=False)
    assert not ngs.removeMeanSlopes

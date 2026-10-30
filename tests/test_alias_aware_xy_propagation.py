"""X/y alias-aware reconstruction uses one consistent measurement basis."""
from configparser import ConfigParser

import numpy as np
import pytest

from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _p3_path
from p3.aoSystem.fourierModel import fourierModel


def _model(path, noise, reconstruction='alias_aware'):
    path.mkdir(parents=True)
    c=ConfigParser();c.optionxform=str
    c.read_string(_INI_TEMPLATE.format(nph=[10000,15000,20000],noisevar=repr(noise)))
    c['sources_science'].update(Zenith='[0.,15.,30.]',Azimuth='[0.,45.,135.]')
    c['sensor_HO'].update(ReconstructionModel=repr(reconstruction),
        GridOffsets='[[-.23,-.23],[.23,-.23],[.23,.23]]',
        TomographicRegularizationFloor='1e-6')
    with (path/'system.ini').open('w') as f:c.write(f)
    return fourierModel(str(path/'system.ini'),path_root=_p3_path(),calcPSF=False,
        display=False,verbose=False,reduce_memory=False,computeFocalAnisoCov=False,
        computeOpticalPSD=True)


@pytest.mark.parametrize('noise',[[.01,.02,.03],[0.,.02,.03],[None]])
def test_noise_covariance_and_propagation_keep_both_axes(tmp_path,noise):
    model=_model(tmp_path/'case',noise)
    assert model.temporalModel == 'polc' and model.exposureTime == 0.
    per_wfs=np.asarray(model.ao.wfs.processing.noiseVar)
    channels=np.repeat(per_wfs,2)
    np.testing.assert_array_equal(model._measurement_noise_variances(),channels)
    assert channels.shape==(2*model.nGs,)
    gain=model.noiseGain
    covariance=np.diag(channels)
    for source in range(model.ao.src.nSrc):
        command=model.PbetaDM[source] @ model.W
        dense=(command @ covariance @ command.conj().swapaxes(-1,-2))[...,0,0].real
        expected=dense*model.freq.mskInAO_*model.freq.pistonFilterAO_*gain
        np.testing.assert_allclose(model.psdNoise[...,source],expected,rtol=3e-13,atol=1e-14)
    assert np.max(model.psdNoise)>0
    assert np.isfinite(model.PSD).all() and np.isfinite(model.opticalPSD).all()
    # Optical-input selection must not change weights, variance reporting or noise.
    config=ConfigParser();config.optionxform=str;config.read(tmp_path/'case/system.ini')
    plain=fourierModel(str(tmp_path/'case/system.ini'),path_root=_p3_path(),calcPSF=False,
        display=False,verbose=False,reduce_memory=False,computeFocalAnisoCov=False)
    for key in ('W','PSD','psdNoise'):
        np.testing.assert_array_equal(getattr(plain,key),getattr(model,key))


def test_embedding_combined_weights_preserves_spatial_noise_power(tmp_path):
    scalar=_model(tmp_path/'scalar',[.01,.02,.03],reconstruction='standard')
    xy=_model(tmp_path/'xy',[.01,.02,.03])
    k=np.hypot(xy.freq.kxAO_,xy.freq.kyAO_)
    ux=np.divide(xy.freq.kxAO_,k,out=np.zeros_like(k),where=k>0)
    uy=np.divide(xy.freq.kyAO_,k,out=np.zeros_like(k),where=k>0)
    xy.W=np.stack((scalar.W*ux[...,None,None],scalar.W*uy[...,None,None]),axis=-1).reshape(
        *scalar.W.shape[:-1],2*scalar.nGs)
    scalar_gain=min(.8,.4+.1333*scalar.ao.rtc.holoop['delay'])**2
    np.testing.assert_allclose(xy.noisePSD()/xy.noiseGain, scalar.noisePSD()/scalar_gain,
                               rtol=3e-13,atol=1e-14)


@pytest.mark.parametrize('piston',[False,True])
def test_main_and_alias_power_match_independent_xy_cells(tmp_path,piston):
    from p3.aoSystem.FourierUtils import pistonFilter
    model=_model(tmp_path/'case',[.01,.02,.03])
    temporal,exposure=model.temporalModel,model.exposureTime
    center=model.freq.resAO//2;ij=(center+2,center+1)
    k=np.array([model.freq.kxAO_[ij],model.freq.kyAO_[ij]])
    d=model.ao.wfs.optics[0].dsub
    heights=model.ao.atm.heights*model.strechFactor
    offsets=np.asarray(model.my_data_map['sensor_HO']['GridOffsets'])*d
    nodes,weights=np.polynomial.legendre.leggauss(32);nodes=nodes*d/2;weights=weights/2
    wind=np.deg2rad(model.ao.atm.wDir)
    velocity=model.ao.atm.wSpeed[:,None]*np.column_stack((np.cos(wind),np.sin(wind)))
    rtc=model.ao.rtc.holoop
    def response(q):
        measured=[]
        for guide,offset in enumerate(offsets):
            area=np.einsum('i,ij,j->',weights,np.exp(2j*np.pi*(
                q[0]*(nodes[:,None]+offset[0])+q[1]*(nodes[None,:]+offset[1]))),weights)
            phase=np.exp(2j*np.pi*(q @ model.gs.direction[:,guide])*heights)
            for axis in range(2):measured.append(2j*np.pi*q[axis]*area*phase)
        return np.array(measured)
    def timing(q):
        frequency=-velocity @ q
        steps=np.arange(2000)
        impulse=rtc['gain']*(1-rtc['gain'])**steps
        return np.sum(impulse*np.exp(-2j*np.pi*frequency[:,None]*
            (steps+rtc['delay'])/rtc['rate']),axis=-1)
    M=response(k)
    frequency=-velocity @ k
    measured=M*np.sinc(exposure*frequency)[None]
    np.testing.assert_allclose(model.MPalphaL[ij],measured,rtol=2e-12,atol=2e-13)
    factor=float(model.freq.pistonFilterAO_[ij])
    power=model.Cphi[ij] if piston else model.Cphi[ij]/factor
    actual=model.spatioTemporalPSD(piston_filter=piston)
    for source in range(model.ao.src.nSrc):
        beta=model.ao.src.direction[:,source]
        phase=heights*(beta @ k)
        if temporal=='integrator':phase+=rtc['delay']/rtc['rate']*frequency
        science=np.exp(2j*np.pi*phase)
        correction=(model.PbetaDM[source][ij] @ model.W[ij] @ measured).ravel()
        if temporal=='polc':correction*=timing(k)
        expected=np.sum(abs(science-correction)**2*power)*model.freq.mskInAO_[ij]
        np.testing.assert_allclose(actual[ij+(source,)],expected,rtol=2e-12,atol=1e-13)
    alias=0.
    T=np.mean([detector.clock_rate for detector in model.ao.wfs.detector])/rtc['rate']
    for m in model._alias_shift_grid():
        for n in model._alias_shift_grid():
            if m==n==0:continue
            q=k-np.array([m,n])/d
            measured=response(q)
            frequency=-velocity @ q
            if temporal=='polc':measured*=np.sinc(exposure*frequency)[None]
            command=(model.W[ij] @ measured).ravel()
            if temporal=='polc':
                command*=timing(q)
                variance=np.sum(model.ao.atm.weights*abs(command)**2)
            else:
                average=np.sum(model.ao.atm.weights*np.sinc(q[0]*velocity[:,0]*exposure)
                    *np.sinc(q[1]*velocity[:,1]*exposure)*np.exp(-2j*np.pi*frequency*T*rtc['delay']))*model.h1[ij]
                variance=np.sum(model.ao.atm.weights*abs(command)**2)*abs(average)**2
            spectral=.0229*model.ao.atm.r0**(-5/3)*(q @ q+1/model.ao.atm.L0**2)**(-11/6)
            pf=pistonFilter(model.ao.tel.D,np.array([[np.linalg.norm(q)]]))[0,0] if piston else 1.
            alias+=variance*spectral*pf*model.freq.mskInAO_[ij]
    np.testing.assert_allclose(model.aliasingPSD(piston_filter=piston)[ij],alias,rtol=3e-12,atol=1e-13)

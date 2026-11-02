"""Public mirror/controller ownership, retired settings and execution policy."""
import numpy as np
import pytest
from astropy.io import fits

from p3.aoSystem.scienceCorrection import configuration, covariance_backend
from test_finite_two_stage import coupled_model, sampling


def sections():
    return {'sensor_HO': {'ReconstructionRegularization': .1},
            'DM_science': {'DmPitchs': [2.], 'NumberActuators': [5]},
            'RTC_science': {'LoopGain_HO': 1., 'LoopDelaySteps_HO': 1}}


def test_one_science_contract_and_explicit_timing():
    stage, regularization = configuration(sections(), 'finite_aperture')
    assert (stage.pitch,stage.actuators,stage.gain,stage.delay)==(2.,5,1.,1)
    assert stage.active and regularization==.1
    assert configuration({},'standard')==(None,0.)
    for missing in ('DM_science','RTC_science'):
        c=sections();c.pop(missing)
        with pytest.raises(ValueError,match='together'):configuration(c,'finite_aperture')
    c=sections();c['RTC_science'].pop('LoopDelaySteps_HO')
    with pytest.raises(ValueError,match='explicit'):configuration(c,'finite_aperture')


@pytest.mark.parametrize('section,key,value',[
    ('RTC_science','CommandGain',1.),('RTC_science','CommandDelaySteps',1),
    ('DM_Stage2','Enabled',False),('DM','SurfaceModel','fourier'),
    ('DM','PathInfluenceGeometry','old.fits'),('DM_science','PathInfluenceGeometry','old.fits'),
    ('DM','SurfaceGeometry','old.npz'),('DM_science','Enabled',False),
    ('sensor_HO','RemoveMeanSlopes',False),('sensor_HO','ReconstructionPrior','phase'),
    ('sensor_HO','ReconstructionNoiseFraction',0.),('sensor_HO','CovarianceArrayBackend','numpy')])
def test_retired_keys_rejected_even_when_inactive(section,key,value):
    c=sections();c.setdefault(section,{})[key]=value
    with pytest.raises(ValueError,match='retired|Retired'):configuration(c,'finite_aperture')


@pytest.mark.parametrize('value',[True,-1.,float('nan'),'0.1'])
def test_regularization_is_explicit_nonnegative_number(value):
    c={'sensor_HO':{'ReconstructionRegularization':value}}
    with pytest.raises(ValueError):configuration(c,'finite_aperture')


@pytest.mark.parametrize('backend', ['cuda', 'mps'])
def test_numerical_backend_is_api_scoped(backend):
    stage,_=configuration(sections(),'finite_aperture')
    assert covariance_backend(backend,'finite_aperture',stage)==backend
    for value,model,science in [('typo','finite_aperture',stage),(backend,'finite_aperture',None),
                                (backend,'alias_aware',stage)]:
        with pytest.raises(ValueError):covariance_backend(value,model,science)
    with pytest.raises(ValueError,match='finite_aperture'):
        configuration({'sensor_HO':{'ReconstructionRegularization':.1}},'alias_aware')


def test_zero_science_gain_does_not_read_inactive_geometry(tmp_path):
    value=coupled_model(tmp_path/'case',False)
    value.my_data_map['DM_science']['PathInfGeometry']='/no/such/science.fits'
    actual,_=value.finite_aperture_covariance(sampling(value),0)
    assert np.isfinite(actual).all()


def test_geometry_has_one_pupil_and_coupling_owner(tmp_path):
    value=coupled_model(tmp_path/'case')
    backend=value._finite_aperture_backend()
    np.testing.assert_array_equal(backend.common_surface.pupil,value.ao.tel.pupil)
    assert backend.common_surface.coupling==value.my_data_map['DM']['InfCoupling'][0]
    with fits.open(tmp_path/'case/common.fits') as h:
        assert {x.name for x in h}=={'PRIMARY','ACTUATOR_AXIS','VALID_ACTUATORS','PUPIL_X','PUPIL_Y'}
    original=backend.weights.copy()
    value.my_data_map['DM']['InfCoupling']=[.3]
    assert not np.array_equal(value._finite_aperture_backend().weights,original)


def test_relative_geometry_uses_p3_path_resolver(tmp_path):
    value=coupled_model(tmp_path/'case')
    value.ao.path_root=str(tmp_path/'case')
    value.my_data_map['DM']['PathInfGeometry']='common.fits'
    value.my_data_map['DM_science']['PathInfGeometry']='science.fits'
    assert value._finite_aperture_backend().common_surface.signature


@pytest.mark.parametrize('bad', ['units','mask','axis','pupil-copy'])
def test_fits_contract_rejects_invalid_geometry(tmp_path,bad):
    value=coupled_model(tmp_path/'case')
    with fits.open(tmp_path/'case/common.fits',mode='update') as h:
        if bad=='units':h['ACTUATOR_AXIS'].header['BUNIT']='mm'
        elif bad=='mask':h['VALID_ACTUATORS'].data[0,0]=2
        elif bad=='axis':h['ACTUATOR_AXIS'].data[0]=float('nan')
        else:h.append(fits.ImageHDU(value.ao.tel.pupil,name='PUPIL'))
    with pytest.raises(ValueError):value._finite_aperture_backend()


def test_mean_projection_belongs_to_lgs_finite_measurements(tmp_path):
    from copy import deepcopy
    from p3.aoSystem.fourierModel import fourierModel
    from test_finite_aperture import small_model
    native = small_model(tmp_path/'case', True)
    assert native.removeMeanSlopes
    config = deepcopy(native.my_data_map)
    config['sources_HO']['Height'] = 0.
    ngs = fourierModel(None, config_dict=config, calcPSF=False, display=False,
                       verbose=False, doComputations=False, computeFocalAnisoCov=False)
    assert not ngs.removeMeanSlopes


def test_regularization_change_rebuilds_the_estimator(tmp_path):
    value = coupled_model(tmp_path/'case')
    old = value._finite_aperture_backend()
    value.my_data_map['sensor_HO']['ReconstructionRegularization'] = .1
    new = value._finite_aperture_backend()
    assert new is not old and new.regularization == .1
    assert not np.array_equal(old.weights, new.weights)


@pytest.mark.parametrize('science', [False, True])
def test_generated_geometry_and_equivalent_fits_produce_same_psf(tmp_path, science):
    value = coupled_model(tmp_path/'case', science)
    for name in ('DM', 'DM_science'):
        value.my_data_map[name].pop('PathInfGeometry')
    sample = sampling(value)
    covariance, _ = value.finite_aperture_covariance(sample, 0)
    backend = value._finiteApertureReconstructor
    image, _ = sample.psf(covariance)
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image.sum(), 1., atol=2e-12)
    surfaces = [('DM', backend.common_surface)]
    if science:
        surfaces.append(('DM_science', backend.science_surface))
    for name, surface in surfaces:
        primary = fits.PrimaryHDU(); primary.header['GEOMVER'] = 1
        hdus = [primary]
        for label, data in [('ACTUATOR_AXIS', surface.axis),
                            ('VALID_ACTUATORS', surface.valid.astype('uint8')),
                            ('PUPIL_X', surface.pupil_x), ('PUPIL_Y', surface.pupil_y)]:
            hdu = fits.ImageHDU(data, name=label)
            if label != 'VALID_ACTUATORS': hdu.header['BUNIT'] = 'm'
            hdus.append(hdu)
        path = tmp_path/(name+'.fits')
        fits.HDUList(hdus).writeto(path)
        value.my_data_map[name]['PathInfGeometry'] = str(path)
    explicit, _ = value.finite_aperture_covariance(sample, 0)
    assert value._finiteApertureReconstructor is not backend
    np.testing.assert_array_equal(covariance, explicit)


def test_regular_geometry_follows_nominal_mirror_and_pupil():
    from p3.aoSystem.mirrorSurface import _regular_geometry
    pupil = np.ones((9, 9))
    pupil[4, 4] = 0  # A central obscuration excludes that actuator centre.
    axis, valid, x, y = _regular_geometry(5, 2., 8., pupil)
    np.testing.assert_array_equal(axis, [-4., -2., 0., 2., 4.])
    np.testing.assert_array_equal(x, np.arange(-4., 5.))
    np.testing.assert_array_equal(y, x)
    assert valid.sum() == 24 and not valid[2, 2]


def test_explicit_missing_geometry_does_not_fall_back(tmp_path):
    value = coupled_model(tmp_path/'case')
    value.my_data_map['DM']['PathInfGeometry'] = str(tmp_path/'missing.fits')
    with pytest.raises(FileNotFoundError): value._finite_aperture_backend()


def test_generated_geometry_cache_tracks_realized_mirror_pitch(tmp_path):
    value = coupled_model(tmp_path/'case', False)
    value.my_data_map['DM'].pop('PathInfGeometry')
    old = value._finite_aperture_backend()
    value.ao.dms.pitch[0] *= .95
    new = value._finite_aperture_backend()
    assert new is not old
    assert not np.array_equal(new.common_surface.axis, old.common_surface.axis)


@pytest.mark.parametrize('change', ['gain', 'delay', 'off', 'on', 'remove', 'pitch', 'actuators'])
def test_changed_science_settings_match_a_fresh_model(tmp_path, change):
    from copy import deepcopy
    from p3.aoSystem.fourierModel import fourierModel
    value = coupled_model(tmp_path/'case', change != 'on')
    if change in ('pitch', 'actuators'):
        value.my_data_map['DM_science'].pop('PathInfGeometry')
    sample = sampling(value)
    before, _ = value.finite_aperture_covariance(sample, 0)
    old = value._finiteApertureReconstructor
    rtc = value.my_data_map['RTC_science']
    if change == 'gain': rtc['LoopGain_HO'] = .4
    elif change == 'delay': rtc['LoopDelaySteps_HO'] = 2
    elif change == 'on': rtc['LoopGain_HO'] = 1.
    elif change == 'off':
        rtc['LoopGain_HO'] = 0.
        value.my_data_map['DM_science']['PathInfGeometry'] = '/missing/inactive.fits'
    elif change == 'remove':
        value.my_data_map.pop('RTC_science')
        value.my_data_map.pop('DM_science')
    elif change == 'pitch': value.my_data_map['DM_science']['DmPitchs'] = [1.8]
    else: value.my_data_map['DM_science']['NumberActuators'] = [4]
    actual, _ = value.finite_aperture_covariance(sample, 0)
    fresh = fourierModel(None, config_dict=deepcopy(value.my_data_map), calcPSF=False,
                        display=False, verbose=False, reduce_memory=False,
                        computeOpticalPSD=True, computeFocalAnisoCov=False)
    expected, _ = fresh.finite_aperture_covariance(sample, 0)
    assert value._finiteApertureReconstructor is not old
    assert value.scienceMirror == fresh.scienceMirror
    assert value.twoStageEnabled == fresh.twoStageEnabled
    assert not np.array_equal(before, actual)
    np.testing.assert_array_equal(actual, expected)
    # Unchanged settings retain the newly built calculation.
    assert value._finite_aperture_backend() is value._finiteApertureReconstructor


def test_changed_science_settings_are_revalidated_before_reuse(tmp_path):
    value = coupled_model(tmp_path/'case')
    old = value._finite_aperture_backend()
    value.my_data_map['RTC_science'].pop('LoopDelaySteps_HO')
    with pytest.raises(ValueError, match='explicit'):
        value._finite_aperture_backend()
    assert value._finiteApertureReconstructor is old


def test_disabling_science_revalidates_numerical_backend_scope(tmp_path):
    value = coupled_model(tmp_path/'case')
    value.finiteApertureBackend = 'mps'
    value.my_data_map['RTC_science']['LoopGain_HO'] = 0.
    with pytest.raises(ValueError, match='active finite-aperture'):
        value._finite_aperture_backend()

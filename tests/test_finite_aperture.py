"""Finite sensing option, physical covariance and cache lifecycle checks."""

from configparser import ConfigParser

import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy import ndimage

from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.finiteAperture import (FiniteApertureReconstructor,
    _ScienceMeasurementQuery, input_signature, validate_native)
from test_p3_bugfixes_noise_tomo import _INI_TEMPLATE, _p3_path


def small_model(path, option=None):
    config = ConfigParser(); config.optionxform = str
    config.read_string(_INI_TEMPLATE.format(nph=[1e4]*3, noisevar='[0.,0.,0.]'))
    config['sensor_HO'].update(NumberLenslets='[4,4,4]',
        TipTiltRemoval="'pupil_fit'", TomographicRegularizationFloor='1e-6',
        GridOffsets='[[-.2,-.2],[.2,-.2],[.2,.2]]')
    config['DM'].update(NumberActuators='[8]', DmPitchs='[1.0]')
    config['sensor_science']['PixelScale'] = repr(1.65e-6*(180/np.pi)*3600*1000/32)
    if option is not None:
        config['sensor_HO']['ReconstructionModel'] = repr('finite_aperture' if option else 'standard')
    path.mkdir()
    if option:
        from _finite_geometry import write_geometry
        geometry=path/'common.fits'
        write_geometry(geometry, int(config['telescope'].get('Resolution',256)))
        config['DM'].update(InfModel="'gaussian'",InfCoupling='[0.25]',
                            PathInfGeometry=repr(str(geometry)))
    with (path/'system.ini').open('w') as f: config.write(f)
    return fourierModel(str(path/'system.ini'), path_root=_p3_path(), calcPSF=False,
        display=False, verbose=False, reduce_memory=False, computeOpticalPSD=True,
        computeFocalAnisoCov=False)


@pytest.mark.parametrize('origin', [[0., 0.], [-.01, .01], [-9.3, 13.7], [8., -8.]])
def test_separable_cubic_matches_periodic_ndimage(origin):
    rng = np.random.default_rng(22)
    size, period = 32, 8.
    x, y = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9))
    pupil = np.column_stack((x.ravel(), y.ravel()))
    pupil = pupil[np.linalg.norm(pupil, axis=1) > .7]
    x, y = np.meshgrid(np.linspace(-3, 3, 5), np.linspace(-3, 3, 5))
    cells = np.column_stack((x.ravel(), y.ravel()))
    cells = cells[np.linalg.norm(cells, axis=1) < 4]
    kernels = np.array([ndimage.spline_filter(rng.normal(size=(size, size)),
        order=3, mode='grid-wrap') for _ in range(2)])
    query = _ScienceMeasurementQuery(pupil, cells, size, period)
    for contraction in (1., .8):
        output = np.full((2, len(pupil), len(cells)), np.nan)
        query.evaluate(kernels, np.asarray(origin), contraction, output)
        c = np.asarray(origin)[None]-contraction*cells
        coordinates = np.array([pupil[:, 1, None]+c[None, :, 1],
            pupil[:, 0, None]+c[None, :, 0]])*size/period
        for axis in range(2):
            expected = ndimage.map_coordinates(kernels[axis], coordinates,
                order=3, mode='grid-wrap', prefilter=False)
            np.testing.assert_allclose(output[axis], expected, rtol=2e-13, atol=2e-13)


def test_omitted_false_preserve_native_predictions(tmp_path):
    omitted = small_model(tmp_path/'omitted'); false = small_model(tmp_path/'false', False)
    assert not omitted.finiteApertureReconstruction
    for name in ('W', 'PSD', 'opticalPSD'):
        np.testing.assert_array_equal(getattr(omitted, name), getattr(false, name))
    assert omitted.finiteApertureBackend is None
    with pytest.raises(ValueError, match='disabled'): omitted.finite_aperture_covariance(None, 0)


def test_scope_rejections_and_geometry_signature(tmp_path):
    model = small_model(tmp_path/'case', True)
    validate_native(model)
    signature = input_signature(model)
    model.ao.atm.wSpeed[0] *= 1.2
    assert signature != input_signature(model)
    model.ao.wfs.processing.noiseVar = np.array([.01,0.,0.])
    with pytest.raises(ValueError, match='noise'): validate_native(model)
    model.ao.wfs.processing.noiseVar[:] = 0
    model.temporalModel = 'integrator'
    with pytest.raises(ValueError, match='POLC'): validate_native(model)


def test_real_covariance_cache_reuse_and_invalidation(tmp_path):
    optics = pytest.importorskip('tiptop.finite_pupil')
    model = small_model(tmp_path/'case', True)
    side = int(2*np.round(float(model.ao.tel.D)/2*model.freq.nOtf*model.freq.PSDstep))
    period = 1/float(model.freq.kxAO_[1,0]-model.freq.kxAO_[0,0])
    sampling = optics.FinitePupilSampling(model.ao.tel.pupil, model.freq.nOtf, side, period)
    a, _ = model.finite_aperture_covariance(sampling, 0)
    backend = model.finiteApertureBackend
    b, _ = model.finite_aperture_covariance(sampling, 0)
    np.testing.assert_array_equal(a, b)
    assert backend is model.finiteApertureBackend
    assert np.linalg.eigvalsh(a).min() > -1e-8
    model.ao.wfs.processing.noiseVar[0] = .01
    with pytest.raises(ValueError, match='noise'):
        model.finite_aperture_covariance(sampling, 0)
    model.ao.wfs.processing.noiseVar[0] = 0.
    model.ao.atm.wSpeed[0] *= 1.2
    c, _ = model.finite_aperture_covariance(sampling, 0)
    assert backend is not model.finiteApertureBackend
    assert np.linalg.norm(c-a) > 1e-8
    # A newly configured grid must create a distinct estimator and signature.
    old = model.finiteApertureBackend
    model.my_data_map['sensor_HO']['GridOffsets'][0][0] += .1
    model.finite_aperture_covariance(sampling, 0)
    assert old is not model.finiteApertureBackend
    assert np.linalg.norm(old.weights-model.finiteApertureBackend.weights) > 1e-8


def test_periodic_full_coverage_has_frequency_local_limit(tmp_path):
    model = small_model(tmp_path/'case', True)
    model.ao.atm.heights[:] = 0
    model.strechFactor[:] = 1
    model.my_data_map['sensor_HO']['GridOffsets'] = [[0.,0.]]*model.nGs
    backend = FiniteApertureReconstructor(model)
    # Mathematical full-period measurement limit, not a physical finite-WFS
    # configuration switch. Full coverage supplies a canonical unitary DFT.
    n = backend.nbase; axis = (np.arange(n)-(n-1)/2)*backend.pitch
    x, y = np.meshgrid(axis, axis, indexing='ij')
    backend.cells = np.column_stack((x.ravel(),y.ravel()))
    backend.nv = n*n; backend.ncell = n; backend.span = 2*n-1
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    indices = np.column_stack((ix.ravel(),iy.ravel()))
    delta = indices[:,None]-indices[None]
    backend.didx = (delta[...,0]+n-1)*backend.span+delta[...,1]+n-1
    try:
        backend._covariances()
    except LinAlgError:
        # Structural DC-slope nulls are inverted only on the measured subspace.
        pass
    k = backend.cov_regular
    frequencies = np.column_stack((ix.ravel(),iy.ravel()))/(n*backend.pitch)
    e = np.exp(-2j*np.pi*(frequencies@backend.cells.T))/n
    transform = np.kron(np.eye(backend.nc), e)
    spectral = transform@k@transform.conj().T
    block = spectral.reshape(backend.nc,n*n,backend.nc,n*n).transpose(1,3,0,2)
    off = block.copy(); off[np.arange(n*n),np.arange(n*n)] = 0
    assert np.linalg.norm(off)/np.linalg.norm(block) < 2e-11
    target = sum(backend._target(l,slice(0,4)) for l in range(len(backend.heights)))
    eigen, vectors = np.linalg.eigh(k)
    keep = eigen > 64*np.finfo(float).eps*eigen[-1]
    global_weights = ((target@vectors[:,keep])/eigen[keep])@vectors[:,keep].T
    target_f = (target@transform.conj().T).reshape(4,backend.nc,n*n)
    local = np.zeros_like(target_f)
    for f in range(n*n):
        values, v = np.linalg.eigh(block[f,f])
        valid = values > 64*np.finfo(float).eps*eigen[-1]
        inverse = (v[:,valid]/values[valid])@v[:,valid].conj().T
        local[:,:,f] = target_f[:,:,f]@inverse
    local = local.reshape(4,-1)@transform
    difference = global_weights-local
    error = np.trace(difference@k@difference.conj().T).real
    scale = np.trace(global_weights@k@global_weights.conj().T).real
    assert error/scale < 1e-14


def test_builtin_extra_error_uses_same_stationary_pupil_fit_optics(tmp_path):
    from mastsel import psdSetToPsfSet
    from tiptop.finite_pupil import FinitePupilSampling
    model=small_model(tmp_path/'case', True)
    n=model.freq.nOtf;side=int(2*np.round(float(model.ao.tel.D)/2*n*model.freq.PSDstep))
    period=1/float(model.freq.kxAO_[1,0]-model.freq.kxAO_[0,0])
    sample=FinitePupilSampling(model.ao.tel.pupil,n,side,period,optical_n=n)
    baseline, _ = model.finite_aperture_covariance(sample,0)
    original=model.finiteApertureBackend.weights.copy()
    model.ao.tel.extraErrorNm=129.46549607941247
    extended, _ = model.finite_aperture_covariance(sample,0)
    backend=model.finiteApertureBackend
    np.testing.assert_array_equal(original,backend.weights)
    # Subtraction of the large projected-feedback covariance loses a few ulps.
    rounding = 8*np.finfo(float).eps*max(np.abs(baseline).max(), 1.)
    np.testing.assert_allclose(extended-baseline, backend.additional_covariance,
                               atol=rounding, rtol=1e-11)
    actual,_=sample.psf(backend.additional_covariance)
    norm=(2*model.freq.kcMax_/model.freq.resAO*model.freq.wvlRef*1e9/(2*np.pi))**2
    nm2=model.extraErrorPSD()*norm
    np.testing.assert_allclose(nm2.sum(),model.ao.tel.extraErrorNm**2,rtol=2e-15)
    expected=psdSetToPsfSet([nm2],model.ao.tel.pupil,[model.freq.wvlRef],
        nPixPup=side,freq_range=n*model.freq.PSDstep,dk=model.freq.dk_,
        nPixPsf=n,oversampling=1,full_flux=True,tip_tilt_removal='pupil_fit')[0].sampling
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-10)


def test_static_phase_does_not_change_reconstruction_or_dynamic_covariance(tmp_path):
    from tiptop.finite_pupil import FinitePupilSampling
    model = small_model(tmp_path/'case',True)
    n = model.freq.nOtf
    side = int(2*np.round(float(model.ao.tel.D)/2*n*model.freq.PSDstep))
    period = 1/float(model.freq.kxAO_[1,0]-model.freq.kxAO_[0,0])
    sampling = FinitePupilSampling(model.ao.tel.pupil,n,side,period)
    before,_ = model.finite_aperture_covariance(sampling,0)
    backend = model.finiteApertureBackend
    model.ao.tel.opdMap_on = np.sin(np.indices(model.ao.tel.pupil.shape)[0])*80
    after,_ = model.finite_aperture_covariance(sampling,0)
    assert model.finiteApertureBackend is backend
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('extra_error_nm', [0., 40.])
def test_residual_blocks_match_dense_covariance_and_keep_cache_ownership(tmp_path, extra_error_nm):
    from tiptop.finite_pupil import FinitePupilSampling
    model=small_model(tmp_path/'case',True)
    model.ao.tel.extraErrorNm=extra_error_nm
    n=model.freq.nOtf
    side=int(2*np.round(float(model.ao.tel.D)/2*n*model.freq.PSDstep))
    period=1/float(model.freq.kxAO_[1,0]-model.freq.kxAO_[0,0])
    pupil=model.ao.tel.pupil
    phase=.2*np.sin(np.indices(pupil.shape)[0])
    sample=FinitePupilSampling(pupil,n,side,period,static_phase=phase)
    residual,_=model.finite_aperture_residual(sample,0)
    assert model.finiteApertureBackend._source_covariance is None
    snapshots=[(first,block.copy()) for first,block in residual.blocks(7)]
    assert model.finiteApertureBackend._source_covariance is None
    dense,_=model.finite_aperture_covariance(sample,0)
    np.testing.assert_allclose(residual.diagonal,np.diag(dense),atol=2e-12,rtol=2e-12)
    for first,block in snapshots:
        np.testing.assert_allclose(block,dense[first:first+len(block),first:],atol=2e-12,rtol=2e-12)
    a,_=sample.psf(dense)
    b,_=sample.psf_blocks(residual.diagonal,residual.blocks(7))
    np.testing.assert_allclose(a,b,atol=2e-13,rtol=2e-10)
    other=FinitePupilSampling(pupil,n,side,period)
    model.finite_aperture_covariance(other,0)
    for (_,saved),(_,block) in zip(snapshots,residual.blocks(7),strict=True):
        np.testing.assert_array_equal(block,saved)

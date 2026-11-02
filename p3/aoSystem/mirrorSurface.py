"""Mirror selection and finite Gaussian phase fitting in physical coordinates.

PathInfGeometry optionally overrides the regular mirror geometry with a
FITS input, not a calibration matrix or estimator. Downstream owns custom
padding and actuator selection. All
responses here are unit phase per unit phase command; the phase fit cutoff is
in those dimensionless units. Optical and WFS coordinates are metres.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy import linalg, special
from astropy.io import fits

# Unit-phase surface-fit singular-value threshold. Part of the finite model.
_PHASE_FIT_CUTOFF = 5e-4


def geometry_signature(section, resolve_path):
    """Hash an explicit override, or return None for geometry derived from inputs."""
    path = section.get('PathInfGeometry')
    if path is None or path == '':
        return None
    if not isinstance(path, str):
        raise ValueError('PathInfGeometry must be a FITS path or empty')
    return hashlib.sha256(Path(resolve_path(path)).read_bytes()).hexdigest()


def _read_geometry(path):
    """Read the explicit FITS layout and its calibration coordinates."""
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].header.get('GEOMVER') != 1:
            raise ValueError('PathInfGeometry requires GEOMVER=1')
        expected = {'PRIMARY', 'ACTUATOR_AXIS', 'VALID_ACTUATORS', 'PUPIL_X', 'PUPIL_Y'}
        if {hdu.name for hdu in hdus} != expected or len(hdus) != len(expected):
            raise ValueError('Geometry FITS requires ACTUATOR_AXIS, VALID_ACTUATORS, PUPIL_X, PUPIL_Y only')
        axis = np.asarray(hdus['ACTUATOR_AXIS'].data, float)
        valid = np.asarray(hdus['VALID_ACTUATORS'].data)
        pupil_x = np.asarray(hdus['PUPIL_X'].data, float)
        pupil_y = np.asarray(hdus['PUPIL_Y'].data, float)
        for name in ('ACTUATOR_AXIS', 'PUPIL_X', 'PUPIL_Y'):
            if hdus[name].header.get('BUNIT') != 'm':
                raise ValueError(f'{name} requires BUNIT=m')
    return axis, valid, pupil_x, pupil_y


def _regular_geometry(actuators, mirror_pitch, diameter, pupil):
    """Centred nominal grid, retaining centres on the sampled illuminated pupil.

    The telescope's endpoint-inclusive physical coordinates define calibration.
    Actuator selection uses nearest pupil samples; padding or other selection
    rules require an explicit geometry override.
    """
    if (not np.isfinite(actuators) or actuators < 2 or int(actuators) != actuators
            or not np.isfinite(mirror_pitch) or mirror_pitch <= 0
            or not np.isfinite(diameter) or diameter <= 0):
        raise ValueError('Regular geometry requires integer actuator count >= 2 and positive pitch/diameter')
    axis = (np.arange(int(actuators))-(actuators-1)/2)*mirror_pitch
    pupil_x = np.linspace(-diameter/2, diameter/2, pupil.shape[1])
    pupil_y = np.linspace(-diameter/2, diameter/2, pupil.shape[0])
    x, y = np.meshgrid(axis, axis)
    ix = np.rint((x/diameter+.5)*(pupil.shape[1]-1)).astype(int)
    iy = np.rint((y/diameter+.5)*(pupil.shape[0]-1)).astype(int)
    inside = (abs(x) <= diameter/2) & (abs(y) <= diameter/2)
    valid = inside & (pupil[np.clip(iy, 0, pupil.shape[0]-1),
                           np.clip(ix, 0, pupil.shape[1]-1)] > 0)
    return axis, valid, pupil_x, pupil_y


def phase_fit_points(cells,pitch):
    """Union of closed 3x3 phase nodes of nominal valid square lenslets."""
    lower=cells.min(0)-pitch/2
    indices=np.rint((cells-lower)/pitch-.5).astype(int)
    nodes=set()
    for x,y in indices:
        nodes.update((2*x+i,2*y+j) for i in range(3) for j in range(3))
    return np.array(sorted(nodes))*pitch/2+lower


class GaussianSurface:
    """Finite actuators, raw WFS-phase fit and subsequent pupil-tilt projection.

    An optional FITS file supplies actuator coordinates, selection and pupil
    calibration coordinates. Otherwise derive a regular grid from the nominal
    mirror and telescope inputs. The telescope owns pupil weights; the INI owns coupling.
    Array columns use row-major
    (y,x) actuator-mask order. Current validation covers centred Cartesian
    Gaussian surfaces. Explicit geometry overrides nominal counts and pitches.
    No Fourier correction is used to create this estimator.
    """
    def __init__(self, section, cells, pitch, pupil, resolve_path, *,
                 diameter, actuators, mirror_pitch):
        if section.get('InfModel', 'gaussian') != 'gaussian':
            raise ValueError("Finite aperture currently requires InfModel='gaussian'")
        coupling = section.get('InfCoupling', [0.2])
        if (not isinstance(coupling, (list, tuple)) or len(coupling) != 1
                or isinstance(coupling[0], bool) or not isinstance(coupling[0], (int, float))
                or not 0 < coupling[0] < 1):
            raise ValueError('InfCoupling requires one Gaussian coupling in (0,1)')
        self.coupling = float(coupling[0])
        self.cutoff = _PHASE_FIT_CUTOFF
        self.signature = geometry_signature(section, resolve_path)
        pupil = np.asarray(pupil, float)
        if (pupil.ndim != 2 or min(pupil.shape) < 2 or not np.isfinite(pupil).all()
                or np.any(pupil < 0) or not pupil.any()):
            raise ValueError('Invalid telescope pupil for Gaussian calibration')
        if self.signature is None:
            geometry = _regular_geometry(actuators, mirror_pitch, diameter, pupil)
        else:
            geometry = _read_geometry(resolve_path(section['PathInfGeometry']))
        self.axis, valid, self.pupil_x, self.pupil_y = geometry
        for axis in (self.axis, self.pupil_x, self.pupil_y):
            if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                    or np.diff(axis)[0] <= 0
                    or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-10, atol=1e-13)
                    or not np.isclose(axis[0], -axis[-1], rtol=1e-10, atol=1e-13)):
                raise ValueError('Geometry axes must be finite, centred, increasing uniform coordinates')
        if (valid.shape != (len(self.axis),)*2 or not np.isin(valid, [0, 1]).all()
                or not valid.any()):
            raise ValueError('VALID_ACTUATORS must be a nonempty binary y/x grid')
        self.valid = valid.astype(bool)
        # The telescope owns the actual pupil. Geometry only specifies the
        # calibration coordinates and any surrounding zero padding.
        padding = np.array([len(self.pupil_y), len(self.pupil_x)]) - pupil.shape
        if np.any(padding < 0) or np.any(padding % 2):
            raise ValueError('Geometry calibration grid must fit the telescope pupil with centred even padding')
        self.pupil = np.pad(pupil, [(int(v//2), int(v//2)) for v in padding])
        self.pitch=float(self.axis[1]-self.axis[0]);self.iy,self.ix=np.nonzero(self.valid)
        self.centres=np.column_stack((self.axis[self.ix],self.axis[self.iy]))
        self.fit_points=phase_fit_points(cells,pitch)
        response=self.basis(self.fit_points)
        u,s,vt=linalg.svd(response,full_matrices=False,check_finite=False)
        keep=s>self.cutoff
        if not keep.any():raise ValueError('Gaussian phase-fit cutoff removes all actuators')
        self.raw_fit=(vt[keep].T/s[keep])@u[:,keep].T
        self.projection,self.calibration_error=self._pupil_projection()
        self.fit=self.projection@self.raw_fit
        self.retained_modes=int(keep.sum());self.singular_values=s

    def basis(self,points):
        points=np.asarray(points,float)
        x=np.exp(np.log(self.coupling)*((points[:,0,None]-self.axis)/self.pitch)**2)
        y=np.exp(np.log(self.coupling)*((points[:,1,None]-self.axis)/self.pitch)**2)
        return x[:,self.ix]*y[:,self.iy]

    def _pupil_projection(self):
        """Exact separable Gram products avoid a giant point/actuator SVD.

        Solve the two ramp fits in the actual supplied pupil, then remove
        their fitted coefficients after the WFS-domain fit. This is the same
        least-squares projection as explicit unit-phase influence matrices.
        Ill-conditioned pupil calibration is rejected rather than regularized.
        """
        c=np.log(self.coupling)/self.pitch**2
        x=np.exp(c*(self.pupil_x[:,None]-self.axis)**2)
        y=np.exp(c*(self.pupil_y[:,None]-self.axis)**2)
        xx=(x[:,:,None]*x[:,None,:]).reshape(len(x),-1)
        yy=(y[:,:,None]*y[:,None,:]).reshape(len(y),-1)
        n=len(self.axis)
        product=(xx.T@self.pupil.T@yy).reshape(n,n,n,n)
        gram=product[self.ix[:,None],self.ix[None,:],self.iy[:,None],self.iy[None,:]]
        mx,my=np.meshgrid(self.pupil_x,self.pupil_y)
        weight=self.pupil/self.pupil.sum()
        ramps=np.stack((mx-(mx*weight).sum(),my-(my*weight).sum()))
        cross=np.array([(x.T@(self.pupil*r).T@y)[self.ix,self.iy] for r in ramps]).T
        tilt=np.einsum('iyx,jyx,yx->ij',ramps,ramps,self.pupil)
        try:fit=linalg.solve(gram,cross,assume_a='pos',check_finite=False)
        except linalg.LinAlgError as exc:raise ValueError('Gaussian pupil calibration is rank deficient') from exc
        error=np.linalg.norm(gram@fit-cross)/np.linalg.norm(cross)
        if not np.isfinite(fit).all() or error>1e-9:
            raise ValueError('Gaussian pupil calibration solve failed its normal equation')
        return np.eye(len(self.ix))-fit@linalg.solve(tilt,cross.T,assume_a='pos'),float(error)

    def slopes(self,cells,pitch,offsets):
        """Exact square-cell average of Gaussian gradients, before tilt fitting."""
        beta=np.sqrt(-np.log(self.coupling))/self.pitch
        result=[]
        for offset in offsets:
            d=cells[:,None,:]+offset-self.centres[None]
            left=d-pitch/2;right=d+pitch/2
            integral=np.sqrt(np.pi)/(2*beta*pitch)*(special.erf(beta*right)-special.erf(beta*left))
            edge=(np.exp(-(beta*right)**2)-np.exp(-(beta*left)**2))/pitch
            result.extend((edge[:,:,0]*integral[:,:,1],integral[:,:,0]*edge[:,:,1]))
        return np.concatenate(result)

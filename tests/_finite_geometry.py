"""Small physical FITS geometry for finite-model integration tests."""
import numpy as np
from astropy.io import fits


def write_geometry(path, pupil_size, count=5):
    axis = np.linspace(-4, 4, count)
    x, y = np.meshgrid(axis, axis)
    valid = np.hypot(x, y) <= 4.5
    primary = fits.PrimaryHDU(); primary.header['GEOMVER'] = 1
    hdus = [primary]
    for name, values in [('ACTUATOR_AXIS', axis), ('VALID_ACTUATORS', valid.astype('uint8')),
                         ('PUPIL_X', np.linspace(-4, 4, pupil_size)),
                         ('PUPIL_Y', np.linspace(-4, 4, pupil_size))]:
        hdu = fits.ImageHDU(values, name=name)
        if name != 'VALID_ACTUATORS': hdu.header['BUNIT'] = 'm'
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)

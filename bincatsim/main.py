from .simulator import GaiaSimulator, CCD
from .core.root import PSF_DATA_PATH
from astropy import units as u


def main(M1: float, M2: float, distance: int, angle: float, ccd: 'CCD' =None):
    if ccd is None:
        ccd = CCD(psf=PSF_DATA_PATH+"/1062x2124_gpsf.fits", pixel_scale_x = 177*u.mas, pixel_scale_y = 59*u.mas)

    bs = GaiaSimulator(ccd=ccd, M1=M1, M2=M2, distance=distance, angle=angle)

    bs.observe(ccd)

if __name__ == "__main__":
    main()

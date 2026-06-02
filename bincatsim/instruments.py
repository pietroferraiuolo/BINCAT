import xupy as xp
import numpy as _np
import poppy as _poppy # type: ignore
import astropy.units as _u
from . import utils as _ut
from xupy import typings as _xt
import matplotlib.pyplot as _plt
from astropy.table import QTable as _qt
from .utils.psfutils import fits as _fits
from .utils.logging import SystemLogger as _SL
from .core.root import BANDS_FILE, PASSBAND_FILE


class CCD:
    """
    Class to create a sky map with a star at the center and Poisson noise.

    Parameters
    ----------
    psf : astropy.io.fits.HDUList or array-like, optional
        The PSF of the Gaia telescope. If not provided, it can be computed using
        the method 'compute_psf' with a `poppy` optical system.
    pixel_area : astropy.units.Quantity, optional
        The area of a single pixel in square microns. Default is 10x30 microns.
    ccd_pixels : list of astropy.units.Quantity, optional
        The number of pixels in the CCD. Default is [4500, 1966] for Gaia.
    pixel_scale_x : astropy.units.Quantity, optional
        The pixel scale in the x direction in arcseconds per pixel. Default is 0.059 arcsec/pixel.
    pixel_scale_y : astropy.units.Quantity, optional
        The pixel scale in the y direction in arcseconds per pixel. Default is 0.059*3 arcsec/pixel.
    t_integration : astropy.units.Quantity, optional
        The integration time per CCD in seconds. Default is 4.42 seconds.
    """

    def __init__(
        self,
        psf: _xt.Optional[_fits.HDUList | _xt.ArrayLike | str] = None,
        **kwargs: dict[str, _xt.Any],
    ):
        """The constructor"""
        self._logger = _SL(__class__)

        if psf is not None:
            is_array = False
            if isinstance(psf, _fits.HDUList):
                self._psf = psf
                self._meta = psf[0].header
            elif isinstance(psf, str):
                with _fits.open(psf) as hdul:
                    self._psf = None
                    self._meta = hdul[0].header.copy()
                    self.psf = hdul[0].data.copy()
            elif isinstance(psf, (xp.ndarray, _np.ndarray)):
                self._psf = None
                is_array = True
            else:
                raise TypeError(
                    "`psf` must be an astropy.io.fits.HDUList or a string path to a FITS file."
                )
            if not isinstance(psf, str):
                self.psf = self._psf[0].data if not is_array else psf
            self.psf_x, self.psf_y = _ut.computeXandYpsf(self.psf)
            self._logger.info("PSF Loaded")
        else:
            self._psf = self.psf = None
            self.psf_x = None
            self.psf_y = None
            self._meta = None
            print(
                """PSF not provided. Use the method 'compute_psf' to compute it, passing a
`poppy` optical system."""
            )

        self._bands = _qt.read(BANDS_FILE)
        self._bands = self._bands[self._bands["band"] == kwargs.get("band", "Gaia_G")]
        self._passbands = _qt.read(PASSBAND_FILE)
        
        ## CCD Specifications
        self.pixel_area = kwargs.get("pixel_area", 10 * 30 * _u.um**2)
        self.ccd_pixels = kwargs.get("ccd_pixels", [4500 * _u.pixel, 1966 * _u.pixel])
        self.ccd_pxscale_x = kwargs.get("pixel_scale_x", 59 * _u.mas / _u.pixel)
        self.ccd_pxscale_y = kwargs.get("pixel_scale_y", 177 * _u.mas / _u.pixel)
        self.ccd_pxscale_factor = (self.ccd_pxscale_y / self.ccd_pxscale_x).value
        self.full_well_capacity = 190000 * _u.electron
        self.gain = kwargs.get("gain", 2 * _u.electron / _u.adu)
        self.fov = (self.ccd_pixels[0] * self.ccd_pxscale_x, self.ccd_pixels[1] * self.ccd_pxscale_y)

        ### Primary mirror's Area
        self.integration_area = 1.45*0.5 * _u.m**2
        ### TDI gates on CCD, based on source Magnitude
        self._TDI_clock_rate = 0.9828e-3 * _u.s
        self.TDIGates = _np.array([4494, 2906, 2057, 1030, 512, 256, 128, 64, 32, 16, 8, 4, 2])
        self.integration_time = self.TDIGates * self._TDI_clock_rate

        ## PSF Specifications
        self.psf_pxscale_x = kwargs.get(
            "psf_pixel_scale_x", 1 * _u.mas / _u.pixel
        )
        self.psf_pxscale_y = kwargs.get(
            "psf_pixel_scale_y", 1 * _u.mas / _u.pixel
        )

        self.rebinned = False
        self.WC = {
            0: {
                "area_px": (18, 12),
                "area_um": 18 * 12 * 300 * _u.um**2,
            },
            1: {
                "area_px": (18, 1),
                "area_um": 18 * 300 * _u.um**2,
            },
            2: {
                "area_px": (12, 1),
                "area_um": 12 * 300 * _u.um**2,
            },
        }
        self._wc_conditions = ["{G}<13", "13<={G}<16", "16<={G}"]

    @property
    def header(self) -> _fits.Header | None:
        """
        Returns the header of the PSF FITS file if available.
        """
        return self._meta

    def rebin_psf(
        self,
        psf: _xt.Optional[_fits.HDUList | _xt.ArrayLike] = None,
        *,
        rebin_factor: int = 2,
        axis_ratio: tuple[int, int] = (1, 1),
    ) -> None:
        """
        Rebin the PSF by a given factor, following Gaia's pixel scale (1:3 ratio).

        Parameters
        ----------
        rebin_factor : int
            The factor by which to rebin the PSF.
        axis_ratio : tuple of int
            The ratio of the pixel scales in the y and x directions. Default is (1, 3).
        """
        self._logger.info("Rebinning CCD PSF...")
        px_ratio = (rebin_factor * axis_ratio[0], rebin_factor * axis_ratio[1])
        psf = self.psf
        if psf is None:
            raise ValueError("PSF has not been computed yet.")
        if not isinstance(rebin_factor, int) or rebin_factor < 1:
            raise ValueError("rebin_factor must be a positive integer.")
        if not (
            isinstance(axis_ratio, tuple)
            and len(axis_ratio) == 2
            and all(isinstance(i, int) and i > 0 for i in axis_ratio)
        ):
            raise ValueError("axis_ratio must be a tuple of two positive integers.")
        self._psf_bk = self.psf.copy()
        self.psf = _poppy.utils.rebin_array(psf, px_ratio)
        self.psf_x, self.psf_y = _ut.computeXandYpsf(self.psf)
        self.rebinned = True
        self._logger.info("...Completed.")
        print("Rebinning complete")

    def sample_psf(self, psf: _xt.ArrayLike, G: float|_u.Quantity) -> _xt.ArrayLike:
        """
        Sample the PSF to match the CCD pixel scale.

        Parameters
        ----------
        psf : array-like
            The PSF to be sampled.
        G : float or astropy.units.Quantity
            The G-band magnitude of the source, used to determine the appropriate
            WC window size for sampling.

        Returns
        -------
        list of arrays
            The sampled PSF, in the order: (psf_2d, psf_x, psf_y).
        """
        self._logger.info("Sampling PSF to match CCD pixel scale...")
        for i, condition in enumerate(self._wc_conditions):
            if eval(condition.format(G=G)):
                wc_px = self.WC[i]["area_px"]
                break
        else:
            raise ValueError(
                f"Invalid G magnitude: {G}. Must be a float or astropy Quantity."
            )
        rbx = psf.shape[1] // wc_px[0]
        rby = psf.shape[0] // wc_px[1]
        rbratio = (rby, rbx) if self.ccd_pxscale_factor > 1 else (rbx, rby)
        psf_2d = _poppy.utils.rebin_array(psf, rbratio)
        psf_x, psf_y = _ut.computeXandYpsf(psf=psf_2d)
        return psf_2d, psf_x, psf_y

    def display_psf(
        self,
        mode: str = "2d",
        **kwargs: dict[str, _xt.Any],
    ) -> None:
        """Display the PSF of the Gaia telescope.

        Parameters
        ----------
        mode : str, optional
            The mode of display. Options are '2d' for 2D display and 'x' or 'y' for
            relative axes PSFs.
            Default is '2d'.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the display function (`plt.imshow`).
        """
        _ut.display_psf(self.psf, self.psf_x, self.psf_y, mode=mode, **kwargs)


    def __repr__(self):
        """String representation of the CCD."""
        return f"""          e2v™ CCD91-72
          -------------
            Band: {self._bands['band'][0]}
      Pixel area: {self.pixel_area}
     Pixel Scale: {self.ccd_pxscale_x.to_value(_u.mas/_u.pix):.1f} x {self.ccd_pxscale_y.to_value(_u.mas/_u.pix):.1f} {_u.mas/_u.pix}
        CCD fov : {self.ccd_pixels[0].value:.0f} x {self.ccd_pixels[1].value:.0f}  {self.ccd_pixels[0].unit}
               -> {self.fov[0].to_value(_u.arcmin):.2f} x {self.fov[1].to_value(_u.arcmin):.2f} {_u.arcmin}
Integration time: {self.integration_time[0]:.2f}
       PSF shape: {list(self.psf.shape) if self.psf is not None else 'Not computed yet'} {"Rebinned" if self.rebinned else ""}
    """

    def display_passbands(self) -> None:
        """
        Display the passbands of the CCD.
        """
        _plt.figure(figsize=(8, 6))
        _plt.grid(linestyle="--", alpha=0.85)
        _plt.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["RP"],
            c="red",
            label="RP",
            linestyle="--",
        )
        _plt.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["BP"],
            c="blue",
            label="BP",
            linestyle="--",
        )
        _plt.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["G"],
            c="green",
            label="G",
            linewidth=2,
        )
        _plt.xlabel("Wavelength [μm]")
        _plt.ylabel("Throughput")
        _plt.title("Gaia DR3 Passbands")
        _plt.yticks(_np.arange(0, 0.85, 0.1))
        _plt.legend()
        _plt.show()



class Band:
    
    def __init__(
        self, 
        name: str,
        wavelength: _u.Quantity|None = None,
        transmission: _xt.Array|None = None,
        zero_point: float|None = None
    ):
        self.name = name

        if not self._resolved_band():
            self._wavelength = wavelength
            self._transmission = transmission
            self._zero_point = zero_point
        
        if all(
            [
                x is None 
                for x in [self._wavelength, self._transmission, self._zero_point]
            ]
        ):
            raise ValueError(
                f"Could not resolve band '{name}' from database, and no parameters provided."
            )

    def _resolved_band(self) -> bool:
        if self.name is None:
            return False
        bands = _qt.read(BANDS_FILE)
        band_info = bands[[x == self.name.lower() for x in map(str.lower, bands['band'])]]
        if len(band_info) == 0:
            return False
        self._zero_point = band_info["zero_point"][0]
        self._wleff = band_info["wavelength"][0].to(_u.nm)
        
        passband = _qt.read(PASSBAND_FILE)
        self._wavelength = passband["lambda"].to(_u.nm)
        self._transmission = passband['G'].filled(0)

        return True
    
    @property
    def wavelength(self) -> _u.Quantity:
        return self._wavelength
    
    @property
    def transmission(self) -> _xt.Array:
        return self._transmission
    
    @property
    def zero_point(self) -> float:
        return self._zero_point
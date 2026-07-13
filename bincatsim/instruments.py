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
        self.ccd_pxscale_y = kwargs.get("pixel_scale_y", 59 * _u.mas / _u.pixel)
        self.ccd_pxscale_x = kwargs.get("pixel_scale_x", 177 * _u.mas / _u.pixel)
        self.ccd_pxscale_factor = (self.ccd_pxscale_y / self.ccd_pxscale_x).value
        self.full_well_capacity = 190000 * _u.electron
        self.gain = kwargs.get("gain", 2 * _u.electron / _u.adu)
        self.fov = (self.ccd_pixels[0] * self.ccd_pxscale_x, self.ccd_pixels[1] * self.ccd_pxscale_y)

        ### Primary mirror's Area
        self.integration_area = 1.45*0.5 * _u.m**2

        ### TDI gates on CCD, based on source Magnitude
        self.integration_time = None
        self._clock_rate = 0.9828e-3 * _u.s
        self._tdi_gates = {
            #  thr Mag   : TDI Lines used
                16.5     :     4494,
                15.75    :     2057,
                15       :     1030,
                14.25    :     512,
                13.5     :     256,
                12.75    :     128,
                10.5     :     16,
                8.25     :     2
        } # Derived empirically by computing total flux for a given magnitude at
            # each TDI gate and comparing to the full well capacity of the CCD.
            # which is in electrons, and i'm doing it in photons... But not really
            # since in the passband is included the CCD QE, so it's more like 
            # electrons. It should be fine.

        ## PSF Specifications
        self.psf_pxscale_x = kwargs.get(
            "psf_pixel_scale_x", 1 * _u.mas / _u.pixel
        )
        self.psf_pxscale_y = kwargs.get(
            "psf_pixel_scale_y", 1 * _u.mas / _u.pixel
        )

        ## CCD Sampling Specifications
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
        self._window = None

    def set_exposure_time(self, G: float|_u.Quantity) -> None:
        """
        Set the integration time based on the G-band magnitude of the source.

        Parameters
        ----------
        G : float or astropy.units.Quantity
            The G-band magnitude of the source, used to determine the appropriate
            integration time based on TDI gates.
        """
        for M, L in self._tdi_gates.items():
            if eval(f"{G} >= {M}"):
                self.integration_time = L * self._clock_rate
                break
        if self.integration_time is None:
            self._saturated = True
            self.integration_time = min(self._tdi_gates.values()) * self._clock_rate
            self._logger.warning(
                f"Source with G={G} is brighter than the saturation threshold. "
                f"Setting integration time to maximum ({self.integration_time.to_value(_u.s):.2f} s) and flagging as saturated."
            )
        else:
            self._saturated = False
            self._logger.info(
                f"Integration time set to {self.integration_time.to_value(_u.s):.2f} s based on G={G}."
            )

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
                self._window = f"wc{i}"
                break
        else:
            raise ValueError(
                f"Invalid G magnitude: {G}. Must be a float or astropy Quantity."
            )
        rbx = psf.shape[1] // wc_px[0]
        rby = psf.shape[0] // wc_px[1]
        rbratio = (rby, rbx) if self.ccd_pxscale_factor < 1 else (rbx, rby)
        psf_2d = _poppy.utils.rebin_array(psf, rbratio)
        psf_x, psf_y = _ut.computeXandYpsf(psf_2d, window=self._window)
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

    def display_passbands(self) -> None:
        """
        Display the passbands of the CCD.
        """
        fig, ax = _plt.subplots(figsize=(8, 6))
        ax.grid(linestyle="--", alpha=0.85)
        ax.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["RP"],
            c="red",
            label="RP",
            linestyle="--",
        )
        ax.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["BP"],
            c="blue",
            label="BP",
            linestyle="--",
        )
        ax.plot(
            self._passbands["lambda"] / 1000,
            self._passbands["G"],
            c="green",
            label="G",
            linewidth=2,
        )
        ax.set_xlabel("λ [μm]", fontsize=16)
        ax.set_ylabel("Throughput", fontsize=16)
        ax.set_title("Gaia DR3 Passbands", fontsize=20)
        ax.set_yticks(_np.arange(0, 0.85, 0.1))
        ax.set_yticklabels([f"{x:.1f}" for x in _np.arange(0, 0.85, 0.1)], fontsize=14)
        ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()], fontsize=14)
        ax.legend()
        _plt.show()


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
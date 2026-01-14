import xupy as xp
import numpy as _np
import poppy as _poppy
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
            if isinstance(psf, _fits.HDUList):
                self._psf = psf
                self._meta = psf[0].header
            elif isinstance(psf, str):
                self._psf = _fits.open(psf)
                self._meta = self._psf[0].header
            else:
                raise TypeError(
                    "`psf` must be an astropy.io.fits.HDUList or a string path to a FITS file."
                )
            self.psf = self._psf[0].data
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
        self.pixel_area = kwargs.get("pixel_area", 10 * 30 * _u.um**2)
        self.ccd_pixels = kwargs.get("ccd_pixels", [4500 * _u.pixel, 1966 * _u.pixel])
        self.pxscale_x = kwargs.get("pixel_scale_x", 59 * _u.mas / _u.pixel)
        self.pxscale_y = kwargs.get("pixel_scale_y", 177 * _u.mas / _u.pixel)
        self.pxscale_factor = (self.pxscale_y / self.pxscale_x).value
        self.tdi = kwargs.get("t_integration", 4.42 * _u.s)
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

    def sample_psf(self, psf: _xt.ArrayLike) -> _xt.ArrayLike:
        """
        Sample the PSF to match the CCD pixel scale.

        Parameters
        ----------
        psf : array-like
            The PSF to be sampled.

        Returns
        -------
        list of arrays
            The sampled PSF, in the order: (psf_2d, psf_x, psf_y).
        """
        self._logger.info("Sampling PSF to match CCD pixel scale...")
        if self.pxscale_factor < 1:
            rbfactor = int(self.pxscale_y.value)
            ratio = int(1 / self.pxscale_factor)
            rbratio = (rbfactor * ratio, rbfactor)
        else:
            rbfactor = int(self.pxscale_x.value)
            ratio = int(self.pxscale_factor)
            rbratio = (rbfactor, rbfactor * ratio)
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
        if self._psf is None:
            raise ValueError("PSF has not been computed yet.")
        if mode == "2d":
            cmap = kwargs.pop("cmap", "gist_heat")
            from astropy.visualization import (
                ImageNormalize,
                MinMaxInterval,
                LogStretch,
            )

            norm = ImageNormalize(
                vmin=xp.np.nanmin(self.psf),
                vmax=xp.np.nanmax(self.psf),
                stretch=LogStretch(500),
                interval=MinMaxInterval(),
            )
            fig = _plt.figure()
            _plt.imshow(self.psf, cmap=cmap, norm=norm, **kwargs)
            _plt.colorbar()
            _plt.title("CCD PSF")
            _plt.xlabel("X [px]")
            _plt.ylabel("Y [px]")
        else:
            fig = _plt.figure()
            _plt.xlabel("arcsec")
            _plt.ylabel("Normalized PSF")
            _plt.grid(linestyle="--")
            y = _np.arange(len(self.psf_y)) * _u.pixel
            y -= len(y) // 2 * _u.pixel  # center the y-axis
            y *= self.pxscale_y
            x = _np.arange(len(self.psf_x)) * _u.pixel
            x -= len(x) // 2 * _u.pixel  # center the x-axis
            x *= self.pxscale_x
            _plt.title(f"PSF in {mode} direction")
            if mode == "x":
                _plt.plot(x, self.psf_x)
            elif mode == "y":
                _plt.plot(y, self.psf_y)
            else:
                raise ValueError("Invalid mode. Use '2d', 'x', or 'y'.")
        _plt.show()
        return fig

    def __repr__(self):
        """String representation of the CCD."""
        return f"""          e2v™ CCD91-72
          -------------
            Band: {self._bands['band'][0]}
      Pixel area: {self.pixel_area}
     Pixel Scale: {self.pxscale_x.to_value(_u.mas/_u.pix):.1f} x {self.pxscale_y.to_value(_u.mas/_u.pix):.1f} {_u.mas/_u.pix}
        CCD fov : {self.ccd_pixels[0].value:.0f} x {self.ccd_pixels[1].value:.0f}  {self.ccd_pixels[0].unit}
               -> {self.fov[0].to_value(_u.arcmin):.2f} x {self.fov[1].to_value(_u.arcmin):.2f} {_u.arcmin}
Integration time: {self.tdi}
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
            self._passbands["G"],
            c="green",
            label="G",
        )
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
        _plt.xlabel("Wavelength [μm]")
        _plt.ylabel("Throughput")
        _plt.title("Gaia DR3 Passbands")
        _plt.yticks(_np.arange(0, 0.85, 0.1))
        _plt.legend()
        _plt.show()


class GaiaTelescope:
    """
    Class to simulate the Gaia telescope PSF.
    This class creates an optical system with the parameters of the Gaia
    telescope and computes the PSF for a given wavelength.

    Telescope Parameters
    --------------------
    aperture_width : float, optional
        The width of the aperture in meters. Default is 1.5 m.
    aperture_height : float, optional
        The height of the aperture in meters. Default is 0.5 m.
    pixel_scale_x : float, optional
        The pixel scale in the x direction in arcseconds per pixel. Default is 0.05 arcsec/pixel.
    pixel_scale_factor : int, optional
        The factor to scale the pixel size in the y direction. Default is 2.
    field_of_view : float, optional
        The field of view in arcseconds. Default is 5 arcsec. A non square field of view
        can be used.
    wavelength_pfs : float, optional
        The wavelength for the PSF calculation in meters. Default is 550 nm.

    CCD Parameters
    --------------
    band : str, optional
        The band for which the CCD is initialized. Options are:
        - Gaia_G
        - Gaia_BP
        - Gaia_RP
    ccd_pixels : list, optional
        The number of pixels in the CCD. Default is [4500, 1966] for Gaia.
    pixel_area : float, optional
        The area of a single pixel in square microns. Default is 10x30 microns.
    ccd_int_time : float, optional
        The integration time per CCD in seconds. Default is 4.42 seconds.
    """

    def __init__(self, **kwargs: dict[str, _xt.Any]):
        """The constructor"""
        self.speed = 59.9641857803 * _u.arcsec / _u.s
        self.period = (6 * _u.hour).to(_u.s)
        self.precession_period = (63 * _u.day).to(_u.s)
        self.__create_optical_system(**kwargs)

    def __repr__(self):
        """String representation of the Gaia Telescope."""
        return f"""           Gaia Telescope V0
           -----------------
          Aperture: {self.apert_w} x {self.apert_h}
    Scansion speed: {self.speed:.2f}
    Orbital Period: {self.period.to(_u.hour)}
 Precession period: {self.precession_period.to(_u.day)}
Wavelength for PSF: {self.wavel_pfs}
"""

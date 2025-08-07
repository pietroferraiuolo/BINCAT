import numpy as _np
import poppy as _poppy
import astropy.units as _u
from astropy import convolution as _c
from typing import Any as _any
from astropy.table import QTable as _qt
import matplotlib.pyplot as _plt


class GaiaTelescopeV0():
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

    def __init__(self, **kwargs: dict[str, _any]):
        """The constructor"""
        self.speed = 59.9641857803 * _u.arcsec / _u.s
        self.period = (6 * _u.hour).to(_u.s)
        self.precession_period = (63 * _u.day).to(_u.s)
        self.__create_optical_system(**kwargs)
        self.psf = self._compute_psf()


    def display_psf(self, mode: str = "2d", **kwargs: dict[str, _any]) -> None:
        """Display the PSF of the Gaia telescope.

        Parameters
        ----------
        mode : str, optional
            The mode of display. Options are '2d' for 2D display and 'x' or 'y' for
            relative axes PSFs.
            Default is '2d'.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the display function.
        """
        if self.psf is None:
            raise ValueError("PSF has not been computed yet.")
        if mode == "2d":
            _poppy.display_psf(self._psf, title="Gaia Telescope PSF", **kwargs)
            return
        _plt.figure()
        _plt.xlabel("acrsec")
        _plt.ylabel("Normalized PSF")
        _plt.grid(linestyle="--")
        y = _np.arange(len(self.psf_y)) * _u.pixel
        y -= len(y) // 2 * _u.pixel  # center the y-axis
        y *= self.pixscale_y
        x = _np.arange(len(self.psf_x)) * _u.pixel
        x -= len(x) // 2 * _u.pixel  # center the x-axis
        x *= self.pixscale_x
        _plt.plot(x, self.psf_x)
        if mode == "x":
            _plt.plot(x, self.psf_x)
        elif mode == "y":
            _plt.plot(y, self.psf_y)
        else:
            raise ValueError("Invalid mode. Use '2d', 'x', or 'y'.")
        _plt.show()
        _plt.title(f"Gaia Telescope PSF in {mode} direction")


    def _compute_psf(self):
        """Compute the PSF of the Gaia telescope."""
        self._psf = self._osys.calc_psf(self.wavel_pfs)
        img = self._psf[0].data
        final_psf = _poppy.utils.rebin_array(img, (1, self.pscale_fact))
        psf_x = _np.sum(final_psf, axis=1)
        psf_x /= _np.sum(psf_x)  # normalize
        psf_y = _np.sum(final_psf, axis=0)
        psf_y /= _np.sum(psf_y)
        self.psf_x = psf_x
        self.psf_y = psf_y
        return final_psf


    def __create_optical_system(self, **kwargs: dict[str, _any]):
        """Create the optical system for the Gaia telescope."""
        osys = _poppy.OpticalSystem()
        # Getting the optical parameters from kwargs
        # If not passed, the default values will be used.
        self.apert_w = (
            _get_kwargs(
                ("aperture_width", "width", "aperture_w", "apert_w"), 1.5, kwargs
            )
            * _u.m
        )
        self.apert_h = (
            _get_kwargs(
                ("aperture_height", "height", "aperture_h", "apert_h"), 0.5, kwargs
            )
            * _u.m
        )
        self.pixscale_x = (
            _get_kwargs(("pixel_scale_x", "pixscale_x", "pixelscale_x"), 0.059, kwargs)
            * _u.arcsec
            / _u.pixel
        )
        self.pscale_fact = _get_kwargs(
            ("pixel_scale_factor", "pscale_fact", "pscale_factor"), 3, kwargs
        )
        self.field_of_view = _get_kwargs(
            ("field_of_view", "fov_arcsecs"), 5 * _u.arcsec, kwargs
        )
        self.wavel_pfs = (
            _get_kwargs(("wavelength_pfs", "wavel_pfs", "wavelength"), 550e-9, kwargs)
            * _u.m
        )
        band = kwargs.get('band', 'Gaia_G')
        ccdpx = _get_kwargs(
            ("ccd_pixels", "tot_ccd_pixels", "ccdpxs", "ccd_pxs"), [4500*_u.pixel,1966*_u.pixel], kwargs
        )
        pixel_area = kwargs.get('pixel_area', 10*_u.um * 30*_u.um)
        t_int_per_ccd = _get_kwargs(
            ('integration_time', 'ccd_int_time', 'ccd_t_int'), 4.42 * _u.s, kwargs
        )
        # Setting the pixel scale in y direction
        self.pixscale_y = self.pixscale_x * self.pscale_fact
        # Creating the optical system
        optic = _poppy.RectangleAperture(width=self.apert_w, height=self.apert_h)
        osys.add_pupil(optic)
        # Creating the detector
        osys.add_detector(pixelscale=self.pixscale_x, fov_arcsec=self.field_of_view)
        self.ccd = _CCD(
            band=band,
            ccd_pixels=ccdpx,
            pixel_scale_x=self.pixscale_x,
            pixel_scale_y=self.pixscale_y,
            pixel_area=pixel_area,
            t_integration=t_int_per_ccd)
        self._osys = osys
        return self.__repr__()


    def __repr__(self):
        """String representation of the Gaia Telescope."""
        return f"""           Gaia Telescope V0
           -----------------
          Aperture: {self.apert_w} x {self.apert_h}
    Scansion speed: {self.speed:.2f}
    Orbital Period: {self.period.to(_u.hour)}
 Precession period: {self.precession_period.to(_u.day)}
Wavelength for PSF: {self.wavel_pfs}

{self.ccd.__repr__()}
"""
        


class _CCD:
    """
    Class to create a sky map with a star at the center and Poisson noise.

    Parameters
    ----------
    band : str, optional
        The band for which the sky map is created.
        Optionas are:
        - Gaia_G
        - Gaia_BP
        - Gaia_RP
        - U
        - B
        - V
        - Rc
        - Ic
        - J
        - H
        - Ks
        - W1 (bandwidth missing)
        - W2 (bandwidth missing)
        - W3 (bandwidth missing)
        - W4 (bandwidth missing)
        - I1 (bandwidth missing)
        - I2 (bandwidth missing)
        - I3 (bandwidth missing)
        - I4 (bandwidth missing)
        - M1 (bandwidth missing)
        - M2 (bandwidth missing)
        - M3 (bandwidth missing)
        - u_SDSS
        - g_SDSS
        - r_SDSS
        - i_SDSS
        - z_SDSS

        Default is 'Gaia_G'.
    """

    def __init__(self, band: str = "Gaia_G", **kwargs: dict[str,_any]):
        """The constructor"""
        self._bands = _qt.read("data/bands.fits")
        self._bands = self._bands[self._bands["band"] == band]
        self._passbands = _qt.read("data/gaiaDR3passband.fits")
        if len(self._bands) == 0:
            raise ValueError(f"Band '{band}' not found in the bands table.")
        self.pixel_area = kwargs.get("pixel_area", 10 * 30 * _u.um**2)
        self.ccd_pixels = kwargs.get("ccd_pixels", [4500*_u.pixel, 1966*_u.pixel] )
        self.pxscale_x = kwargs.get("pixel_scale_x", 0.059 * _u.arcsec / _u.pixel)
        self.pxscale_y = kwargs.get("pixel_scale_y", 0.059 *3* _u.arcsec / _u.pixel)
        self.fov = [self.ccd_pixels[0] * self.pxscale_x,
                    self.ccd_pixels[1] * self.pxscale_y]
        self.tdi = kwargs.get('t_integration', 4.42 * _u.s)
    
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
    """

    def display_passbands(self) -> None:
        """
        Display the passbands of the CCD.
        """
        _plt.figure(figsize=(8, 6))
        _plt.grid(linestyle='--', alpha=0.85)
        _plt.plot(self._passbands['lambda']/1000, self._passbands['GPb'] , c='green', label='G')
        _plt.plot(self._passbands['lambda']/1000, self._passbands['RPPb'], c='red'  , label='RP', linestyle='--')
        _plt.plot(self._passbands['lambda']/1000, self._passbands['BPPb'], c='blue' , label='BP', linestyle='--')
        _plt.xlabel('Wavelength [μm]')
        _plt.ylabel('Throughput')
        _plt.title('Gaia DR3 Passbands')
        _plt.yticks(_np.arange(0, 0.85, 0.1))
        _plt.legend()
        _plt.show()

    def create_binary_cube(
        self,
        M1: float | _u.Quantity,
        M2: float | _u.Quantity,
        distance: float | _u.Quantity,
        collecting_area: float | _u.Quantity,
        t_integration: float | _u.Quantity,
        shape: tuple[int, int] = (220, 220),
    ) -> _np.ndarray:
        """
        Create a cube of images of a binary star system, in which each image corresponds
        to a different angular position of one star with respect to the other.

        Parameters
        ----------
        M1 : float or u.Quantity
            Magnitude of the central star.
        M2 : float or u.Quantity
            Magnitude of the companion star.
        distance : float or u.Quantity
            Distance to the binary system in mas. Resolution is 1 mas.
        shape : tuple of int, optional
            Shape of the sky map (height, width). Default is (220,220).

        Returns
        -------
        numpy.ndarray
            The binary cube with the star at the center.
        """
        if any([s < distance * 2 for s in list(shape)]):
            raise ValueError(
                f"Map shape must be larger than twice the distance of the binary system: {shape} < {distance*2}"
            )
        center = (shape[0] // 2, shape[1] // 2)
        _map = _np.zeros(shape, dtype=float)
        star1 = self._compute_star_flux(
            M1, collecting_area=collecting_area, t=t_integration
        )
        companion = self._compute_star_flux(
            M2,
            collecting_area=collecting_area,
            t=t_integration,
        )
        ring = self._create_ring(radius=distance, shape=shape)
        _map[center] += star1
        pos_cube = []
        x, y = _np.where(ring == 1)
        for x, y in zip(x, y):
            mapp = _np.copy(_map)
            mapp[x, y] += companion
            # Add Poisson noise to the map
            noise = _np.random.poisson(_np.random.uniform(0.5, 5, 1), size=mapp.shape)
            mapp += noise
            pos_cube.append(mapp)
        pos_cube = _np.dstack(pos_cube)
        pos_cube = _np.rollaxis(pos_cube, -1)
        return pos_cube

    def _compute_star_flux(
        self,
        M: float | _u.Quantity,
        collecting_area: float | _u.Quantity,
        integration_time: float | _u.Quantity = 1 * _u.s,
    ) -> _u.Quantity:
        """
        Calculate photon flux for magnitude M star in V-band.

        Parameters
        ----------
        M : u.Quantity or float
            Magnitude of the star.
        collecting_area : u.Quantity or float
            Collecting area of the telescope in cm².
        integration_time : u.Quantity or float, optional
            Integration time in seconds. Default is 1 second.

        Returns
        -------
        u.Quantity
            Photon flux in photons per second per cm².
        """
        if not isinstance(M, _u.Quantity):
            M = M * _u.mag
        base_flux = self._band_flux()
        mag_values = M.value
        magnitude_factor = 10 ** (-0.4 * mag_values)
        photon_flux = base_flux * magnitude_factor
        tot_photons = photon_flux*collecting_area*integration_time # n_photons collected
        if tot_photons < 1:
            import warnings
            warnings.WarningMessage(
                "The number of photons collected is less than 1. "
                "This may lead to inaccurate results.",
                RuntimeWarning
            )
        return _np.minimum(1, tot_photons)

    def _band_flux(self) -> _u.Quantity:
        """
        Calculate photon flux for magnitude 0 star in V-band.

        Returns
        -------
        u.Quantity
            Photon flux in photons per second per cm².
        """
        from astropy.constants import h, c

        f_nu_0 = self._bands["zero_point"]  # Standard zero-point in Janskys
        lambda_eff = self._bands["wavelength"].to(_u.nm)
        delta_lambda = self._bands["bandwidth"].to(_u.nm)
        f_lambda = f_nu_0 * c / lambda_eff**2
        energy_flux = f_lambda * delta_lambda
        photon_energy = h * c / lambda_eff
        photon_flux = (energy_flux / photon_energy).to(1 / (_u.s * _u.um**2))
        return photon_flux

    def _create_ring(
        self, radius: int, shape: tuple[int, int], show: bool = False
    ) -> _np.ndarray:
        """
        Create a ring mask for the sky map.

        Parameters
        ----------
        radius : float
            The radius of the ring in pixels.
        shape : tuple of int
            The shape of the sky map (height, width).

        Returns
        -------
        ring_mask : numpy.ndarray
            A boolean mask where the ring is True.
        """
        from skimage.draw import disk

        outer_disk = disk((shape[0] // 2, shape[1] // 2), radius, shape=shape)
        inner_disk = disk((shape[0] // 2, shape[1] // 2), radius - 1, shape=shape)
        ring_mask = _np.zeros(shape, dtype=bool)
        ring_mask[outer_disk] = True
        ring_mask[inner_disk] = False  # remove inner disk, leaving only the ring
        extent = (-shape[1] // 2, shape[1] // 2, -shape[0] // 2, shape[0] // 2)
        if show:
            _plt.figure(figsize=(8, 7))
            _plt.imshow(ring_mask, origin="lower", extent=extent, cmap="inferno")
            _plt.xlabel("X [mas]")
            _plt.ylabel("Y [mas]")
            _plt.title(f"Ring mask with radius {radius} mas")
            _plt.show()
        return ring_mask


def _get_kwargs(names: tuple[str], default: _any, kwargs: dict[str, _any]) -> _any:
    """
    Gets a tuple of possible kwargs names for a variable and checks if it was
    passed, and in case returns it.

    Parameters
    ----------
    names : tuple
        Tuple containing all the possible names of a variable which can be passed
        as a **kwargs argument.
    default : any type
        The default value to assign the requested key if it doesn't exist.
    kwargs : dict
        The dictionary of variables passed as 'Other Parameters'.

    Returns
    -------
    key : value of the key
        The value of the searched key if it exists. If not, the default value will
        be returned.
    """
    possible_keys = names
    for key in possible_keys:
        if key in kwargs:
            return kwargs[key]
    return default

import numpy as _np
import poppy as _poppy
import astropy.units as _u
from astropy import convolution as _c
from typing import Any as _any
from astropy.table import QTable as _qt


class GaiaTelescopeV0():
    """    
    Class to simulate the Gaia telescope PSF.
    This class creates an optical system with the parameters of the Gaia
    telescope and computes the PSF for a given wavelength.
    
    Parameters
    ----------
    aperture_width : float, optional
        The width of the aperture in meters. Default is 1.5 m.
    aperture_height : float, optional
        The height of the aperture in meters. Default is 0.5 m.
    pixel_scale_x : float, optional
        The pixel scale in the x direction in arcseconds per pixel. Default is 0.05 arcsec/pixel.
    pixel_scale_factor : int, optional
        The factor to scale the pixel size in the y direction. Default is 2.
    field_of_view : float, optional
        The field of view in arcseconds. Default is 5 arcsec.
    wavelength_pfs : float, optional
        The wavelength for the PSF calculation in meters. Default is 550 nm.
    """
    
    def __init__(self, **kwargs: dict[str,_any]):
        """The constructor"""
        self._osys = self.__create_optical_system(kwargs)
        self.psf = self._compute_psf()
        self.speed = 59.9641857803 * _u.arcsec / _u.s
        self.period = (6 * _u.hour).to(_u.s)
        self.precession_period = (63 * _u.day).to(_u.s)


    def observeSkyMap(self, sky_map: _np.ndarray) -> _np.ndarray:
        """
        Simulate the observation of a sky map with the Gaia telescope PSF.

        Parameters
        ----------
        sky_map : numpy.ndarray
            The sky map to be observed, typically a 2D array representing the sky.

        Returns
        -------
        numpy.ndarray
            The observed image after convolving with the PSF.
        """
        if self.psf is None:
            raise ValueError("PSF has not been computed yet.")
        # resample the sky map to match the PSF pixel scale
        sky_map = _poppy.utils.rebin_array(sky_map, (self.pscale_fact, 1))
        # Convolve the sky map with the PSF
        convolved_image = _c.convolve_fft(sky_map, self.psf, boundary='wrap')
        # Compute the TDI integration
        n_steps = int(self.field_of_view.to(_u.arcsec).value / self.pixscale_x.to(_u.arcsec).value)
        px_per_step = int(self.pixscale_x.to(_u.arcsec).value / self.field_of_view.to(_u.arcsec).value * self.pscale_fact)
        for i in range(1,n_steps):
            convolved_image += _np.roll(convolved_image, shift=i*px_per_step, axis=1)
        return convolved_image
    

    def display_psf(self):
        """Display the PSF of the Gaia telescope."""
        if self.psf is None:
            raise ValueError("PSF has not been computed yet.")
        _poppy.display_psf(self.psf, title="Gaia Telescope PSF", 
                           cmap='gray', colorbar=True, 
                           show=False, scale=self.pixscale_x)

    def _compute_psf(self):
        """Compute the PSF of the Gaia telescope."""
        psf = self._osys.calc_psf(self.wavel_pfs)
        img = psf[0].data
        final_psf = _poppy.utils.rebin_array(img, (1,self.pscale_fact))
        return final_psf


    def __create_optical_system(self, **kwargs: dict[str,_any]):
        """Create the optical system for the Gaia telescope."""
        osys = _poppy.OpticalSystem()
        # Getting the optical parameters from kwargs
        # If not passed, the default values will be used.
        self.apert_w      = _get_kwargs(('aperture_width', 'width', 'aperture_w', 'apert_w'), 1.5, kwargs) * _u.m
        self.apert_h      = _get_kwargs(('aperture_height', 'height', 'aperture_h', 'apert_h'), 0.5, kwargs) * _u.m
        self.pixscale_x   = _get_kwargs(('pixel_scale_x', 'pixscale_x', 'pixelscale_x'), 0.05, kwargs) * _u.arcsec / _u.pixel
        self.pscale_fact  = _get_kwargs(('pixel_scale_factor', 'pscale_fact', 'pscale_factor'), 2, kwargs)
        self.field_ofview = _get_kwargs(('field_of_view', 'fov', 'field_ofview'), 5, kwargs) * _u.arcsec
        self.wavel_pfs    = _get_kwargs(('wavelength_pfs', 'wavel_pfs', 'wavelength'), 550e-9, kwargs) * _u.m
        # Setting the pixel scale in y direction
        self.pixscale_y = self.pixscale_x * self.pscale_fact
        # Creating the optical system
        optic = _poppy.RectangleAperture(width=self.apert_w, height=self.apert_h)
        osys.add_pupil(optic)
        # Creating the detector 
        osys.add_detector(pixelscale=self.pixscale_x, fov_arcsec=self.field_ofview)
        return osys


class SkyMap():
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
    
    def __init__(self, band: str = 'Gaia_G', map_noise: float = 5.0):
        """The constructor"""
        self._bands = _qt.read('data/bands.fits')
        self.map_noise = map_noise
    
    def create_single_star_sky_map(self, M: float, shape: tuple[int,int]) -> _np.ndarray:
        """
        Create a sky map with a star at the center and Poisson noise.

        Parameters
        ----------
        flux : float
            Total counts for the star.
        background : float
            Mean background value.
        shape : tuple of int
            Shape of the sky map (height, width).

        Returns
        -------
        numpy.ndarray
            The generated sky map.
        """
        flux = self._compute_star_flux(M)
        sky_map = _np.random.poisson(self.map_noise, shape)
        center_y, center_x = (shape[0] // 2, shape[1] // 2)
        sky_map[center_y, center_x] += flux.value
        return sky_map
    
    def _band_flux(self) -> _u.Quantity:
        """
        Calculate photon flux for magnitude 0 star in V-band.
        
        Returns
        -------
        u.Quantity
            Photon flux in photons per second per cm².
        """
        from astropy.constants import h, c
        f_nu_0 = self._bands['zero_point'] # Standard zero-point in Janskys
        lambda_eff = self._bands['wavelength'].to(_u.nm)
        delta_lambda = self._bands['bandwidth'].to(_u.nm) 
        f_lambda = f_nu_0 * c / lambda_eff**2
        energy_flux = f_lambda * delta_lambda
        photon_energy = h * c / lambda_eff
        photon_flux = (energy_flux / photon_energy).to(1/(_u.s * _u.cm**2))
        return photon_flux

    def _compute_star_flux(self, M: float) -> _u.Quantity:
        """
        Calculate photon flux for magnitude M star in V-band.
        
        Parameters
        ----------
        M : u.Quantity or float
            Magnitude of the star.
        efficiency : float, optional
            Photon collection efficiency.
            
        Returns
        -------
        u.Quantity
            Photon flux in photons per second per cm².
        """
        if isinstance(M, (int, float)):
            M = M * _u.mag
        base_flux = self._band_flux()
        if hasattr(M, 'unit'):
            mag_values = M.value
        else:
            mag_values = M
        magnitude_factor = 10**(-0.4 * mag_values)
        photon_flux = base_flux * magnitude_factor
        return photon_flux  # in photons per second per cm^2



def _get_kwargs(names: tuple[str], default: _any, kwargs: dict[str,_any]) -> _any:
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

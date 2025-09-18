import xupy as xp
_np = xp.np
from xupy import typings as _xt
import poppy as _poppy
import astropy.units as _u
from astropy import convolution as _c
from utils import load_fits, save_fits, _Any, _fits, Logger
from astropy.table import QTable as _qt
import matplotlib.pyplot as _plt
import multiprocessing as _mp
import gc
_l = Logger()

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

    def __init__(self, **kwargs: dict[str,_Any]):
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
        **kwargs: dict[str, _Any],
    ):
        """The constructor"""
        if not psf is None:
            if isinstance(psf, _fits.HDUList):
                self._psf = psf
            elif isinstance(psf, str):
                self._psf = _fits.open(psf)
            else:
                raise TypeError(
                    "`psf` must be an astropy.io.fits.HDUList or a string path to a FITS file."
                )
            self.psf = self._psf[0].data
            self._computeXandYpsf()
        else:
            self._psf = self.psf = None
            self.psf_x = None
            self.psf_y = None
            print(
                """PSF not provided. Use the method 'compute_psf' to compute it, passing a
`poppy` optical system."""
            )
        self._bands = _qt.read("data/bands.fits")
        self._bands = self._bands[self._bands["band"] == kwargs.get("band", "Gaia_G")]
        self._passbands = _qt.read("data/gaiaDR3passband.fits")
        self.pixel_area = kwargs.get("pixel_area", 10 * 30 * _u.um**2)
        self.ccd_pixels = kwargs.get("ccd_pixels", [4500 * _u.pixel, 1966 * _u.pixel])
        self.pxscale_x = kwargs.get("pixel_scale_x", 0.059 * _u.arcsec / _u.pixel)
        self.pxscale_y = kwargs.get("pixel_scale_y", 0.059 * 3 * _u.arcsec / _u.pixel)
        self.fov = [
            self.ccd_pixels[0] * self.pxscale_x,
            self.ccd_pixels[1] * self.pxscale_y,
        ]
        self.tdi = kwargs.get("t_integration", 4.42 * _u.s)
        self.px_ratio = self.pxscale_y / self.pxscale_x
        self.rebinned = False

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
        px_ratio = (rebin_factor * axis_ratio[0], rebin_factor * axis_ratio[1])
        if psf is None:
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
            self._computeXandYpsf()
            self.rebinned = True
            return "Rebinning complete."
        else:
            psf_out = _poppy.utils.rebin_array(psf, px_ratio)
            psfx_out, psfy_out = self._computeXandYpsf(psf=psf_out)
            return psf_out, psfx_out, psfy_out

    def compute_psf(self, osys: _poppy.OpticalSystem, overwrite: bool = False) -> None:
        """
        Compute the PSF of the Gaia telescope using a given optical system.

        Parameters
        ----------
        osys : poppy.OpticalSystem
            The optical system to use for the PSF computation.
        overwrite : bool, optional
            Whether to overwrite the existing PSF if it exists. Default is False.
        """
        if not isinstance(osys, _poppy.OpticalSystem):
            raise TypeError("osys must be an instance of poppy.OpticalSystem")
        if self._psf is not None:
            print("PSF already computed.")
            if not overwrite:
                print("Use overwrite=True to recompute the PSF.")
                return
        print("Overwriting existing PSF.")
        wvls = self._passbands["lambda"]
        weights = self._passbands["G"].filled(0)
        self._psf = osys.calc_psf(
            progressbar=True, source={"wavelengths": wvls, "weights": weights}
        )[0].data
        self.psf = self._psf[0].data
        self._computeXandYpsf()
        return "Computation complete."

    def display_psf(self, mode: str = "2d", **kwargs: dict[str, _Any]) -> None:
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
        if self._psf is None:
            raise ValueError("PSF has not been computed yet.")
        if mode == "2d":
            if not self.rebinned:
                _poppy.display_psf(self._psf, title="CCD PSF", **kwargs)
            else:
                _plt.figure()
                _plt.imshow(self.psf, **kwargs)
                _plt.colorbar()
                _plt.title("CCD PSF")
                _plt.xlabel("X [px]")
                _plt.ylabel("Y [px]")
                _plt.show()
            return
        else:
            _plt.figure()
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
            return

    def _computeXandYpsf(
        self, psf: _xt.Optional[_xt.ArrayLike] = None
    ) -> None | tuple[_xt.ArrayLike, _xt.ArrayLike]:
        """
        Subroutine to compute the normalized psf in the X and Y axis of the
        2D PSF.
        """
        if psf is None:
            img = self.psf.copy()
        else:
            img = psf.copy()
        psf_x = _np.sum(img, axis=0)
        psf_x /= _np.sum(psf_x)  # normalize
        psf_y = _np.sum(img, axis=1)
        psf_y /= _np.sum(psf_y)
        self.psf_x = psf_x
        self.psf_y = psf_y
        if psf is not None:
            return psf_x, psf_y
        return

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


class BinarySystem:
    """
    Class to simulate a binary star system.
    
    Parameters
    ----------
    M1 : float
        Magnitude of the primary star.
    M2 : float
        Magnitude of the secondary star.
    distance : float or astropy.units.Quantity
        Angular separation between the two stars in milliarcseconds (mas).
    shape : tuple of int, optional
        Shape of the sky map (height, width). Default is (250, 250).
    band : str, optional
        The band for which the CCD is initialized. Options are:
        - Gaia_G
        - Gaia_BP
        - Gaia_RP
        Default is 'Gaia_G'.
        
    Attributes
    ----------
    map : numpy.ndarray
        The cube of images of the binary star system.
    is_observed : bool
        Indicates whether the system has been observed with a CCD.
        
    Methods
    -------
    create_raw_binary_cube(collecting_area, t_integration, shape=(512, 512), out=False)
        Create a cube of images of the binary star system.
    observe(ccd)
        Observe the sky map with the given CCD.
    show_system(**kwargs)
        Plots the fits image of the binary system's cube.
    writeto(filepath)
        Write the cube of images to a FITS file.    
    
    """
    
    def __init__(
        self, M1: float, M2: float, distance: float, **kwargs: dict[str, _Any]
    ):
        """The constructor"""
        self.M1 = M1
        self.M2 = M2
        if not isinstance(distance, _u.Quantity):
            self.distance = distance * _u.mas  # assuming input is in mas
        else:
            if distance.unit != _u.mas:
                self.distance = distance.to(_u.mas)
        self.distance = int(self.distance.value)  # mas
        self.map_shape = kwargs.get("shape", (250, 250))
        self.is_observed = False
        self._bands = _qt.read("data/bands.fits")
        self._bands = self._bands[self._bands["band"] == kwargs.get("band", "Gaia_G")]

    @property
    def map(self) -> _np.ndarray[float, _Any]:
        """
        Returns the cube of images of the binary star system.
        """
        if not hasattr(self, "_cube"):
            raise AttributeError(
                "Cube has not been created yet. Use 'create_raw_binary_cube' method."
            )
        return self._cube

    def observe(self, ccd: CCD):
        """
        Observe the sky map with the given CCD:
        
        Given the CCD (with its psf):
        - convolves each image in the cube with the CCD's PSF
        - applies the photon noise afterwards
        - repeats the process shifting the image by an increasing amount of pixels in x and y
        - creates a new cube for configuration (angle) and saves it
        
        Parameters
        ----------
        ccd : CCD
            The CCD to use for the observation.
        """
        check = self._check_everything_is_fine(ccd)
        convolved_cube = []
        _l.log("Starting convolution computation on cube...", level="INFO")
        from tqdm import tqdm

        for img in tqdm(self._cube, desc="Observing", unit="config"):
            if check == 'pad':
                xdiff = ccd.psf.shape[1] - img.shape[1]
                ydiff = ccd.psf.shape[0] - img.shape[0]
                img = _np.pad(img, ((ydiff//2, ydiff//2), (xdiff//2, xdiff//2)), mode='constant')
            convolved = _c.convolve_fft(img, ccd.psf, boundary='wrap', normalize_kernel=False, allow_huge=True)
            # Add Poisson noise to the convolved image
            noisy = _np.random.poisson(convolved).astype(_np.float32)
            del convolved
            gc.collect()
            convolved_cube.append(ccd.rebin_psf(noisy, rebin_factor=59, axis_ratio=(1, 3))[0])
            del noisy
            gc.collect()

        _l.log("Convolution complete.", level="INFO")
        outcube = _np.dstack(convolved_cube)
        outcube = _np.rollaxis(outcube, -1)
        self.is_observed = True
        return outcube

    def show_system(self, **kwargs: dict[str, _Any]) -> None:
        """
        Plots the fits image of the binary system's cube (A random image of the cube).
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to the imshow function.
        """
        n = _np.random.randint(0, self._cube.shape[0]-1)
        _plt.figure()
        _plt.imshow(self._cube[n], **kwargs)
        _plt.colorbar()
        _plt.title(f"Binary sample: {self.M1} mag + {self.M2} mag, {self.distance} mas")
        _plt.xlabel("X [mas]")
        _plt.ylabel("Y [mas]")
        _plt.show()

    def writeto(self, filepath: str) -> None:
        """
        Write the cube of images to a FITS file.

        Parameters
        ----------
        filepath : str
            The path to the output FITS file.
        """
        if not hasattr(self, "_cube"):
            raise AttributeError(
                "Cube has not been created yet. Use 'create_raw_binary_cube' method."
            )
        hdu = _fits.PrimaryHDU(self._cube)
        hdu.writeto(filepath, overwrite=True)
        print(f"Cube written to {filepath}")

    def create_raw_binary_cube(
        self,
        collecting_area: float | _u.Quantity,
        t_integration: float | _u.Quantity,
        shape: tuple[int, int] = (512, 512),
        out: bool = False,
    ) -> _np.ndarray[float, _Any]:
        """
        Create a cube of images of a binary star system, in which each image corresponds
        to a different angular position of one star with respect to the other.

        Parameters
        ----------
        collecting_area : float or u.Quantity
            Collecting area of the telescope in m².
        t_integration : float or u.Quantity
            Integration time in seconds. Default is 1 second.
        shape : tuple of int, optional
            Shape of the sky map (height, width). Default is (220,220).
        out : bool, optional
            If True, saves the cube as a FITS file. Default is False.

        Returns
        -------
        numpy.ndarray
            The binary cube with the star at the center.
        """
        if any([s < self.distance for s in list(shape)]):
            raise ValueError(
                f"Map shape must be larger than twice the distance of the binary system: {shape} < {self.distance*2}"
            )
        center = (shape[0] // 2, shape[1] // 2)
        _map = _np.zeros(shape, dtype=_np.float32)
        star1 = self._compute_star_flux(
            self.M1, collecting_area=collecting_area, integration_time=t_integration
        )
        companion = self._compute_star_flux(
            self.M2,
            collecting_area=collecting_area,
            integration_time=t_integration,
        )
        ring = self._create_ring(radius=self.distance, shape=shape)
        _map[center] += star1
        pos_cube = []
        x, y = _np.where(ring == 1)
        for x, y in zip(x, y):
            mapp = _np.copy(_map)
            mapp[x, y] += companion
            # Add Poisson noise to the map
            # noise = _np.random.poisson(_np.random.uniform(0.5, 5, 1), size=mapp.shape)
            # mapp += noise
            pos_cube.append(mapp)
        pos_cube = _np.dstack(pos_cube)
        pos_cube = _np.rollaxis(pos_cube, -1)
        self._cube = pos_cube
        if out:
            return pos_cube
        
    def _check_everything_is_fine(self, ccd: CCD) -> None:
        """
        Check if everything is fine before observing the binary system with the CCD.
        """
        if not hasattr(self, "_cube"):
            _l.log("Cube has not been created yet. Use 'create_raw_binary_cube' method.", level="ERROR")
            raise AttributeError(
                "Cube has not been created yet. Use 'create_raw_binary_cube' method."
            )
        if ccd.psf is None:
            _l.log("CCD PSF has not been computed yet.", level="ERROR")
            raise ValueError("CCD PSF has not been computed yet.")
        if not self._cube.shape[1:] == ccd.psf.shape:
            _l.log("Cube and CCD PSF shapes do not match.", level="WARNING")
            _l.log("Padding to match PSF shape...", level="WARNING")
            return 'pad'
        return None

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
        if not isinstance(collecting_area, _u.Quantity):
            collecting_area = collecting_area * _u.m**2  # Assuming input is in m²
        collecting_area = collecting_area.to(_u.cm**2)
        base_flux = self._band_flux()
        mag_values = M.value
        magnitude_factor = 10 ** (-0.4 * mag_values)
        photon_flux = base_flux * magnitude_factor
        tot_photons = (
            photon_flux * collecting_area * integration_time
        )  # n_photons collected
        if tot_photons.value < 1:
            import warnings

            warnings.warn(
                message="The number of photons collected is less than 1. This may lead to inaccurate results.",
                category=RuntimeWarning,
            )
        return _np.maximum(1, tot_photons.value)

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
        photon_flux = (energy_flux / photon_energy).to(1 / (_u.s * _u.cm**2))
        return photon_flux

    def _create_ring(
        self, radius: int, shape: tuple[int, int], show: bool = False
    ) -> _np.ndarray[int, _Any]:
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
    
    def __repr__(self):
        """String representation of the Binary System."""
        return f"""         Binary System
         -------------
       Star 1 Mag: {self.M1} mag
       Star 2 Mag: {self.M2} mag
    Separation: {self.distance} mas
      Map shape: {self.map_shape} pixels
       Band: {self._bands['band'][0]}
       Cube shape: {self._cube.shape if hasattr(self, '_cube') else 'Not created yet'}
       Observed: {'Yes' if self.is_observed else 'No'}
"""


def _get_kwargs(names: tuple[str], default: _Any, kwargs: dict[str, _Any]) -> _Any:
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


class _Convolver():
    
    def __init__(self, ccd: CCD = None):
        self.psf = ccd.psf
        self.ccd = ccd

    def convolve(self, img):
        """
        Convolve the input image with the PSF.
    
        Parameters
        ----------
        img : numpy.ndarray
            The input image to convolve.

        Returns
        -------
        numpy.ndarray
            The convolved image.
        """
        xdiff = self.psf.shape[1] - img.shape[1]
        ydiff = self.psf.shape[0] - img.shape[0]
        img = _np.pad(img, ((ydiff//2, ydiff//2), (xdiff//2, xdiff//2)), mode='constant')
        convolved = _c.convolve_fft(img, self.psf, boundary='wrap', normalize_kernel=False, allow_huge=True)
        # Add Poisson noise to the convolved image
        noisy = _np.random.poisson(convolved).astype(_np.float32)
        del convolved
        gc.collect()
        fin_psf = ccd.rebin_psf(noisy, rebin_factor=59, axis_ratio=(1, 3))[0]
        del noisy
        gc.collect()
        return fin_psf
import os, gc, xupy as xp
_np = xp.np
from instruments import CCD
import matplotlib.pyplot as _plt
from xupy import typings as _xt
import astropy.units as _u
from utils import *
from utils import _header_from_dict as hfd
import processing as _p

_l = Logger()
basepath = os.path.dirname(os.path.abspath(__file__))

class BinarySystem:
    """
    Class to simulate a binary star system.

    Parameters
    ----------
    ccd : CCD
        The CCD to use for the observation.
    M1 : float
        Magnitude of the primary star.
    M2 : float
        Magnitude of the secondary star.
    distance : float or astropy.units.Quantity
        Angular separation between the two stars in milliarcseconds (mas).

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
        self,
        *,
        ccd: CCD,
        M1: float,
        M2: float,
        distance: float,
        **kwargs: dict[str, _xt.Any],
    ):
        """The constructor"""
        self.ccd = ccd
        self.map_shape = self.ccd.psf.shape
        if not isinstance(distance, _u.Quantity):
            self.distance = distance * _u.mas  # assuming input is in mas
        else:
            if distance.unit != _u.mas:
                self.distance = distance.to(_u.mas)
        self.distance = int(self.distance.value)  # mas
        if any([s < 2 * self.distance for s in list(self.map_shape)]):
            raise ValueError(
                f"Map shape must be larger than twice the distance of the binary system: {self.map_shape} < {2*self.distance*2}"
            )
        self.M2 = M2
        self.M1 = M1
        self._Mtot = -2.5 * _np.log10(10**(-0.4 * M1) + 10**(-0.4 * M2))
        self.is_observed = False
        self._bands = self.ccd._bands
        G = self._Mtot
        if G < 13:
            idx = 0
        elif 13 <= G < 16:
            idx = 1
        else:
            idx = 2
        if not eval(self.ccd._wc_conditions[idx].format(G=G)):
            raise ValueError(
            f"Something's wrong..."
            )
        del G
        self.collecting_area = self.ccd.WC[idx]['area_um']
        self.t_int = self.ccd.tdi
        self.comp_star_flux = self._compute_star_flux(
            self.M2, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self.central_star_flux = self._compute_star_flux(
            self.M1, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self._base_map = self._create_base_map()
 

    def observe(self, ccd: CCD, map_dtype: str = "float32") -> str:
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
        tn = newtn()
        N = self._create_ring(radius=self.distance, shape=self.map_shape).sum()
        datapath = os.path.join(
            basepath, "data", "simulations", "observations", f"{tn}"
        )
        if not os.path.exists(datapath):
            os.makedirs(datapath, exist_ok=True)
        _l.log("Starting convolution computation on binary system...", level="INFO")
        _l.log(f"Data Tracking Number : {tn}", level="INFO")
        from tqdm import tqdm

        i = 0
        header = self._prepare_main_header()
        header['PIXELSCL'] = (self.ccd.header['PIXELSCL'], 'Pixel scale [mas/pixel]')
        imgHeader = self._prepare_main_header()
        imgHeader['PXSCLREB'] = (self.ccd.pxscale_factor, 'Pixel scale rebin ratio (y/x)')
        imgHeader['PXSCLXRB'] = (self.ccd.pxscale_x.value, 'Rebinned pixel scale in x [mas/pixel]')
        imgHeader['PXSCLYRB'] = (self.ccd.pxscale_y.value, 'Rebinned pixel scale in y [mas/pixel]')
        assert ccd.psf.shape == self._base_map.shape, "PSF and map shapes do not match."
        h2 = hfd(header)
        h1 = hfd(imgHeader)
        mdtype = xp.float if map_dtype == "float32" else xp.double
        
        for img in tqdm(
            self.transit(), desc=f"[{tn}] Observing...", unit="images", total=N
        ):
            y, x = _np.where(img != 0)
            if len(y) != 2 or len(x) != 2:
                raise ValueError("Something's wrong with the binary map")
            phi = _np.arctan(_np.abs(y[1]-y[0])/_np.abs(x[1]-x[0]))*_u.rad
            h1['PHI'] = (phi.to_value(_u.deg), 'Position angle of companion star [deg]')
            h2['PHI'] = (phi.to_value(_u.deg), 'Position angle of companion star [deg]')
            convolved = _p.convolve_fft(
                img, ccd.psf, dtype=mdtype, boundary="wrap", normalize_kernel=True)
            
            # Add Poisson noise and Readout Noise to the convolved image
            # noisy = _np.random.poisson(convolved).astype(_np.float32)
            # noisy += _np.random.normal(0, 5, size=noisy.shape)  # readout noise
            
            final = ccd.sample_psf(psf=convolved)
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU(final[0], header=h1))
            hdul.append(fits.ImageHDU(final[1], name="PSF_X"))
            hdul.append(fits.ImageHDU(final[2], name="PSF_Y"))
            hdul.append(fits.ImageHDU(convolved, name="HighRes obs", header=h2))
            hdul.writeto(os.path.join(datapath, f"{i:04d}.fits"), overwrite=True)
            del convolved, final
            gc.collect()
            i += 1
        
        # Computing reference PSF for fitting purposes
        base_source = self._base_map.copy()
        base_source[_np.where(base_source != 0)] = self._Mtot
        convolved = _p.convolve_fft(
            base_source, ccd.psf, dtype=mdtype, boundary="wrap", normalize_kernel=True)
        final = ccd.sample_psf(psf=convolved)
        hdul = fits.HDUList()
        for k in ["M1", "M2", "DISTMAS", 'PHI']:
            h1.pop(k)
            h2.pop(k)
        h1['GMAG'] = (self._Mtot, 'Calibration G-Magnitude of the expected source')
        h2['GMAG'] = (self._Mtot, 'Calibration G-Magnitude of the expected source')
        hdul.append(fits.PrimaryHDU(final[0], header=h1))
        hdul.append(fits.ImageHDU(final[1], name="PSF_X"))
        hdul.append(fits.ImageHDU(final[2], name="PSF_Y"))
        hdul.append(fits.ImageHDU(convolved, name="HighRes Calib", header=h2))
        hdul.writeto(os.path.join(datapath, f"calibration.fits"), overwrite=True)
        del convolved, final
        gc.collect()
        _l.log("Convolution complete.", level="INFO")
        self.is_observed = True
        return tn

    def show_system(self, out: bool = False, **kwargs: dict[str, _xt.Any]) -> None:
        """
        Plots the fits image of the binary system's cube (A random image of the cube).

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to the imshow function.
        """
        c = 0
        for i in self.transit():
            map = i.copy()
            c += 1
            if c == 1:
                break
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        _plt.figure()
        _plt.imshow(map, **kwargs)
        _plt.colorbar()
        _plt.title(f"Binary sample: {self.M1} mag + {self.M2} mag, {self.distance} mas")
        _plt.xlabel("X [mas]")
        _plt.ylabel("Y [mas]")
        if xlim is not None:
            _plt.xlim(xlim)
        if ylim is not None:
            _plt.ylim(ylim)
        if out:
            return map
        _plt.show()

    def transit(self):
        """
        Create a generator of images of a binary star system, in which each image corresponds
        to a different angular position of one star with respect to the other.

        Yields
        -------
        numpy.ndarray
            Each binary map with the star at the center for a different position.
        """
        for coord in self.coordinates():
            mmap = self._base_map.copy()
            mmap[coord[0], coord[1]] += self.comp_star_flux
            yield mmap

    def coordinates(self):
        """
        Get the coordinates of the secondary star in the binary system.

        Yields
        -------
        tuple of int
            (x, y) coordinates of the secondary star in pixels, one at a time.
        """
        ring = self._create_ring(radius=self.distance, shape=self.map_shape)
        x, y = _np.where(ring == 1)
        for i, j in zip(x, y):
            yield (i, j)

    def _create_base_map(self):
        """
        Create the base map with the primary star at the center and the secondary star
        at a distance of `self.distance` in a random direction.
        """
        center = (self.map_shape[0] // 2, self.map_shape[1] // 2)
        _map = _np.zeros(self.map_shape, dtype=_np.float64)
        _map[center] += self.central_star_flux
        return _map


    def _prepare_main_header(self):
        """
        Prepare the main header for the FITS file.
        """
        header = {}
        header["DISTMAS"] = (self.distance, "Angular separation in mas")
        header["M1"] = (self.M1, "Magnitude of the primary (central) star")
        header["M2"] = (self.M2, "Magnitude of the secondary (companion) star")
        header["BAND"] = (self._bands["band"][0], "Photometric band")
        header["WAVELEN"] = (self._bands["wavelength"][0].to_value(_u.nm), "Effective wavelength in nm")
        header["ZP"] = (self._bands["zero_point"][0].value, "Zero point in Jy")
        header["BANDWID"] = (self._bands["bandwidth"][0].to_value(_u.nm), "Bandwidth in nm")
        return header


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
    ) -> _np.ndarray[int, _xt.Any]:
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


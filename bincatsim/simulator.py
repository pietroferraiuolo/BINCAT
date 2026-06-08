import os
import gc
import xupy as xp
import numpy as _np
from .utils import *
import astropy.units as _u
from astropy.io import fits
from .instruments import CCD, Band
import matplotlib.pyplot as _plt
from xupy import typings as _xt
from .utils.logging import SystemLogger as _SL
from .core.root import OBS_DATA_PATH, SIMPATH
from opticalib.ground.osutils import _header_from_dict as hfd
import synphot as _sp

_SPECTRAL_TYPE_MAPPING = {
    "O": 30000 * _u.K,
    "B": 10000 * _u.K,
    "A": 7500 * _u.K,
    "F": 6000 * _u.K,
    "G": 5200 * _u.K,
    "K": 3700 * _u.K,
    "M": 2400 * _u.K
}

class GaiaSimulator:
    """
    Class to simulate a binary star system.

    Parameters
    ----------
    ccd : CCD
        The CCD to use for the observation.
    central_star : Star
        The primary star of the binary system.
    companion_star : Star
        The secondary star of the binary system.
    distance : float or astropy.units.Quantity
        Angular separation between the two stars in milliarcseconds (mas).
    angle : float or astropy.units.Quantity, optional
        Total angle to cover in the simulation in degrees. Default is 90 degrees.

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
        central_star: "Star",
        companion_star: "Star",
        distance: float | _u.Quantity,
        angle: float | _u.Quantity = 90.0,
    ):
        """The constructor"""
        self._logger = _SL(__class__)

        self.ccd = ccd

        self.map_shape = self.ccd.psf.shape
        self._logger.info(f"Binary System map shape set to {self.map_shape} pixels.")

        self.angle = (
            angle * _u.deg if not isinstance(angle, _u.Quantity) else angle.to(_u.deg)
        )
        self._logger.info(f"Configurations spanning 0-{self.angle} degrees.")

        self.distance = int(
            (distance * _u.mas if not isinstance(distance, _u.Quantity) else distance.to(_u.mas)).value
        )  # mas
        self._logger.info(f"Angular separation set to {self.distance} mas.")

        if any([s < 2 * self.distance for s in list(self.map_shape)]):
            raise ValueError(
                f"Map shape must be larger than twice the distance of the binary system: {self.map_shape} < {2*self.distance*2}"
            )

        self.central_star = central_star
        self.companion_star = companion_star
        self.M1 = self.central_star.magnitude
        self.M2 = self.companion_star.magnitude

        self._Mtot = -2.5 * _np.log10(10 ** (-0.4 * self.M1) + 10 ** (-0.4 * self.M2))
        self._logger.info(
            f"Calibration G-Magnitude of the single source: {self._Mtot} mag."
        )

        self.is_observed = False
        self._bands = self.ccd._bands
        self.ccd.set_exposure_time(self._Mtot)
        if self.ccd._saturated:
            print("WARNING: Source is brighter than the saturation threshold.")

        self.collecting_area = self.ccd.integration_area
        self.integration_time = self.ccd.integration_time

        self.central_star_flux = self._compute_star_flux(
            self.central_star
        )
        self._logger.info(f"Central star flux: {self.central_star_flux} photons/s/cm².")

        self.comp_star_flux = self._compute_star_flux(
            self.companion_star
        )
        self._logger.info(f"Companion star flux: {self.comp_star_flux} photons/s/cm².")

        self._base_map = self._create_base_map()
        self._noisegen = _np.random.Generator(_np.random.PCG64())


    def observe(
        self,
        ccd: CCD | None = None,
        read_out_noise: bool = False,
        shot_noise: bool = False,
        map_dtype: str = "float32"
    ) -> str:
        """
        Observe the sky map with the given CCD:

        Given the CCD (with its psf):
        - convolves each image in the cube with the CCD's PSF
        - applies the photon noise afterwards
        - repeats the process shifting the image by an increasing amount of pixels in x and y
        - creates a new cube for configuration (angle) and saves it

        Parameters
        ----------
        ccd : CCD, optional
            The CCD to use for the observation. If not provided, the default CCD will be used.
        read_out_noise : bool, optional
            Whether to add read-out noise to the images. Default is False.
        shot_noise : bool, optional
            Whether to add photon shot noise to the images. Default is False.
        map_dtype : str, optional
            Data type for the convolved maps. Default is "float32". Can be "float
            32" or "float64".
            
        Returns
        -------
        tn : str
            The tracking number of the observation, which corresponds to the folder name where the data is saved
        """
        if ccd is None:
            ccd = self.ccd

        N = self._create_ring(radius=self.distance).sum()
        datapath = create_data_folder(OBS_DATA_PATH)
        tn = datapath.split("/")[-1]
        self._logger.info("Starting binary star observations...")
        self._logger.info(f"Data Tracking Number : {tn}")

        from tqdm import tqdm

        header = self._prepare_main_header()
        header["PIXELSCL"] = (self.ccd.header["PIXELSCL"], "Pixel scale [mas/pixel]")
        imgHeader = self._prepare_main_header()
        imgHeader["PXSCLREB"] = (self.ccd.ccd_pxscale_factor, "Pxscale rebin ratio (y/x)")
        imgHeader["PXSCLXRB"] = (
            self.ccd.ccd_pxscale_x.value,
            "Rebinned pxscale_x [mas/pixel]",
        )
        imgHeader["PXSCLYRB"] = (
            self.ccd.ccd_pxscale_y.value,
            "Rebinned pxscale_y [mas/pixel]",
        )

        if not ccd.psf.shape == self._base_map.shape:
            self._logger.error("CCD PSF shape does not match binary map shape.")
            raise ValueError(
                f"CCD PSF shape {ccd.psf.shape} does not match binary map shape {self._base_map.shape}."
            )

        h2 = hfd(header)
        h1 = hfd(imgHeader)
        mdtype = xp.float if map_dtype == "float32" else xp.double

        i = 0
        for img in tqdm(
            self.transit(), desc=f"[{tn}] Observing...", unit="images", total=N
        ):
            phi = self.compute_scan_angle(img)
            h1["PHI"] = (phi, "Position angle of companion star [deg]")
            h2["PHI"] = (phi, "Position angle of companion star [deg]")

            # ---- Convolution ---- #
            convolved = convolve_fft(
                img, ccd.psf, dtype=mdtype, boundary="wrap", normalize_kernel=True
            )
            self._logger.info(
                f"Image {i:05d}: Convolution complete at angle {phi:.2f} degrees."
            )

            # ---- Shot Noise ---- #
            if shot_noise:
                noisy = _np.random.poisson(convolved).astype(_np.float32)
                convolved = noisy
                self._logger.info(f"Image {i:05d}: photon shot noise added.")
            # ------------------- #

            psf_2d, _, _ = ccd.sample_psf(psf=convolved, G=self._Mtot)
            self._logger.info(f"Image {i:05d}: PSF read-out complete (binning).")
            h1['WINDOW'] = (ccd._window, "CCD window used for the observation")
            h2['WINDOW'] = (ccd._window, "CCD window used for the observation")

            self._logger.debug(
                f"Image {i:05d}: shifting map based on sources distance."
            )
            if self.distance > 450:
                if phi < 30.0 or phi >= 330.0:
                    psf_2d = _np.roll(psf_2d, (-psf_2d.shape[1] // 4, 0), (1, 0))
                elif 30.0 <= phi < 60.0 or 300.0 <= phi < 330.0:
                    psf_2d = _np.roll(psf_2d, (-psf_2d.shape[1] // 6, 0), (1, 0))
                elif 60.0 <= phi < 90.0 or 270.0 <= phi < 300.0:
                    psf_2d = _np.roll(psf_2d, (-psf_2d.shape[1] // 8, 0), (1, 0))
                elif 90.0 <= phi < 120.0 or 240.0 <= phi < 270.0:
                    psf_2d = _np.roll(psf_2d, (psf_2d.shape[1] // 4, 0), (0, 1))
                elif 120.0 <= phi < 150.0 or 210.0 <= phi < 240.0:
                    psf_2d = _np.roll(psf_2d, (psf_2d.shape[1] // 6, 0), (0, 1))
                elif 150.0 <= phi < 210.0:
                    psf_2d = _np.roll(psf_2d, (psf_2d.shape[1] // 8, 0), (0, 1))

            # ---- Read-Out Noise ---- #
            if read_out_noise:
                ron = (
                    _np.random.normal(0, _np.random.randint(2, 6), size=psf_2d.shape) * 0.5
                )
                ron[ron < 0] = 0
                psf_2d += ron
                self._logger.info(f"Image {i:05d}: read-out noise added.")
            # ------------------------ #

            psf_x, psf_y = computeXandYpsf(psf=psf_2d, window=self.ccd._window)
            self._logger.info(f"Image {i:05d}: PSF X and Y computed.")
            final = (psf_2d, psf_x, psf_y)

            with fits.HDUList() as hdul:
                hdul.append(fits.PrimaryHDU(final[0], header=h1))
                hdul.append(fits.ImageHDU(final[1], name="PSF_X"))
                hdul.append(fits.ImageHDU(final[2], name="PSF_Y"))
                hdul.append(fits.ImageHDU(convolved, name="HighRes obs", header=h2))
                hdul.writeto(os.path.join(datapath, f"{i:05d}.fits"), overwrite=True)
            self._logger.info(f"Image {i:05d}: FITS file saved.")
            del convolved, final
            gc.collect()

            i += 1

        # Computing reference PSF for fitting purposes
        base_source = self._base_map.copy()
        base_source[_np.where(base_source != 0)] = self._Mtot
        self._logger.info("Computing reference PSF for fitting purposes.")
        convolved = convolve_fft(
            base_source, ccd.psf, dtype=mdtype, boundary="wrap", normalize_kernel=True
        )
        final = ccd.sample_psf(psf=convolved, G=self._Mtot)

        for k in ["M1", "M2", "DISTMAS", "PHI"]:
            h1.pop(k)
            h2.pop(k)
        h1["GMAG"] = (self._Mtot, "Calibration G-Magnitude of the expected source")
        h2["GMAG"] = (self._Mtot, "Calibration G-Magnitude of the expected source")

        with fits.HDUList() as hdul:
            hdul.append(fits.PrimaryHDU(final[0], header=h1))
            hdul.append(fits.ImageHDU(final[1], name="PSF_AL"))
            hdul.append(fits.ImageHDU(final[2], name="PSF_AC"))
            hdul.append(fits.ImageHDU(convolved, name="HighRes Calib", header=h2))
            hdul.writeto(os.path.join(datapath, f"calibration.fits"), overwrite=True)
        self._logger.info("Calibration FITS file saved.")

        del convolved, final
        gc.collect()

        self._logger.info("Observation complete.")
        self.is_observed = True

        return tn

    def show_system(self, out: bool = False, **kwargs: dict[str, _xt.Any]) -> None: # type: ignore
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
        Create a generator of images of a binary star system, in which each image
        corresponds to a different angular position of one star with respect to 
        the other.

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
        ring = self._create_ring(radius=self.distance)
        x, y = _np.where(ring == 1)
        for i, j in zip(x, y):
            yield (i, j)

    def compute_scan_angle(self, img: _xt.Array) -> _u.Quantity:
        """
        Compute the scan angle of the binary system from a given image.

        Parameters
        ----------
        img : _xt.Array
            The input image of the binary system.

        Returns
        -------
        _u.Quantity
            The scan angle in degrees.
        """
        y, x = xp.np.where(img != 0)
        xc, yc = self._base_map.shape[1] // 2, self._base_map.shape[0] // 2
        xi = [k for k in x if k != xc]
        yi = [k for k in y if k != yc]
        if len(xi) == 0 and len(yi) > 0:
            xi = xc
            yi = yi[0]
        elif len(yi) == 0 and len(xi) > 0:
            yi = yc
            xi = xi[0]
        elif len(yi) == 0 and len(xi) == 0:
            raise ValueError("Something's wrong with the binary map")
        else:
            xi = xi[0]
            yi = yi[0]
        dx = xi - xc
        dy = yi - yc
        phi = (xp.np.arctan2(dy, dx)) % (2 * xp.np.pi) * _u.rad
        return phi.to_value(_u.deg)

    def _create_base_map(self):
        """
        Create the base map with the primary star at the center and the secondary
        star at a distance of `self.distance` in a random direction.
        """
        center = (self.map_shape[0] // 2, self.map_shape[1] // 2)
        _map = _np.zeros(self.map_shape, dtype=_np.float64)
        _map[center] += self.central_star_flux
        self._logger.info("Base map with central star created.")
        return _map

    def _prepare_main_header(self):
        """
        Prepare the main header for the FITS file.
        """
        header = {}
        header["DISTMAS"] = (self.distance, "Angular separation in mas")
        header["M1"] = (self.M1, "Magnitude of the primary (central) star")
        header["M2"] = (self.M2, "Magnitude of the secondary (companion) star")
        header['GMAG'] = (self._Mtot, "Calibration G-Magnitude of the expected source")
        header["BAND"] = (self._bands["band"][0], "Photometric band")
        header["WAVELEN"] = (
            self._bands["wavelength"][0].to_value(_u.nm),
            "Effective wavelength in nm",
        )
        header["ZP"] = (self._bands["zero_point"][0].value, "Zero point in Jy")
        header["BANDWID"] = (
            self._bands["bandwidth"][0].to_value(_u.nm),
            "Bandwidth in nm",
        )
        return header

    def _compute_star_flux(
        self,
        star: "Star",
    ) -> _u.Quantity:
        """
        Calculate photon flux for magnitude M star in V-band.

        Parameters
        ----------
        star : Star
            The star object containing magnitude and other properties.

        Returns
        -------
        Quantity
            Photon flux in photons per second per cm².
        """
        return (
            star.flux * \
                self.collecting_area.to(_u.cm**2) * \
                    self.integration_time.to(_u.s)
        ).to_value()


    def _create_ring(
        self, radius: int, show: bool = False
    ):
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
        shape = self.map_shape
        angle = self.angle.value
        from skimage.draw import disk

        outer_disk = disk((shape[0] // 2, shape[1] // 2), radius, shape=shape)
        inner_disk = disk((shape[0] // 2, shape[1] // 2), radius - 1, shape=shape)
        ring_mask = _np.zeros(shape, dtype=bool)
        ring_mask[outer_disk] = True
        ring_mask[inner_disk] = False  # remove inner disk, leaving only the ring
        if angle < 360.0:
            yc, xc = shape[0] // 2, shape[1] // 2
            theta = _np.deg2rad(angle)
            for y in range(shape[0]):
                for x in range(shape[1]):
                    if ring_mask[y, x]:
                        dy = y - yc
                        dx = x - xc
                        phi = _np.arctan2(dy, dx)
                        if not (0 <= phi <= theta):
                            ring_mask[y, x] = False
        if show:
            extent = (-shape[1] // 2, shape[1] // 2, -shape[0] // 2, shape[0] // 2)
            _plt.figure(figsize=(8, 7))
            _plt.imshow(ring_mask, origin="lower", extent=extent, cmap="inferno")
            _plt.xlabel("X [mas]")
            _plt.ylabel("Y [mas]")
            _plt.title(f"Ring mask with radius {radius} mas")
            _plt.show()
        return ring_mask
    
    def update_record_file(self, tn: str, other_params: dict[str,_xt.Any] = None) -> None: # type: ignore
        """
        Update the record file with the current binary system parameters.
        """
        import pandas as pd

        record_path = os.path.join(SIMPATH, "simulations_record.csv")

        basedict ={
            "TN": [tn],
            "M1": [self.M1],
            "M2": [self.M2],
            "G" : [self._Mtot],
            "D_mas": [self.distance],
            "φ_max": [self.angle.value],
        }

        if other_params:
            basedict.update(other_params)

        df = pd.DataFrame(basedict)

        if os.path.exists(record_path):
            existing_df = pd.read_csv(record_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(record_path, index=False)

        self._logger.info("Simulation record file updated.")

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


class Star:
    
    def __init__(
        self,
        *,
        magnitude: float|_u.Quantity,
        type_or_temp: str|int = 5800 * _u.K,
        band: str = 'gaia_g'
    ):
        """Initialize a Star object with its magnitude and band information.
        
        Parameters
        ----------
        magnitude : float or astropy.units.Quantity
            The magnitude of the star. If a float is provided, it is assumed to 
            be in magnitudes and will be converted to an astropy Quantity with 
            units of magnitude.
        type_or_temp : str or int, optional
            The spectral type or effective temperature of the star. This can be
            used to select a template spectrum for the star. Default is 5800 K,
            which corresponds to a G-type star like the Sun.
        band_dict : dict
            A dictionary containing the band information, including 'name', 
            'wavelength', 'throughput', and 'zero_point'. The values in the 
            dictionary should be astropy Quantities with appropriate units.
        """
        
        self.magnitude = magnitude
        self.band = band

        self._build_passband()
        self._resolve_spectral_type(type_or_temp)
        self._build_spectrum()
    
    def __repr__(self):
        """String representation of the Star."""
        return f"""Star(G={self.magnitude} mag, T={self.temperature:.0f} K)"""
    
    @property
    def flux(self):
        """Return the integrated flux of the star over the bandpass."""
        return self.spectrum.integrate(self.wavelength)
    
    def show_spectrum(self, **kwargs: dict[str, _xt.Any]) -> None: # type: ignore
        """
        Plot the spectrum of the star.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plot function. 
            This include 
            - 'xlim' and 'ylim' to set the limits of the axes.
        """
        w, y = self.spectrum._get_arrays(self.wavelength)
        
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        _plt.figure()
        _plt.plot(w.to_value(self.wavelength.unit), y, **kwargs)
        _plt.xlabel(f"Wavelength [{self.wavelength.unit.to_string()}]")
        _plt.ylabel(f"Flux Density [{y.unit.to_string()}]")
        _plt.title(f"Star Spectrum: {self.magnitude} mag, {self.temperature:.0f} K")
        if xlim is not None:
            _plt.xlim(xlim)
        if ylim is not None:
            _plt.ylim(ylim)
        _plt.show()
        
    def _build_passband(self):
        """Extract band information from the provided dictionary."""
        band = Band(self.band)
        
        for attr in ["name", "wavelength", "zero_point", "transmission"]:
            if not hasattr(band, attr):
                raise ValueError(f"Band information is missing the '{attr}' attribute.")
            else:
                setattr(self, attr, getattr(band, attr))
        
        if all([self.transmission is None, self.wavelength is not None]):
            raise RuntimeError(
                "Transmission information is missing. Cannot build the bandpass without it."
            )
        
        self.bandpass = _sp.SpectralElement(
            _sp.models.Empirical1D,
            points = self.wavelength,
            lookup_table = _np.asarray(self.transmission, dtype=_np.float64)
        )
        
    
    def _build_spectrum(self):
        """Build the spectrum of the star based on its magnitude."""
        base_spectra = _sp.SourceSpectrum(
            _sp.BlackBodyNorm1D,
            temperature=self.temperature,
        )

        ## Get the band-averaged Flux density
        flux_density = _sp.Observation(
            base_spectra,
            self.bandpass,
            binset=self.wavelength,
            force='taper'
        ).effstim(_u.Jy)

        ## Flux Density Scale factor from to Gmag
        sf = (self.zero_point / flux_density).decompose().value

        ## Now scale the base_spectra to the 0 G magnitude flux density in the G
        ## band observe it, and then scale it to the actual magnitude of the star
        final_spectra = _sp.Observation(
            base_spectra * sf,
            self.bandpass,
            binset=self.wavelength,
            force='taper',
        )
        
        self.spectrum = final_spectra * 10 ** (-0.4 * self.magnitude)
    
    def _resolve_spectral_type(self, type_or_temp: str|int):
        """Resolve the spectral type or effective temperature of the star."""
        if isinstance(type_or_temp, str):
            type_or_temp = type_or_temp.upper()
            if type_or_temp in _SPECTRAL_TYPE_MAPPING:
                self.temperature = _SPECTRAL_TYPE_MAPPING[type_or_temp]
            else:
                raise ValueError(f"Unknown spectral type: {type_or_temp}. Valid types are: {list(_SPECTRAL_TYPE_MAPPING.keys())}")
        else:
            self.temperature = self._resolve_unit(type_or_temp, "K")
    
    @staticmethod
    def _resolve_unit(q: float|_u.Quantity, unit: str):
        """
        Resolve the unit of a given value based on the specified unit string.
        
        Parameters
        ----------
        q : float or astropy.units.Quantity
            The value to be converted or assigned a unit.
        unit : str
            The unit to convert to or assign if the value is a float.
        
        Returns
        -------
        astropy.units.Quantity
            The value with the specified unit.
        """
        if hasattr(q, 'unit'):
            return q.to(unit)
        else:
            return q * _u.Unit(unit)
            
    
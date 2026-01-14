import os
import gc
import xupy as xp
import numpy as _np
from .utils import *
import astropy.units as _u
from astropy.io import fits
from .instruments import CCD
import matplotlib.pyplot as _plt
from xupy import typings as _xt
from .utils.logging import SystemLogger as _SL
from opticalib.ground.osutils import _header_from_dict as hfd


class GaiaSimulator:
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
    angle : float or astropy.units.Quantity, optional
        Total angle to cover in the simulation in degrees. Default is 360 degrees.

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
        distance: float | _u.Quantity,
        angle: float | _u.Quantity = 360.0,
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

        if not isinstance(distance, _u.Quantity):
            self.distance = distance * _u.mas  # assuming input is in mas
        else:
            if distance.unit != _u.mas:
                self.distance = distance.to(_u.mas)
        self.distance = int(self.distance.value)  # mas
        self._logger.info(f"Angular separation set to {self.distance} mas.")

        if any([s < 2 * self.distance for s in list(self.map_shape)]):
            raise ValueError(
                f"Map shape must be larger than twice the distance of the binary system: {self.map_shape} < {2*self.distance*2}"
            )

        self.M2 = M2
        self.M1 = M1
        self._logger.info(f"Central star magnitude: {self.M1} mag.")
        self._logger.info(f"Companion star magnitude: {self.M2} mag.")

        self._Mtot = -2.5 * _np.log10(10 ** (-0.4 * M1) + 10 ** (-0.4 * M2))
        self._logger.info(
            f"Calibration G-Magnitude of the single source: {self._Mtot} mag."
        )

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
            raise ValueError(f"Something's wrong...")

        del G

        self.collecting_area = self.ccd.WC[idx]["area_um"]
        self.t_int = self.ccd.tdi

        self.central_star_flux = self._compute_star_flux(
            self.M1, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self._logger.info(f"Central star flux: {self.central_star_flux} photons/s/cm².")

        self.comp_star_flux = self._compute_star_flux(
            self.M2, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self._logger.info(f"Companion star flux: {self.comp_star_flux} photons/s/cm².")

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
        N = self._create_ring(radius=self.distance).sum()
        datapath = create_data_folder()
        tn = datapath.split("/")[-1]
        self._logger.info("Starting binary star observations...")
        self._logger.info(f"Data Tracking Number : {tn}")

        from tqdm import tqdm

        header = self._prepare_main_header()
        header["PIXELSCL"] = (self.ccd.header["PIXELSCL"], "Pixel scale [mas/pixel]")
        imgHeader = self._prepare_main_header()
        imgHeader["PXSCLREB"] = (self.ccd.pxscale_factor, "Pxscale rebin ratio (y/x)")
        imgHeader["PXSCLXRB"] = (
            self.ccd.pxscale_x.value,
            "Rebinned pxscale_x [mas/pixel]",
        )
        imgHeader["PXSCLYRB"] = (
            self.ccd.pxscale_y.value,
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
                f"Image {i:04d}: Convolution complete at angle {phi:.2f} degrees."
            )

            # Add Poisson noise to the convolved image
            # noisy = _np.random.poisson(convolved).astype(_np.float32)
            # self._logger.info(f"Image {i:04d}: photon shot noise added.")

            psf_2d, _, _ = ccd.sample_psf(psf=convolved)
            self._logger.info(f"Image {i:04d}: PSF read-out complete (binning).")

            self._logger.debug(
                f"Image {i:04d}: shifting map based on sources distance."
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
            ron = (
                _np.random.normal(0, _np.random.randint(2, 6), size=psf_2d.shape) * 0.5
            )
            ron[ron < 0] = 0
            psf_2d += ron
            self._logger.info(f"Image {i:04d}: read-out noise added.")

            psf_x, psf_y = computeXandYpsf(psf=psf_2d)
            self._logger.info(f"Image {i:04d}: PSF X and Y computed.")
            final = (psf_2d, psf_x, psf_y)

            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU(final[0], header=h1))
            hdul.append(fits.ImageHDU(final[1], name="PSF_X"))
            hdul.append(fits.ImageHDU(final[2], name="PSF_Y"))
            hdul.append(fits.ImageHDU(convolved, name="HighRes obs", header=h2))
            hdul.writeto(os.path.join(datapath, f"{i:04d}.fits"), overwrite=True)
            self._logger.info(f"Image {i:04d}: FITS file saved.")
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
        final = ccd.sample_psf(psf=convolved)

        hdul = fits.HDUList()
        for k in ["M1", "M2", "DISTMAS", "PHI"]:
            h1.pop(k)
            h2.pop(k)
        h1["GMAG"] = (self._Mtot, "Calibration G-Magnitude of the expected source")
        h2["GMAG"] = (self._Mtot, "Calibration G-Magnitude of the expected source")

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
        Create the base map with the primary star at the center and the secondary star
        at a distance of `self.distance` in a random direction.
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
        self, radius: int, show: bool = False
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

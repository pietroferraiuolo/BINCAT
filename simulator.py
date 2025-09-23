from dataclasses import dataclass
import os, gc, xupy as xp, poppy as _poppy

_np = xp.np
import matplotlib.pyplot as _plt
from xupy import typings as _xt
from astropy.table import QTable as _qt
from astropy import convolution as _c
import astropy.units as _u
from utils import *
from utils import _header_from_dict as hfd

_l = Logger()
basepath = os.path.dirname(os.path.abspath(__file__))


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
        psf: _xt.Optional[fits.HDUList | _xt.ArrayLike | str] = None,
        **kwargs: dict[str, _xt.Any],
    ):
        """The constructor"""
        if psf is not None:
            if isinstance(psf, fits.HDUList):
                self._psf = psf
                self._meta = psf[0].header
            elif isinstance(psf, str):
                self._psf = fits.open(psf)
                self._meta = self._psf[0].header
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
            self._meta = None
            print(
                """PSF not provided. Use the method 'compute_psf' to compute it, passing a
`poppy` optical system."""
            )
        self._bands = _qt.read("data/bands.fits")
        self._bands = self._bands[self._bands["band"] == kwargs.get("band", "Gaia_G")]
        self._passbands = _qt.read("data/gaiaDR3passband.fits")
        self.pixel_area = kwargs.get("pixel_area", 10 * 30 * _u.um**2)
        self.ccd_pixels = kwargs.get("ccd_pixels", [4500 * _u.pixel, 1966 * _u.pixel])
        self.pxscale_x = kwargs.get("pixel_scale_x", None)
        self.pxscale_y = kwargs.get("pixel_scale_y", None)
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
        self._wc_conditions = ["G<13", "13<=G<16", "16<=G"]

    @property
    def header(self) -> fits.Header | None:
        """
        Returns the header of the PSF FITS file if available.
        """
        return self._meta

    def rebin_psf(
        self,
        psf: _xt.Optional[fits.HDUList | _xt.ArrayLike] = None,
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

    def display_psf(
        self,
        psf: _xt.Optional[_xt.ArrayLike] = None,
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
        if psf is not None:
            try:
                if isinstance(psf, (list, tuple)) and len(psf) == 3:
                    psf_x = psf[1]
                    psf_y = psf[2]
                    psf = psf[0]
                else:
                    psf_x, psf_y = self._computeXandYpsf(psf=psf[0])
                    if (
                        psf_x.shape[0] != psf.shape[0]
                        and psf_y.shape[0] != psf.shape[1]
                    ):
                        raise ValueError("Something's wrong with the passed PSF")
            except Exception as e:
                _l.log(e, level="ERROR")
                raise (e)

            fig = _plt.figure(figsize=(8, 4))

            # Left: imshow (spans full height, 1/3 width)
            cmap = kwargs.pop("cmap", "gist_heat")
            aspect = kwargs.pop("aspect", "auto")
            ax1 = _plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
            ax1.imshow(psf, cmap=cmap, aspect=aspect)
            ax1.axis("off")  # to hide axes

            # Right top: first plot (top half of right, 2/3 width)
            ax2 = _plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
            ax2.plot(psf_x, linewidth=2, color="tab:red")
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.set_ticks_position("right")
            ax2.xaxis.set_ticklabels([])
            ax2.grid(True, linestyle="--", alpha=0.85)

            # Right bottom: second plot (bottom half of right, 2/3 width)
            ax3 = _plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
            ax3.plot(psf_y, linewidth=2, color="tab:red")
            ax3.xaxis.set_ticklabels([])
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.set_ticks_position("right")
            ax3.grid(True, linestyle="--", alpha=0.85)

            fig.suptitle(
                "2D PSF (left) and 1D Profiles (right): X (up) and Y (down)",
                fontsize=14,
                weight="semibold",
            )
            _plt.tight_layout()
            _plt.show()

            return fig
        if self._psf is None:
            raise ValueError("PSF has not been computed yet.")
        if mode == "2d":
            if not self.rebinned:
                _poppy.display_psf(
                    self._psf, title="CCD PSF", normalize="peak", **kwargs
                )
                return
            else:
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
                _plt.show()
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
        self.is_observed = False
        self._bands = self.ccd._bands
        # For now hardcoded, but should be passed as parameters (or read from CCD)
        self.collecting_area = (18 * 10 * _u.um) * (
            12 * 30 * _u.um
        )  # Gaia CCD pixel window area (G<13)
        self.t_int = 4.42 * _u.s  # Gaia CCD integration time
        self.comp_star_flux = self._compute_star_flux(
            self.M2, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self.central_star_flux = self._compute_star_flux(
            self.M1, collecting_area=self.collecting_area, integration_time=self.t_int
        )
        self._base_map = self._create_base_map()

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
        header = hfd(ccd.header)
        assert ccd.psf.shape == self._base_map.shape, "PSF and map shapes do not match."
        for img in tqdm(
            self.transit(), desc=f"[{tn}] Observing...", unit="images", total=N
        ):
            convolved = _c.convolve_fft(
                img, ccd.psf, boundary="wrap", normalize_kernel=True, allow_huge=True
            )
            # Add Poisson noise and Readout Noise to the convolved image
            # noisy = _np.random.poisson(convolved).astype(_np.float32)
            # noisy += _np.random.normal(0, 5, size=noisy.shape)  # readout noise
            # del convolved
            # gc.collect()
            final = ccd.rebin_psf(convolved, rebin_factor=59, axis_ratio=(1,3))
            del convolved
            gc.collect()
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU(final[0], header=header))
            hdul.append(fits.ImageHDU(final[1], name="PSFX"))
            hdul.append(fits.ImageHDU(final[2], name="PSFY"))
            hdul.writeto(os.path.join(datapath, f"{i:04d}.fits"), overwrite=True)
            i += 1

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


import dataclasses
@dataclass(init=True, frozen=True, repr=True)
class _PSFData():

    def __init__(self, psf: list[_xt.Array]):
        self.psf = psf
        self.psf_x = None
        self.psf_y = None
        self.meta = None
        self.shape = None
    
    def __repr__(self):
        return f"PSFData(shape={self.shape}, meta={self.meta})"
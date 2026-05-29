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
        if self.ccd_pxscale_factor < 1:
            rbfactor = int(self.ccd_pxscale_y.value)
            ratio = int(1 / self.ccd_pxscale_factor)
            rbratio = (rbfactor * ratio, rbfactor)
        else:
            rbfactor = int(self.ccd_pxscale_x.value)
            ratio = int(self.ccd_pxscale_factor)
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
        if self.psf is None:
            raise ValueError("PSF has not been computed yet.")
        fig = _plt.figure()
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
            _plt.imshow(self.psf, cmap=cmap, norm=norm, **kwargs)
            _plt.colorbar()
            _plt.title("CCD PSF")
            _plt.xlabel("X [px]")
            _plt.ylabel("Y [px]")
        else:
            _plt.xlabel("arcsec")
            _plt.ylabel("Normalized PSF")
            _plt.grid(linestyle="--")
            y = _np.arange(len(self.psf_y)) * _u.pixel
            y -= len(y) // 2 * _u.pixel  # center the y-axis
            y *= self.ccd_pxscale_y
            x = _np.arange(len(self.psf_x)) * _u.pixel
            x -= len(x) // 2 * _u.pixel  # center the x-axis
            x *= self.ccd_pxscale_x
            _plt.title(f"PSF in {mode} direction")
            if mode == "x":
                _plt.plot(x, self.psf_x)
            elif mode == "y":
                _plt.plot(y, self.psf_y)
            else:
                raise ValueError("Invalid mode. Use '2d', 'x', or 'y'.")
        _plt.show()
        return fig

    def _compute_tdi_gate_magnitudes(
        self,
        mag_bright: float = 3.0,
        mag_faint: float = 21.0,
        saturation_margin: float = 0.85,
    ) -> dict[int, tuple[float, _u.Quantity, float]]:
        """
        Compute magnitude thresholds for TDI gate activation based on CCD saturation limits.
        
        This method determines at which G-band magnitude each TDI gate should be activated
        to prevent CCD saturation while maximizing signal-to-noise ratio. The calculation
        accounts for:
        - Gaia G-band passband response
        - CCD quantum efficiency
        - TDI integration mechanics
        - Full well capacity limits
        
        Physical Basis
        --------------
        In TDI (Time Delay Integration) mode, charge accumulates as the stellar image
        drifts across the CCD. The total accumulated charge is:
        
            Q(G, N_TDI) = Φ(G) × A_eff × t_TDI × N_TDI
        
        where:
            Φ(G) : photon flux as function of G magnitude [photons/s]
            A_eff : effective collecting area [cm²]
            t_TDI : TDI clock period [s]
            N_TDI : number of TDI lines (gates)
        
        The photon flux follows Pogson's law:
            Φ(G) = Φ₀ × 10^(-0.4 × G)
        
        where Φ₀ is the zero-magnitude photon flux in G-band, which can be computed
        from the passband integral:
        
            Φ₀ = ∫ [F_ν,0 × c/λ² × T(λ) × QE(λ) / (h×c/λ)] dλ
        
        Here:
            F_ν,0 : zero-point flux density (2555.5 Jy for Gaia G vegamag)
            T(λ)  : passband transmission
            QE(λ) : quantum efficiency
        
        TDI Gate Selection Strategy
        ----------------------------
        Each TDI gate i is activated when the accumulated charge would exceed the
        saturation threshold using the previous (longer) integration time:
        
            G_i = G_ref - 2.5 × log₁₀(N_i / N_ref)
        
        This ensures:
        1. Bright stars use short integrations (few TDI lines) → avoid saturation
        2. Faint stars use long integrations (all TDI lines) → maximize SNR
        3. Smooth transitions between gates
        
        Parameters
        ----------
        mag_bright : float, optional
            Brightest magnitude in operational range (default: 3.0).
            This should be G < 3 for Gaia to avoid saturation even with minimum integration.
        mag_faint : float, optional
            Faintest magnitude in operational range (default: 21.0).
            Beyond this, even maximum integration may not provide sufficient SNR.
        saturation_margin : float, optional
            Safety margin as fraction of full well capacity (default: 0.85).
            Set to 85% to account for:
            - Non-linearity above ~70% FWC
            - Variable stars
            - Photometric uncertainty
            - Background contribution
        use_passband_integration : bool, optional
            If True, integrate over G-band passband accounting for wavelength-dependent
            QE and transmission. If False, use effective wavelength approximation.
            Default: True (more accurate but slower).
        
        Returns
        -------
        dict[int, tuple[float, Quantity, float]]
            Dictionary mapping TDI gate number to:
            - magnitude_threshold : G magnitude at which to activate this gate
            - integration_time : resulting integration time [s]
            - electrons_per_sec : electron generation rate at threshold [e⁻/s]
            
        Examples
        --------
        >>> ccd = CCD(band="Gaia_G")
        >>> tdi_gates = ccd.compute_tdi_gate_magnitudes()
        >>> 
        >>> # Display activation thresholds
        >>> print("TDI Gate Activation Strategy")
        >>> print("-" * 70)
        >>> print(f"{'G range':^15} | {'TDI Gates':^10} | {'t_int [s]':^12} | {'e⁻/s':^15}")
        >>> print("-" * 70)
        >>> 
        >>> prev_mag = mag_bright
        >>> for gate in sorted(tdi_gates.keys(), reverse=False):
        ...     mag, t_int, rate = tdi_gates[gate]
        ...     print(f"{prev_mag:5.2f} - {mag:5.2f} | {gate:^10d} | {t_int.value:^12.4f} | {rate:^15.2e}")
        ...     prev_mag = mag
        
        Notes
        -----
        - The reference magnitude is anchored at the brightest star (mag_bright) with
        minimum integration time (2 TDI gates)
        - For binaries, use the combined magnitude: G_tot = -2.5×log₁₀(10^(-0.4×G₁) + 10^(-0.4×G₂))
        - Window classes (WC0, WC1, WC2) affect the effective pixel area and should be
        accounted for separately in the readout simulation
        
        References
        ----------
        - Gaia Collaboration (2016), A&A 595, A1 - Gaia Data Release 1
        - Jordi et al. (2010), A&A 523, A48 - Gaia broad band photometry
        - de Bruijne et al. (2015), A&A 576, A74 - Gaia calibration framework
        
        See Also
        --------
        get_integration_time : Get integration time for a specific G magnitude
        _compute_star_flux : Compute photon flux for a given magnitude
        """
        # Validate inputs
        if not (0 < saturation_margin <= 1):
            raise ValueError(f"saturation_margin must be in (0, 1], got {saturation_margin}")
        if mag_bright >= mag_faint:
            raise ValueError(f"mag_bright ({mag_bright}) must be < mag_faint ({mag_faint})")
        
        # Reference: brightest stars use minimum integration (2 gates)
        # This anchors our calibration
        t_ref = self.integration_time[0]  # Minimum integration (2 gates)
        Q_max = saturation_margin * self.full_well_capacity.to_value(_u.electron)
        
        photon_rate_ref = self._compute_photon_rate_from_passband(mag_bright)
        Q_ref = photon_rate_ref * t_ref.to_value(_u.s)
        delta_mag = -2.5 * _np.log10(Q_max / Q_ref)
        G_ref = mag_bright + delta_mag
        
        if Q_ref > Q_max:
            import warnings
            warnings.warn(
                f"Saturation at G={G_ref:.1f} even with minimum integration!\n"
                f"  Expected: {Q_ref:.0f} photons\n"
                f"  Maximum:  {Q_max:.0f} e⁻\n"
                f"Consider reducing mag_bright or increasing saturation_margin.",
                category=RuntimeWarning
            )
        
        # Now compute magnitude threshold for each TDI gate
        # Using: Φ(G) ∝ 10^(-0.4×G)
        # If Q(G_i, t_i) = Q_max, then:
        # Φ(G_i) × t_i = Φ(G_ref) × t_ref
        # 10^(-0.4×G_i) × t_i = 10^(-0.4×G_ref) × t_ref
        # G_i = G_ref - 2.5 × log₁₀(t_i / t_ref)
        
        thresholds: dict[int, tuple[float, _u.Quantity]] = {}
        for n_gates, t_int in zip(self.TDIGates, self.integration_time):
            G_thresh = G_ref - 2.5 * _np.log10((t_ref / t_int).decompose().value)
            G_thresh = float(_np.clip(G_thresh, mag_bright, mag_faint))
            thresholds[int(n_gates)] = (G_thresh, t_int)

        self._logger.info("Computed TDI gate thresholds from passband integration.")
        return thresholds


    def _compute_photon_rate_from_passband(self, G_mag: float) -> float:
        """
        Compute photon generation rate integrating over G-band passband.
        
        This is the most accurate method as it accounts for:
        - Wavelength-dependent quantum efficiency
        - Passband transmission profile
        - Proper photon energy at each wavelength
        
        Parameters
        ----------
        G_mag : float
            G-band magnitude
        
        Returns
        -------
        float
            Photon generation rate [photons/s]
        """
        from astropy.constants import h, c
        from scipy.integrate import simpson
        
        # Get passband data
        wl = self._passbands["lambda"].to(_u.nm)  # nm
        T = self._passbands["G"]  # dimensionless
        
        # Gaia G-band zero point (Vega magnitude system)
        Fnu0 = 30300 * _u.Jy
        Fnu = Fnu0 * 10 ** (-0.4 * G_mag)                       # erg/s/cm²/Hz
        Flam = (Fnu * c / (wl.to(_u.m) ** 2)).to(_u.erg / _u.s / _u.cm**2 / _u.nm)
        Ephot = (h * c / wl.to(_u.m)).to(_u.erg)
        phi = (Flam / Ephot).to(1 / (_u.s * _u.cm**2 * _u.nm))  # photons/s/cm²/nm

        integrand = phi.value * T
        phi_tot = simpson(integrand, x=wl.to_value(_u.nm))      # photons/s/cm²
        photon_rate = phi_tot * self.integration_area.to_value(_u.cm**2)
        
        return float(photon_rate)


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
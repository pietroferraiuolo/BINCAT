import numpy as _np
import astropy.units as _u
from pprint import pformat as _pformat
from inspect import Parameter, Signature
from .. import utils as _ut
from grasp.stats import fit_data_points
from grasp.plots import histogram
from scipy.signal import find_peaks, argrelextrema


def _harmonic_decomposition(x, *coeffs):
    """
    Generalized harmonic decomposition of order N:
    
    ... math::
        f(φ) = c0 + Σ_{k=1}^{N} [c_{2k} cos(2k φ) + s_{2k} sin(2k φ)]
    
    coeffs: [c0, c2, s2, c4, s4, ..., c_{2N}, s_{2N}]
    """
    x = _np.deg2rad(x)
    result = coeffs[0]
    for i, k in enumerate(range(1, (len(coeffs)) // 2 + 1)):
        c = coeffs[2*i + 1]
        s = coeffs[2*i + 2]
        result += c * _np.cos(2*k * x) + s * _np.sin(2*k * x)
    return result

def harmonic_decomposition(order: int):
    """
    Factory function that returns a harmonic decomposition function of order N.
    
    Parameters
    ----------
    order : int
        The maximum harmonic order N.
    
    Returns
    -------
    callable
        A function f(x, c0, c2, s2, ..., c_{2N}, s_{2N}).
    """
    params = [Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)]
    params += [Parameter('c0', Parameter.POSITIONAL_OR_KEYWORD)]
    for k in range(1, order + 1):
        params += [
            Parameter(f'c{2*k}', Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(f's{2*k}', Parameter.POSITIONAL_OR_KEYWORD),
        ]
    def func(x, *coeffs):
        return _harmonic_decomposition(x, *coeffs)
    func.__signature__ = Signature(params) # type: ignore
    func.__name__ = f'harmonic_decomposition_order{order}'
    return func


class IPD():
    """
    Class to perform IPD analysis on a PSF cube.
    
    Parameters
    ----------
    tn : str
        The transit name for which to run the IPD analysis.
    fitted_parameters : int, optional
        The number of fitted parameters. Default is 1.
    
    Attributes
    ----------
    tn : str
        The transit name.
    cube : _ut.PSFCube
        The PSF cube loaded from the transit name.
    calibration : _ut.PSFData
        The calibration PSF data extracted from the cube.
    P : int
        The number of fitted parameters.
    N : int
        The number of observations in the cube.
    dof : int
        The degrees of freedom for the chi-squared calculation (N - P).
    chi2 : np.ndarray or None
        The chi-squared values for each observation in the cube. Initially None.
    phi : np.ndarray or None
        The scan angles for each observation in the cube. Initially None.
    
    Methods
    -------
    ``chiphi()``
        Generator that yields (chi2, phi) pairs sorted by increasing phi.
    ``chi2(which='2d')``
        Compute the reduced chi-squared statistic for the PSF fit of each observation in the cube.
    ``GoF(which='2d')``
        Compute the goodness-of-fit statistic based on the chi-squared values.
    ``uwe()``
        Compute the unit weight error (UWE) for the astrometric fit of a 5-parameter solution.
    ``fit(degree=2)``
        Fit a polynomial of given degree to the chi-squared values as a function of phi.
    ``__call__()``
        Alias for the fit method, allowing the object to be called directly to perform the fit.
    """
    def __init__(
        self, 
        tn: str|None = None, *,
        cube: _ut.PSFCube | None = None,
        fitted_parameters: int = 1
    ):
        """The constructor"""
        ## Data ##
        if tn is not None:
            self.tn = tn
            self.cube = cube if cube is not None else _ut.load_psf(tn)
        elif cube is not None:
            self.cube = cube
            self.tn = None
        else:
            raise ValueError("Either 'tn' or 'cube' must be provided.")
        self.calibration = self.cube.calibration
        
        ## Parameters ##
        self.P = fitted_parameters
        self.N = _np.size(self.cube[0].psf_2d)
        self.dof = self.N - self.P
        
        ## Computed ##
        self.chi2_2d = None
        self.chi2_al = None
        self.chi2_ac = None
        self.phi = None
        self._fit = None
        self.gof_amp = None
        self.gof_phase = None
        self.uwe = None
        
        # pre-compute chi2 values for all PSFs in the cube
        self._compute_cube_chi2()

    def set_tn(self, tn: str):
        """
        Set a new tracking number and reload the cube and calibration.
        
        Parameters
        ----------
        tn : str
            The new tracking number data folder to load.
        """
        self.tn = tn
        self.cube = _ut.load_psf(tn)
        self.calibration = self.cube.calibration
        self.N = _np.size(self.cube[0].psf_2d)
        self.dof = self.N - self.P
        self.chi2_2d = None
        self.chi2_al = None
        self.chi2_ac = None
        self.phi = None
        self._fit = None
        self.gof_amp = None
        self.gof_phase = None
        self.uwe = None
        self._compute_cube_chi2()  # re-compute chi2 values for the new cube

    def _compute_cube_chi2(self):
        """
        Compute the reduced chi-squared statistic for the PSF fit of each observation in the cube.
        """
        chi2 = []
        for w in ['2d', 'al', 'ac']:
            chi2.append(self._compute_chi2(which=w))
        
        self.chi2_2d, self.chi2_al, self.chi2_ac = chi2
        return "Chi-squared values computed for 2D, AL, and AC PSFs."
    
    def GoF(self, which: str = '2d'):
        """
        Compute the goodness-of-fit statistic based on the chi-squared values.
        
        The GoF statistic is defined as:
        
        ... math::
            GoF = sqrt(9*DOF/2) * ((chi2/DOF)^(1/3) - 1 + 2/(9*DOF))
    
        where DOF is the degrees of freedom (N - P) and chi2 is the chi-squared 
        value for the chosen PSF fit.
        """
        match which:
            case '2d':
                stat = self.chi2_2d
            case 'al':
                stat = self.chi2_al
            case 'ac':
                stat = self.chi2_ac
            case _:
                raise ValueError("Parameter 'which' must be one of '2d', 'al', or 'ac'.")

        return _np.sqrt(9*self.dof/2) * (stat**(1/3) - 1 + 2/(9*self.dof))

    def _uwe(self):
        """
        Compute the unit weight error (UWE) for the astrometric fit of a 
        5-parameter solution.

        UWE is defined as the square root of the reduced chi-squared statistic.
        """
        n_params = 5
        astrometric_chi2_al = self.chi2_al or self._compute_chi2(which='al')
        self.uwe = _np.sqrt(astrometric_chi2_al / (self.N - n_params))
        return self.uwe.copy()
    
    def harmonic_fit(self, order: int = 1, which: str = "2d"):
        """
        Fit the harmonic decomposition to the logarithm of the GoF reduced χ^2 
        values as a function of phi.
        
        Parameters
        ----------
        order : int, optional
            The order of the harmonic decomposition. Defaults to 1, i.e., up to
            the first harmonic:
            ... math::
                f(φ) = c_0 + c_2 cos(2φ) + s_2 sin(2φ)
        which : str, optional
            The PSF fit to use for the harmonic decomposition. 
            Must be one of '2d', 'al', or 'ac'. Defaults to '2d'.
        """
        y = _np.log(getattr(self, f"chi2_{which}").copy())
        X = self.phi.copy()

        fit = fit_data_points(
            data=y,
            x_data=X,
            method=harmonic_decomposition(order=order),
        )
        _, c2, s2, *__ = fit.coeffs

        self._fit = fit
        self.gof_amp = _np.sqrt(c2**2 + s2**2)
        self.gof_phase = _np.arctan2(s2, c2) * _u.rad.to(_u.deg) % 180
        
        return self.gof_amp, self.gof_phase
    
    def frac_multi_peak(
        self,
        threshold: float = 0.1,
        epsilon: float = 1e-6,
        verbose: bool = False
     ) -> tuple[float, float]:
        """
        Calculate the fraction of PSFs in the observed cube that have multiple
        peaks in the AL LSF, above a certain threshold.
        
        Parameters
        ----------
        threshold : float, optional
            The threshold value for identifying multiple peaks. Defaults to 0.1.
        epsilon : float, optional
            Tolerance value for the chi-squared threshold. Defaults to 1e-6.
        verbose : bool, optional
            If True, print the number of peaks for each PSF. Defaults to False.
        
        Returns
        -------
        (ipd_frac_multi_peak, ipd_frac_badfit) : tuple
            The fraction of PSFs in the cube that have multiple peaks above the 
            specified threshold and the fraction of PSFs that are considered bad fits.
        """
        _, pt = self._find_chi_threshold(epsilon=epsilon, verbose=verbose)
        
        mcounter: int = 0
        bfcounter: int = 0

        for psfd in self.cube:
            psf = psfd.psf_al
            peaks_al, _ = find_peaks(psf, height=threshold*psf.max())

            if len(peaks_al) > 1:
                mcounter += 1
            
            if psfd.phi <= pt:
                bfcounter += 1

            if verbose:
                print(f"Phi: {psfd.phi:.1f} deg - Peaks AL: {len(peaks_al)}")

        ipd_frac_multipeak = mcounter / self.N
        ipd_frac_badfit   = bfcounter / self.N

        self.frac_multipeak = ipd_frac_multipeak
        self.frac_badfit = ipd_frac_badfit

        return ipd_frac_multipeak, ipd_frac_badfit

    def show_harmonic_fit(self):
        """
        Plots the harmonic fit to the logarithm of the GoF reduced χ^2 values as 
        a function of phi.
        """
        
        from grasp import plots
        
        if len(self._fit.coeffs) > 5:
            legenon = False
        else:
            legenon = True

        fig, fax, _ = plots.regressionPlot(
            self._fit,
            f_type="datapoint",
            title=r"$A_{ipd}$"
            + f"={self.gof_amp:.2e}   |   "
            + r"$\varphi_{ipd}$"
            + f"={self.gof_phase:.1f} deg",
            xlabel=r"Scan Angle $\varphi$ [deg]",
            legend=legenon,
            dont_show=not legenon,
        )

        if legenon:
            label = plots._kde_labels(self._fit.kind, self._fit.coeffs)  # type: ignore
            label = (
                label.replace("Custom", "Harmonic")
                .replace("A", "c_0")
                .replace("B", "c_2")
                .replace("C", "s_2")
            )
            fax.legend([fax.lines[0]], [label], loc="best", fontsize="medium")

        fig.show()
    
    def show_polyfit(self,):
        """
        Shows the polynomial fit to the chi-squared values distribution.
        """
        
        from grasp import plots
        
        fig, fax, _ = plots.regressionPlot(
            self._polyfit,
            f_type="datapoint",
            title=rf"$\chi_{{\nu,\,AL}}^2$ Polynomial Fit",
            xlabel=r"$\chi_{\nu,\,AL}^2 $",
            size=15,
            grid=True,
            pc='blue'
        )
        
        label = plots._kde_labels(self._polyfit.kind, self._polyfit.coeffs)  # type: ignore
        label = (
            label.replace("Polynomial of degree 2", r"$ax^2 + bx + c$")
            .replace("A", "a")
            .replace("B", "b")
            .replace("C", "c")
        )
        fax.legend([fax.lines[0]], [label], loc="best", fontsize="medium")
        fig.show()
        
    def _find_chi_threshold(self, epsilon: float = 1e-4, verbose: bool = False):
        """
        Find the threshold in chi-squared values that separates good fits from bad fits.
        
        This is done by looking for a significant increase in chi-squared values as a 
        function of phi, which indicates a transition from good to bad fits.
        
        Parameters
        ----------
        epsilon : float, optional
            tolerance value for the chi-squared threshold. Defaults to 1e-4.
        verbose : bool, optional
            If True, print the phi value at the chi-squared threshold. Defaults 
            to False.
        
        Returns
        -------
        (chi_threshold, phi_threshold) : tuple
            The chi-squared threshold value that separates good fits from bad fits,
            with the corresponding ``phi`` value.
        """
        h = histogram(self.chi2_al, out=True, bins='detailed', dont_show=True)['h']
        hp = h['counts']
        hp_x = h['bins']
        polyfit = fit_data_points(hp, x_data=hp_x, method='poly2', plot=False)
        
        # analytical minimum of the fitted parabola: x = -b/(2a)
        a, b, _ = polyfit.coeffs
        ythresh = -b / (2 * a)

        # Find the index of the chi2_al value closest to the threshold
        k = int(_np.argmin(_np.abs(self.chi2_al - ythresh)))
        phithresh = self.phi[k]
        
        if verbose:
            print(f"Chi-squared threshold: {ythresh:.2e} at phi={phithresh:.1f} deg")
        
        self._chi2_threshold = ythresh
        self._phi_threshold = phithresh
        self._polyfit = polyfit

        return ythresh, phithresh

    def _compute_chi2(self, which: str = "2d", errors: float | None = None):
        """
        Compute the reduced chi-squared statistic for the PSF fit of each 
        observation in the cube.

        Parameters
        ----------
        which : str, optional
            Which PSF to use for the fit. Options are "2d", "al" or "ac".

        Returns
        -------
        np.ndarray
            An array of chi-squared values for each observation in the cube.
        """
        attr = f'psf_{which}'
        expected = getattr(self.calibration, attr)
        if errors is None:
            errors = expected #_np.ones_like(expected)

        chi2 = []
        phi  = []
        for obs in self.cube:
            psf = getattr(obs, attr)
            chi2.append(_np.sum(((psf - expected)**2/errors)) / self.dof)
            phi.append(obs.phi)

        chiphi = _np.array(list(zip(chi2, phi)))
        chiphi = chiphi[_np.argsort(chiphi[:, 1])]

        if self.phi is None:
            self.phi = chiphi[:,1].copy()

        return chiphi[:,0]
    
    @property
    def ipd(self):
        """
        Dictionary for the IPD model of the TN
        
        Returns
        -------
        dict
            A dictionary containing the IPD model parameters.
        """
        dt = {}
        dt['fit'] = self._fit
        dt['gof_amplitude'] = self.gof_amp
        dt['gof_phase'] = self.gof_phase
        dt['ipd_frac_multipeak'] = self.frac_multipeak
        dt['ipd_frac_badfit'] = self.frac_badfit
        dt['distance_mas'] = self.cube[0].primary_meta['DISTMAS']
        dt['central_mag'] = self.cube[0].primary_meta['M1']
        dt['secondary_mag'] = self.cube[0].primary_meta['M2']
        return dt

    @property
    def chi2(self):
        """
        Return the chi-squared values for the 2D, AL, and AC PSF fits.
        
        Returns
        -------
        np.ndarray
            An array of chi-squared values for each observation in the cube.
        """
        return (getattr(self, f"chi2_{x}").copy() for x in ['2d', 'al', 'ac'])
    
    def __call__(
        self,
        order: int = 1,
        which: str = '2d',
        threshold: float = 0.1,
        epsilon: float = 1e-4,
        verbose: bool = False
    ) -> "IPD":
        """
        Run the full IPD analysis pipeline, including chi-squared computation,
        harmonic fitting, and multi-peak fraction calculation.
        """
        self.harmonic_fit(order=order, which=which)
        self.frac_multi_peak(threshold, epsilon, verbose)
        return self
    
    def __repr__(self):
        txt = f"IPD(tn={self.tn},"
        if self.gof_amp is not None:
            txt += f" gof_amp={self.gof_amp:.2e},"
        if self.gof_phase is not None:
            txt += f" gof_phase={self.gof_phase:.1f} deg"
        if self.frac_multipeak is not None:
            txt += f" ipd_frac_multipeak={self.frac_multipeak:.2%},"
        if self.frac_badfit is not None:
            txt += f" ipd_frac_badfit={self.frac_badfit:.2%}"
        txt += ")"
        return txt

    def __str__(self):        
        return _pformat(self.ipd, sort_dicts=False, indent=2)

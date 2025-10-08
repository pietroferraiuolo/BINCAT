import xupy as _xp
from xupy import typings as _xt
import scipy.signal as _ss
import utils as _ut
import astropy.units as _u
from skimage.transform import resize
from matplotlib import pyplot as _plt
from grasp.stats import fit_data_points


######################
## IPD COMPUTATIONS ##
######################

def ipd_gof_harmonic(
    observed_cube: _ut.PSFData, calibrated_psf: _ut.PSFData, errors: _xt.Array = None, ddof: int = 0, show : bool = False
) -> float:
    """
    Calculate the goodness-of-fit statistic based on harmonic mean.
    """
    with _xp.NumpyContext() as np:
        chiphi = get_chi_phi(observed_cube, calibrated_psf, errors=errors, ddof=ddof)

        def harmonic_decomposition(x: float, c0: float, c2: float, s2: float) -> float:
            x = x * _u.deg.to(_u.rad)
            return c0 + c2*np.cos(2*x) + s2*np.sin(2*x)

        fit = fit_data_points(
            data = np.log(chiphi[:,0]),
            x_data = chiphi[:,1], 
            method = harmonic_decomposition
        )
        _, c2, s2 = fit.coeffs
        gof_amplitude = np.sqrt(c2**2 + s2**2)
        gof_phase = np.arctan2(s2, c2) * _u.rad.to(_u.deg)
    if show:
        from grasp import plots
        _plt.ion()
        fig, fax, _ = plots.regressionPlot(
            fit, 
            f_type='datapoint', 
            title=r'$A_{ipd}$'+f'={gof_amplitude:.1e}   |   '+r'$\varphi_{ipd}$'+f'={gof_phase:.2f} deg', 
            xlabel=r'Scan Angle $\varphi$'
        )
        label = plots._kde_labels(fit.kind, fit.coeffs)
        label = label.replace('Custom', 'Harmonic').replace('A', 'c_0').replace('B', 'c_2').replace('C', 's_2')
        fax.legend([fax.lines[0]], [label], loc='best', fontsize='medium')
        fig.show()
    return gof_amplitude, gof_phase


def ipd_frac_multipeak(
    cube: _ut.PSFData, threshold: float = 0.4, show: bool = False
) -> float:
    """
    Calculate the fraction of PSFs in the observed cube that have multiple peaks above a certain threshold.
    """
    maxs = []
    for img in cube:
        maxs.append(find_local_maxima(img, which='al', threshold=threshold))
    mcounter = 0
    for maxima in maxs:
        if len(maxima) > 1:
            mcounter += 1
            continue
    ipd_frac_multipeak = mcounter / len(cube)
    if show:
        _ut.create_interactive_psf_plot(cube)
    return ipd_frac_multipeak


def get_chi_phi(
    observed_cube: _ut.PSFData, calibrated_psf: _ut.PSFData, errors: _xt.Array = None, ddof: int = 0
) -> tuple[_xt.Array, _xt.Array]:
    """
    Calculate the chi-squared and phi values for each PSF in the observed cube.
    """
    chi_arr = []
    phi_arr = []
    exp = calibrated_psf.psf_2d
    for i in range(len(observed_cube)):
        psf_i = observed_cube[i].psf_2d
        phi_i = observed_cube[i].phi
        chi_i = _reduced_chi_squared(psf_i, exp, errors=errors, ddof=ddof)
        chi_arr.append(chi_i)
        phi_arr.append(phi_i)
    with _xp.NumpyContext() as np:
        chiphi = np.array(list(zip(chi_arr, phi_arr)))
        chiphi = chiphi[np.argsort(chiphi[:,1])]
    return chiphi


def _reduced_chi_squared(
    observed: _xt.Array, expected: _xt.Array, errors: _xt.Optional[_xt.Array] = None, ddof: int = 0
) -> float:
    """
    Calculate the reduced chi-squared statistic.
    """
    with _xp.NumpyContext() as np:
        if errors is None:
            errors = np.ones_like(observed)
        chi_squared = np.sum(((observed - expected) / errors) ** 2)
    return chi_squared / (len(observed) - ddof)


def find_local_maxima(psf: _xt.Array | _ut.PSFData, which: str = 'al', threshold: float = 0.4, show: bool = False) -> _xt.Array:
    """
    Find local maxima in a 2D PSF array above a certain threshold.

    Parameters
    ----------
    psf : _xt.Array or _ut.PSFData
        The input PSF data. If a _ut.PSFData object is provided, the AL PSF array
        will be extracted from it.
    which : str
        Which PSF to analyze: 'al' for along-scan, 'ac' for across-scan, 'both' for both.
    threshold : float
        The threshold above which to consider a local maximum usefull. It is given as a fraction
        of the maximum value of the PSF. Defaults to 0.4 (40% of the maximum).
    show : bool
        Whether to show a plot of the PSF and its local maxima.

    Returns
    -------
    _xt.Array
        An array of (x, value), where `x` is the pixel coordinate of the maxima and `value` its value.
    """
    if isinstance(psf, _ut.PSFData):
        if which == 'al':
            psf = psf.psf_x
        elif which == 'ac':
            psf = psf.psf_y
        elif which == 'both':
            maxima_al = find_local_maxima(psf, which='al', show=show)
            maxima_ac = find_local_maxima(psf, which='ac', show=show)
            return maxima_al, maxima_ac
        else:
            raise ValueError("Parameter 'which' must be one of 'al', 'ac', or 'both'.")
    elif psf.ndim != 1:
        raise ValueError("Input psf must be a 1D array or a _ut.PSFData object.")
    # Now psf is guaranteed to be a 1D array
    idx_f = psf.shape[0]
    maxima = []
    for ix in range(1,idx_f-1):
        p_i = psf[ix]
        p_i_1 = psf[ix-1]
        p_i_2 = psf[ix+1]
        if (p_i > p_i_1 or p_i == p_i_1) and p_i > p_i_2:
            if p_i > threshold * psf.max():
                maxima.append((ix, p_i))
            else: continue
    if show:
        _plt.figure()
        _plt.plot(psf, label='PSF')
        _plt.scatter(*zip(*maxima), color='red', label='Maxima')
        _plt.legend()
        _plt.title('Local Maxima in PSF')
        _plt.xlabel('Pixel')
        _plt.ylabel('Intensity')
        _plt.grid()
    return maxima



def upsample(img: _xt.Array, s: int = 4, order: str = "cubic") -> _xt.Array:
    """
    Upsample an image by a factor of `s` using skimage's resize function.

    Parameters
    ----------
    img : _xt.Array
        The input image to be upsampled. Can be a cube, with the last dimension
        representing the image id.
    s : int
        The upsampling factor.
    order : str
        The interpolation order to use:
        - 'nearest': Nearest-neighbor interpolation.
        - 'linear': Bilinear interpolation.
        - 'quadratic': Bi-quadratic interpolation.
        - 'cubic': Bicubic interpolation.
        - 'quartic': Bi-quartic interpolation.
        - 'quintic': Bi-quintic interpolation.
    """
    if order not in ["nearest", "linear", "quadratic", "cubic", "quartic", "quintic"]:
        raise ValueError(
            "Invalid order. Must be one of 'nearest', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic'."
        )
    order_map = {
        "nearest": 0,
        "linear": 1,
        "quadratic": 2,
        "cubic": 3,
        "quartic": 4,
        "quintic": 5,
    }
    out_shape = (
        (img.shape[0] * s, img.shape[1] * s)
        if img.ndim == 2
        else (img.shape[0] * s, img.shape[1] * s, img.shape[2])
    )
    y = resize(
        img, out_shape, order=order_map[order], anti_aliasing=False, preserve_range=True
    )
    return y.astype(img.dtype)

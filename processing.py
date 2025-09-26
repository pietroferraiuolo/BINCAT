import xupy as _xp
from xupy import typings as _xt
import scipy.signal as _ss
from astropy.convolution import convolve_fft as _as_convolve_fft
from skimage.transform import resize


def convolve_fft(
    image: _xt.Array,
    kernel: _xt.Array,
    dtype: _xt.Optional[_xt.DTypeLike] = _xp.float,
    normalize_kernel: bool = True,
    **kwargs: dict[str, _xt.Any],
) -> _xt.Array:
    """
    Convolve an image with a kernel using FFT on GPU.

    Parameters
    ----------
    image : xp.ndarray
        Input image to be convolved.
    kernel : xp.ndarray
        Convolution kernel.
    dtype : xp.dtype
        Data type for computation.
    normalize_kernel : bool or callable, optional
        If specified, this is the function to divide kernel by to normalize it.
        e.g., ``normalize_kernel=np.sum`` means that kernel will be modified to be:
        ``kernel = kernel / np.sum(kernel)``.  If True, defaults to
        ``normalize_kernel = np.sum``.

    Returns
    -------
    numpy.ndarray
        The convolved image (on CPU).

    Raises
    ------
    RuntimeError
        If GPU is not available.
    """
    try:
        if _xp.on_gpu:
            MAX_NORMALIZATION = 100
            normalization_zero_tol = 1e-8
            complex_dtype = _xp.cfloat if dtype == _xp.float else _xp.cdouble
            if normalize_kernel is True:
                if kernel.sum() < 1.0 / MAX_NORMALIZATION:
                    raise RuntimeError(
                        "The kernel can't be normalized, because its sum is close "
                        "to zero. The sum of the given kernel is < "
                        f"{1.0 / MAX_NORMALIZATION:.2f}. For a zero-sum kernel, set "
                        "normalize_kernel=False or pass a custom normalization "
                        "function to normalize_kernel."
                    )
                kernel_scale = _xp.sum(kernel)
                normalized_kernel = kernel / kernel_scale
                kernel_scale = 1.0
            elif normalize_kernel:
                kernel_scale = normalize_kernel(kernel)
                normalized_kernel = kernel / kernel_scale
            else:
                kernel_scale = kernel.sum()
                if _xp.abs(kernel_scale) < normalization_zero_tol:
                    kernel_scale = 1.0
                    normalized_kernel = kernel
                else:
                    normalized_kernel = kernel / kernel_scale
            img_g = _xp.asarray(image, dtype=complex_dtype)
            psf_g = _xp.asarray(normalized_kernel, dtype=complex_dtype)
            img1 = _xp.fft.fftn(img_g)
            psffft = _xp.fft.fftn(_xp.fft.ifftshift(psf_g))
            fftmult = img1 * psffft
            fftmult *= kernel_scale
            convolved = (_xp.fft.ifftn(fftmult).real).get()
        else:
            convolved = _as_convolve_fft(
                image, kernel, normalize_kernel=normalize_kernel, **kwargs
            )
    # for extra safety
    except Exception as e:
        print("Falling back to CPU convolution due to:", e)
        convolved = _as_convolve_fft(
            image, kernel, normalize_kernel=normalize_kernel, **kwargs
        )
    return convolved


def reduced_chi_squared(
    observed: _xt.Array, expected: _xt.Array, errors: _xt.Array = None, ddof: int = 0
) -> float:
    """
    Calculate the reduced chi-squared statistic.
    """
    if errors is None:
        errors = _xp.np.ones_like(observed)
    chi_squared = _xp.np.sum(((observed - expected) / errors) ** 2)
    return chi_squared / (len(observed) - ddof)


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

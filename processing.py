import xupy as _xp
from xupy import typings as _xt
import scipy.signal as _ss
from astropy.convolution import convolve_fft as _as_convolve_fft


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
            convolved = _as_convolve_fft(image, kernel, normalize_kernel=normalize_kernel, **kwargs)
    # for extra safety
    except Exception as e:
        print("Falling back to CPU convolution due to:", e)
        convolved = _as_convolve_fft(image, kernel, normalize_kernel=normalize_kernel, **kwargs)
    return convolved


def ipd_gof(): ...

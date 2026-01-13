import dataclasses as _dc
import xupy as _xp
from astropy.io import fits
from typing import Any as _Any
from numpy.typing import ArrayLike as _Array
import xupy.typings as _xt

@_dc.dataclass(init=True, frozen=True, repr=False)
class PSFData:
    psf: list[_Array] | fits.HDUList | str

    def __post_init__(self):
        if isinstance(self.psf, list):
            object.__setattr__(self, "_psf_2d", self.psf[0])
            object.__setattr__(self, "_psf_x", self.psf[1])
            object.__setattr__(self, "_psf_y", self.psf[2])
            object.__setattr__(self, "_psf_hr", self.psf[3])
            object.__setattr__(self, "_meta", None)
            object.__setattr__(self, "_shape", self._psf_2d.shape)
        elif isinstance(self.psf, fits.HDUList):
            object.__setattr__(self, "_psf_2d", self.psf[0].data)
            object.__setattr__(self, "_psf_x", self.psf[1].data)
            object.__setattr__(self, "_psf_y", self.psf[2].data)
            object.__setattr__(self, "_psf_hr", self.psf[3].data)
            object.__setattr__(self, "_meta", self.psf[0].header)
            object.__setattr__(self, "_shape", self._psf_2d.shape)
        elif isinstance(self.psf, str):
            with fits.open(self.psf) as hdul:
                object.__setattr__(self, "_psf_2d", hdul[0].data)
                object.__setattr__(self, "_psf_x", hdul[1].data)
                object.__setattr__(self, "_psf_y", hdul[2].data)
                object.__setattr__(self, "_psf_hr", hdul[3].data)
                object.__setattr__(
                    self,
                    "_meta",
                    {
                        0: hdul[0].header,
                        1: hdul[1].header,
                        2: hdul[2].header,
                        3: hdul[3].header,
                    },
                )
                object.__setattr__(self, "_shape", self._psf_2d.shape)
        else:
            raise TypeError(
                "PSF must be a list, fits.HDUList, or a fits file path string."
            )

    @property
    def psf_2d(self) -> _Array:
        return self._psf_2d

    @property
    def psf_x(self) -> _Array:
        return self._psf_x

    @property
    def psf_y(self) -> _Array:
        return self._psf_y

    @property
    def meta(self) -> dict[str, _Any]:
        if self._meta is None:
            raise TypeError(
                "For header to be returned PSF must be a fits.HDUList or a fits file path string."
            )
        return self._meta

    @property
    def shape(self):
        return self._shape

    @property
    def psf_hr(self):
        return self._psf_hr
    
    @property
    def phi(self):
        try:
            return self.meta[0]['PHI']
        except (KeyError, TypeError):
            raise KeyError("PHI not found in metadata.")

    def __repr__(self):
        try:
            phi = self.meta[0]["PHI"]
            arg = f"φ={phi:.3f}°"
        except (KeyError, TypeError):
            G = self.meta[0]["GMAG"]
            arg = f"calibration, G={G:.3f}"
        return f"PSFData({arg})"


def display_psf(
    psf: _xt.ArrayLike,
    mode: str = "all",
    save: str | bool = False,
    **kwargs: dict[str, _xt.Any],
) -> None:
    """
    Display the input PSF.

    Parameters
    ----------
    psf : xp.ndarray | list of xp.ndarray, optional
        The PSF to be displayed. It can be a 2D array or a list containing
        [psf_2d, psf_x, psf_y].
    mode : str, optional
        The mode of display. Options are:
        - '2d' for 2D display 
        - 'x' or 'y' for relative axes PSFs.
        - 'all' for combined display of 2D and 1D profiles.  
        
        Default is 'all'.
    save : str | bool, optional
        If provided, the figure will be saved, in the specified location if a string.
        
        Default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the display function (`plt.imshow`).
    """
    import os as _os
    import matplotlib.pyplot as _plt
    from astropy.visualization import (
        ImageNormalize,
        MinMaxInterval,
        LogStretch,
    )
    
    _plt.ion()

    try:
        if isinstance(psf, (list, tuple)) and len(psf) == 3:
            psf_x = psf[1]
            psf_y = psf[2]
            psf = psf[0]
        elif isinstance(psf, _xt.ArrayLike):
            psf_x, psf_y = computeXandYpsf(psf=psf)
            if psf_x.shape[0] != psf.shape[1] and psf_y.shape[0] != psf.shape[0]:
                raise ValueError("Something's wrong with the passed PSF")
    except Exception as e:
        raise (e)
    norm = ImageNormalize(
        vmin=_xp.np.nanmin(psf),
        vmax=_xp.np.nanmax(psf),
        stretch=LogStretch(500),
        interval=MinMaxInterval(),
    )
    normal = kwargs.pop("norm", norm)
    figsize = kwargs.pop('figsize', None)
    if mode == "all":
        fz = (8,4) if figsize is None else figsize
        fig = _plt.figure(figsize=fz)

        # Left: imshow (spans full height, 1/3 width)
        cmap = kwargs.pop("cmap", "gist_heat")
        aspect = kwargs.pop("aspect", "auto")
        extent = kwargs.pop(
            "extent",
            (
                -psf.shape[0] // 2,
                psf.shape[0] // 2,
                -psf.shape[1] // 2,
                psf.shape[1] // 2,
            ),
        )
        origin = kwargs.pop("origin", "lower")

        ax1 = _plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
        ax1.imshow(
            psf, cmap=cmap, aspect=aspect, extent=extent, origin=origin, norm=norm, **kwargs
        )
        ax1.axis("off")  # to hide axes

        # AL Right top: first plot (top half of right, 2/3 width)
        ax2 = _plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
        ax2.plot(psf_x, linewidth=2, color="tab:red")
        
        if not psf_x.shape[0] > 20:
            ax2.set_xticks(
                _xp.np.arange(0, int(psf_x.shape[0] + 1)),
                labels=[]
            )
            ax2.set_yticks(
                _xp.np.linspace(0, psf_x.max(), 4)
            )
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")
        ax2.set_xlim(0, psf_x.shape[0]-1)
        ax2.grid(True, linestyle="--", alpha=0.85)

        # AC Right bottom: second plot (bottom half of right, 2/3 width)
        ax3 = _plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
        ax3.plot(psf_y, linewidth=2, color="tab:red")
        
        if not psf_y.shape[0] > 20:
            ax3.set_xticks(
                _xp.np.arange(0, int(psf_y.shape[0] + 1)),
                labels=[]
            )
            ax3.set_yticks(
                _xp.np.linspace(0, psf_y.max(), 4)
            )
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        ax3.set_xlim(0, psf_y.shape[0]-1)
        ax3.grid(True, linestyle="--", alpha=0.85)

        fig.suptitle(
            "2D PSF" + " " * 25 + "1D Profiles: AL (up) | AC (down)",
            fontsize=14,
            weight="semibold",
        )
        _plt.tight_layout()
        _plt.show()
    elif mode == "2d":
        title = kwargs.pop('title', 'PSF')
        cmap = kwargs.pop("cmap", "gist_heat")
        extent = kwargs.pop(
            "extent",
            (
                -psf.shape[1] // 2,
                psf.shape[1] // 2,
                -psf.shape[0] // 2,
                psf.shape[0] // 2,
            ),
        )
        aspect = kwargs.pop("aspect", "auto")
        origin = kwargs.pop("origin", "lower")

        fig = _plt.figure(figsize=figsize)
        _plt.imshow(
            psf,
            origin=origin,
            cmap=cmap,
            norm=normal,
            extent=extent,
            aspect=aspect,
            **kwargs,
        )
        _plt.colorbar()
        _plt.title(title, fontdict={'size':14, 'weight':'semibold'})
        _plt.xlabel("AL [mas]")
        _plt.ylabel("AC [mas]")
    else:
        fig = _plt.figure(figsize=figsize)
        _plt.ylabel("Normalized PSF")
        _plt.grid(linestyle="--")
        y = _xp.np.arange(len(psf_y)) - len(psf_y) // 2
        x = _xp.np.arange(len(psf_x)) - len(psf_x) // 2
        _plt.title(f"PSF in {mode} direction")
        if mode == "x":
            _plt.plot(x, psf_x)
            _plt.xlabel("AL [px]")
            _plt.xlim(x.min(), x.max())
            _plt.xticks(_xp.np.arange(int(x.min()), int(x.max()) + 1))
        elif mode == "y":
            _plt.plot(y, psf_y)
            _plt.xlabel("AC [px]")
            _plt.xlim(y.min(), y.max())
            _plt.xticks(_xp.np.arange(int(y.min()), int(y.max()) + 1))
        else:
            raise ValueError("Invalid mode. Use `all`, '2d', 'x', or 'y'.")
    _plt.show()
    if save:
        if isinstance(save, str):
            save, ext = _os.path.splitext(save)
            if ext == '':
                ext = '.svg'
        else:
            save, ext = "psf", ".svg"
        fig.savefig(f"{save}{ext}", transparent=True, dpi=450)
    return fig

def computeXandYpsf(
    psf: _xt.Optional[_xt.ArrayLike] = None,
) -> None | tuple[_xt.ArrayLike, _xt.ArrayLike]:
    """
    Subroutine to compute the normalized psf in the X and Y axis of the
    2D PSF.
    """
    img = psf.copy()
    psf_x = _xp.sum(img, axis=0)
    psf_x /= _xp.sum(psf_x)  # normalize
    psf_y = _xp.sum(img, axis=1)
    psf_y /= _xp.sum(psf_y)
    return psf_x, psf_y

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
        If kernel can't be normalized.
    """
    from astropy.convolution import convolve_fft as _as_convolve_fft    

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

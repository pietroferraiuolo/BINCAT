import os as _os
import matplotlib.pyplot as _plt
from astropy.visualization import (
    ImageNormalize,
    MinMaxInterval,
    LogStretch,
)
from collections import OrderedDict as _OrderedDict
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from threading import Lock as _Lock
import xupy as _xp
from astropy.io import fits
from typing import Any as _Any, Literal as _Literal
from numpy.typing import ArrayLike as _Array
import yaml as _yaml
import xupy.typings as _xt
from ..core.root import SIMULATION_PARAMETERS_PATH as _spp
from . import osutils as _osu


class PSFData:

    def __init__(self, psf: list[_Array] | fits.HDUList | str, tn: str | None = None):
        """Container for PSF data and metadata."""
        self.psf = psf
        self.tn = tn

        if isinstance(self.psf, list):
            if len(self.psf) < 4:
                raise ValueError(
                    "When `psf` is a list, it must contain "
                    "[psf_2d, psf_x, psf_y, psf_hr]."
                )
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
            object.__setattr__(
                self,
                "_meta",
                {
                    0: self.psf[0].header,
                    1: self.psf[1].header,
                    2: self.psf[2].header,
                    3: self.psf[3].header,
                },
            )
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

        try:
            if self.tn is None:
                raise ValueError("Missing tracking number.")
            with open(_spp(self.tn), "r", encoding="utf-8") as f:
                simparams = _yaml.safe_load(f)
            object.__setattr__(self, "_simparams", simparams)
        except Exception:
            object.__setattr__(self, "_simparams", None)
    
    @property
    def simparams(self) -> dict[str, _Any]:
        return self._simparams

    @property
    def psf_2d(self) -> _Array:
        return self._psf_2d

    @property
    def psf_al(self) -> _Array:
        return self._psf_x

    @property
    def psf_ac(self) -> _Array:
        return self._psf_y
    
    @property
    def shape_al(self):
        return self.psf_al.shape
    
    @property
    def shape_ac(self):
        return self.psf_ac.shape

    @property
    def meta(self) -> dict[str, _Any]:
        if self._meta is None:
            raise TypeError(
                "For header to be returned PSF must be a fits.HDUList or a fits file path string."
            )
        return self._meta

    @property
    def primary_meta(self) -> dict[str, _Any]:
        return self.meta[0]

    @property
    def shape(self):
        return self._shape

    @property
    def psf_hr(self):
        return self._psf_hr

    @property
    def phi(self):
        try:
            return self.primary_meta["PHI"]
        except (KeyError, TypeError):
            raise KeyError("PHI not found in metadata.")

    def __repr__(self):
        try:
            phi = self.primary_meta["PHI"]
            arg = f"φ={phi:.3f}°"
        except (KeyError, TypeError):
            G = self.primary_meta["GMAG"]
            arg = f"calibration, G={G:.3f}"
        return f"PSFData({arg})"
    
    def __add__(self, other):
        if not isinstance(other, PSFData):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("PSF shapes must match for addition.")
        combined_psf_2d = self.psf_2d + other.psf_2d
        combined_psf_x = self.psf_al + other.psf_al
        combined_psf_y = self.psf_ac + other.psf_ac
        combined_psf_hr = self.psf_hr + other.psf_hr
        newpsf = PSFData(
            psf=[combined_psf_2d, combined_psf_x, combined_psf_y, combined_psf_hr],
            tn=self.tn,
        )
        newpsf.meta = self.meta
        newpsf.primary_meta = self.primary_meta
        return newpsf
    
    def __sub__(self, other):
        if not isinstance(other, PSFData):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("PSF shapes must match for subtraction.")
        combined_psf_2d = self.psf_2d - other.psf_2d
        combined_psf_x = self.psf_al - other.psf_al
        combined_psf_y = self.psf_ac - other.psf_ac
        combined_psf_hr = self.psf_hr - other.psf_hr
        newpsf = PSFData(
            psf=[combined_psf_2d, combined_psf_x, combined_psf_y, combined_psf_hr],
            tn=self.tn,
        )
        newpsf._meta = self.meta
        newpsf._primary_meta = self.primary_meta
        return newpsf

class PSFCube:

    def __init__(
        self,
        tn: str,
        load_mode: _Literal["eager", "lazy"] = "eager",
        cache_size: int | None = 64,
        prefetch: int = 0,
    ):
        """
        Container for PSFs of a simulation tracking number.

        Parameters
        ----------
        tn : str
            Simulation tracking number.
        load_mode : {'eager', 'lazy'}, optional
            Loading strategy for PSF files.
            - 'eager': load all PSFs at construction time.
            - 'lazy': load PSFs on demand.
            Default is 'eager'.
        cache_size : int or None, optional
            Maximum number of lazily-loaded PSFs kept in memory.
            If None, the cache is unbounded.
            If 0, no cache is used.
            Default is 64.
        prefetch : int, optional
            Number of upcoming PSFs to prefetch when iterating.
            Only used in lazy mode.
            Default is 0.
        """
        if load_mode not in {"eager", "lazy"}:
            raise ValueError("`load_mode` must be either 'eager' or 'lazy'.")
        if cache_size is not None and cache_size < 0:
            raise ValueError("`cache_size` must be >= 0 or None.")
        if prefetch < 0:
            raise ValueError("`prefetch` must be >= 0.")
        if load_mode == "lazy" and prefetch > 0 and cache_size == 0:
            raise ValueError("`prefetch` requires `cache_size` > 0 or None.")

        self.tn = tn
        self.load_mode = load_mode
        self.cache_size = cache_size
        self.prefetch = prefetch

        psflist = _osu.getFileList(tn, fold='observations', key=r"0")
        self.calibration = _osu.load_psf_calibration(tn)

        self._cache_lock = _Lock()
        self._cache: _OrderedDict[int, PSFData] = _OrderedDict()
        self._prefetch_executor: _ThreadPoolExecutor | None = None
        self._prefetch_futures: dict[int, _Any] = {}

        self._init_data(psflist)
        
    @property
    def shape(self):
        if len(self) == 0:
            return (0, 0, 0)
        first_psf = self._get_psf(0)
        return (len(self),) + first_psf.shape
    
    @property
    def shape_al(self):
        if len(self) == 0:
            return (0,)
        first_psf = self._get_psf(0)
        return first_psf.shape_al

    def _init_data(self, psflist: list[str]):
        """
        Initializes the PSF cube data by loading the calibration and PSF files.

        Parameters
        ----------
        psflist : list of str
            A list of file paths to the PSF FITS files.
        calib_file : str
            The file path to the calibration FITS file.

        Returns
        -------
        None
        """
        self._psf_files = list(psflist)
        if self.load_mode == "eager":
            self._psfs: list[PSFData | None] = [
                PSFData(psf_file, tn=None) for psf_file in self._psf_files
            ]
        else:
            self._psfs = [None] * len(self._psf_files)

        self._simpar = self.calibration.simparams

    def _build_psf(self, index: int) -> PSFData:
        return PSFData(self._psf_files[index], tn=None)

    def _cache_get(self, index: int) -> PSFData | None:
        if self.cache_size == 0:
            return None
        with self._cache_lock:
            psf = self._cache.get(index)
            if psf is not None:
                self._cache.move_to_end(index)
            return psf

    def _cache_set(self, index: int, psf: PSFData) -> None:
        if self.cache_size == 0:
            return
        with self._cache_lock:
            self._cache[index] = psf
            self._cache.move_to_end(index)
            if self.cache_size is not None:
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

    def _ensure_prefetch_executor(self) -> None:
        if self._prefetch_executor is None:
            # One worker is enough for overlapping network I/O with compute.
            self._prefetch_executor = _ThreadPoolExecutor(max_workers=1)

    def _schedule_prefetch(self, start_index: int) -> None:
        if self.load_mode != "lazy" or self.prefetch <= 0:
            return
        self._ensure_prefetch_executor()
        upper = min(len(self._psf_files), start_index + self.prefetch)
        for index in range(start_index, upper):
            if self._cache_get(index) is not None:
                continue
            if index in self._prefetch_futures:
                continue
            self._prefetch_futures[index] = self._prefetch_executor.submit(
                self._build_psf, index
            )

    def _get_psf(self, index: int) -> PSFData:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("PSFCube index out of range.")

        if self.load_mode == "eager":
            psf = self._psfs[index]
            if psf is None:
                raise RuntimeError("Eager PSF cache is corrupted.")
            return psf

        cached = self._cache_get(index)
        if cached is not None:
            return cached

        prefetched = self._prefetch_futures.pop(index, None)
        if prefetched is not None:
            psf = prefetched.result()
        else:
            psf = self._build_psf(index)

        self._cache_set(index, psf)
        return psf
    
    @property
    def simparams(self) -> dict[str, _Any]:
        return self._simpar

    @property
    def psfs(self) -> list[PSFData]:
        return [self._get_psf(i) for i in range(len(self))]

    @property
    def files(self) -> list[str]:
        return self._psf_files.copy()

    @property
    def cache_info(self) -> dict[str, _Any]:
        return {
            "mode": self.load_mode,
            "cache_size": self.cache_size,
            "cached_items": len(self._cache),
            "prefetch": self.prefetch,
            "pending_prefetch": len(self._prefetch_futures),
        }

    def __iter__(self):
        for index in range(len(self)):
            self._schedule_prefetch(index + 1)
            yield self._get_psf(index)

    def __len__(self):
        return len(self._psf_files)
    
    def __repr__(self):
        return (
            f"PSFCube(n_psfs={len(self)}, tn='{self.tn}', "
            f"mode='{self.load_mode}', cache_size={self.cache_size}, "
            f"prefetch={self.prefetch})"
        )
    
    def __str__(self):
        return self.__repr__()
    
    def __add__(self, other):
        if not isinstance(other, PSFCube):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("PSFCubes must have the same number of PSFs for addition.")
        combined_psfs = [self[i] + other[i] for i in range(len(self))]
        combined_cube = self.__copy__()
        combined_cube._psfs = combined_psfs
        return combined_cube
    
    def __sub__(self, other):
        if not isinstance(other, PSFCube):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("PSFCubes must have the same number of PSFs for subtraction.")
        combined_psfs = [self[i] - other[i] for i in range(len(self))]
        combined_cube = self.__copy__()
        combined_cube._psfs = combined_psfs
        return combined_cube
    
    def __copy__(self):
        new_cube = PSFCube(
            tn=self.tn,
            load_mode=self.load_mode,
            cache_size=self.cache_size,
            prefetch=self.prefetch,
        )
        new_cube._psf_files = self._psf_files.copy()
        if self.load_mode == "eager":
            new_cube._psfs = self._psfs.copy()
        else:
            new_cube._psfs = [None] * len(self._psf_files)
        return new_cube
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            rng = range(*index.indices(len(self)))
            return [self._get_psf(i) for i in rng]
        return self._get_psf(index)

    def close(self) -> None:
        for future in self._prefetch_futures.values():
            future.cancel()
        self._prefetch_futures.clear()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False)
            self._prefetch_executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def display_psf(
    psf: _xt.Array,
    mode: str = "all",
    save: str | bool = False,
    **kwargs: dict[str, _xt.Any], # type: ignore
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
    try:
        if isinstance(psf, PSFData):
            psf_x = psf.psf_al
            psf_y = psf.psf_ac
            psf = psf.psf_2d
        elif isinstance(psf, (list, tuple)) and len(psf) == 3:
            psf_x = psf[1]
            psf_y = psf[2]
            psf = psf[0]
        elif isinstance(psf, _Array):
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
    figsize = kwargs.pop("figsize", None)

    fig = _plt.figure(figsize=(8,5) if figsize is None else figsize)

    if mode == "all":
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
            psf,
            cmap=cmap,
            aspect=aspect,
            extent=extent,
            origin=origin,
            norm=norm,
            **kwargs,
        )
        ax1.axis("off")  # to hide axes
        ax1.set_title("PSF\n", fontdict={"size": 14, "weight": "semibold"})

        # AL Right top: first plot (top half of right, 2/3 width)
        ax2 = _plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
        ax2.plot(psf_x, linewidth=2, color="tab:red")

        if not psf_x.shape[0] > 20:
            ax2.set_xticks(_xp.np.arange(0, int(psf_x.shape[0] + 1)), labels=[])
            ax2.set_yticks(_xp.np.linspace(0, psf_x.max(), 4))
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")
        ax2.set_xlim(0, psf_x.shape[0] - 1)
        ax2.grid(True, linestyle="--", alpha=0.85)
        ax2.set_title("LSFs : AL (up) | AC (down)\n", fontdict={"size": 14, "weight": "semibold"})

        # AC Right bottom: second plot (bottom half of right, 2/3 width)
        ax3 = _plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
        ax3.plot(psf_y, linewidth=2, color="tab:red")

        if not psf_y.shape[0] > 20:
            ax3.set_xticks(_xp.np.arange(0, int(psf_y.shape[0] + 1)), labels=[])
            ax3.set_yticks(_xp.np.linspace(0, psf_y.max(), 4))
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        ax3.set_xlim(0, psf_y.shape[0] - 1)
        ax3.grid(True, linestyle="--", alpha=0.85)

        _plt.tight_layout()

    elif mode == "2d":
        title = kwargs.pop("title", "PSF")
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
        _plt.title(title, fontdict={"size": 14, "weight": "semibold"})
        _plt.xlabel("AL [mas]")
        _plt.ylabel("AC [mas]")

    else:

        _plt.ylabel("Normalized PSF")
        _plt.grid(linestyle="--")
        y = _xp.np.arange(len(psf_y)) - len(psf_y) // 2
        x = _xp.np.arange(len(psf_x)) - len(psf_x) // 2
        _plt.title(f"PSF in {mode} direction")
        if mode == "x":
            _plt.plot(x, psf_x)
            _plt.xlabel("AL [px]")
            _plt.xlim(x.min(), x.max())
            if not len(psf_x) > 20:
                _plt.xticks(_xp.np.arange(int(x.min()), int(x.max()) + 1))
        elif mode == "y":
            _plt.plot(y, psf_y)
            _plt.xlabel("AC [px]")
            _plt.xlim(y.min(), y.max())
            if not len(psf_y) > 20:
                _plt.xticks(_xp.np.arange(int(y.min()), int(y.max()) + 1))
        else:
            raise ValueError("Invalid mode. Use `all`, '2d', 'x', or 'y'.")

    _plt.show()

    if save:
        if isinstance(save, str):
            save, ext = _os.path.splitext(save)
            if ext == "":
                ext = ".svg"
        else:
            save, ext = "psf", ".svg"
        fig.savefig(f"{save}{ext}", transparent=True, dpi=450)

    return fig

def computeXandYpsf(
    psf: _xt.Array | None = None,
) -> None | tuple[_xt.Array, _xt.Array]:
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
    dtype = _xp.float,
    normalize_kernel: bool = True,
    **kwargs: dict[str, _xt.Any], # type: ignore
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


def benchmark_psfcube_loading(
    tn: str,
    configs: list[dict[str, _Any]] | None = None,
    repeats: int = 1,
    max_items: int | None = None,
    passes: int = 1,
) -> list[dict[str, _Any]]:
    """
    Benchmark PSFCube loading/iteration strategies for a tracking number.

    Parameters
    ----------
    tn : str
        Simulation tracking number to benchmark.
    configs : list of dict, optional
        List of PSFCube keyword argument dictionaries.
        If None, a default set of common strategies is used.
    repeats : int, optional
        Number of repeated runs per configuration. Default is 1.
    max_items : int or None, optional
        Maximum number of PSFs to touch per pass.
        If None, iterate all PSFs. Default is None.
    passes : int, optional
        Number of complete passes over the selected PSFs per run.
        Default is 1.

    Returns
    -------
    results : list of dict
        Benchmark results sorted by mean total time.
        Each dict contains configuration and timing statistics in seconds.
    """
    import time as _time

    if repeats < 1:
        raise ValueError("`repeats` must be >= 1.")
    if passes < 1:
        raise ValueError("`passes` must be >= 1.")
    if max_items is not None and max_items < 1:
        raise ValueError("`max_items` must be >= 1 when provided.")

    if configs is None:
        configs = [
            {"load_mode": "eager"},
            {"load_mode": "lazy", "cache_size": 0, "prefetch": 0},
            {"load_mode": "lazy", "cache_size": 64, "prefetch": 0},
            {"load_mode": "lazy", "cache_size": 128, "prefetch": 8},
            {"load_mode": "lazy", "cache_size": 256, "prefetch": 16},
        ]

    n_psfs_total = len(_osu.getFileList(tn, key="psf"))

    results: list[dict[str, _Any]] = []
    for cfg in configs:
        run_init: list[float] = []
        run_iter: list[float] = []
        run_total: list[float] = []

        for _ in range(repeats):
            t0 = _time.perf_counter()
            cube = PSFCube(tn=tn, **cfg)
            t1 = _time.perf_counter()

            n_items = len(cube) if max_items is None else min(len(cube), max_items)
            for _ in range(passes):
                for i, psf in enumerate(cube):
                    if i >= n_items:
                        break
                    # Touch data to avoid the loop being optimized away.
                    _ = psf.psf_2d.shape

            t2 = _time.perf_counter()
            cube.close()

            init_s = t1 - t0
            iter_s = t2 - t1
            total_s = t2 - t0
            run_init.append(init_s)
            run_iter.append(iter_s)
            run_total.append(total_s)

        result = {
            "config": cfg,
            "repeats": repeats,
            "passes": passes,
            "max_items": max_items,
            "n_psfs_total": n_psfs_total,
            "init_s_mean": float(_xp.np.mean(run_init)),
            "iter_s_mean": float(_xp.np.mean(run_iter)),
            "total_s_mean": float(_xp.np.mean(run_total)),
            "init_s_std": float(_xp.np.std(run_init)),
            "iter_s_std": float(_xp.np.std(run_iter)),
            "total_s_std": float(_xp.np.std(run_total)),
        }
        results.append(result)

    results.sort(key=lambda r: r["total_s_mean"])
    return results


__all__ = [
    "PSFData",
    "PSFCube",
    "benchmark_psfcube_loading",
    "computeXandYpsf",
    "convolve_fft",
    "display_psf",
]

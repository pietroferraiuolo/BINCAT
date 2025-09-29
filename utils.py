import time as _time
import xupy as _xp, os as _os
from xupy import typings as _xt
from astropy.io import fits
from numpy.typing import ArrayLike as _Array
from numpy.ma import MaskedArray as _masked_array
from numpy import uint8 as _uint8
from typing import Any as _Any

basepath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
simpath = f"{basepath}/simulations"

__all__ = [
    "load_fits",
    "save_fits",
    "get_kwargs",
    "newtn",
    "Logger",
    "fits",
    'convolve_fft'
]

##############
## OS UTILS ##
##############


def getFileList(tn: str = None, fold: str = None, key: str = None) -> list[str]:
    """
    Search for files in a given tracking number or complete path, sorts them and
    puts them into a list.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    fold : str, optional
        Folder in which searching for the tracking number. If None, the default
        folder is the OPD_IMAGES_ROOT_FOLDER.
    key : str, optional
        A key which identify specific files to return
    """
    if tn is None and fold is not None:
        fl = sorted([_os.path.join(fold, file) for file in _os.listdir(fold)])
    else:
        try:
            paths = _findTracknum(tn, complete_path=True)
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if fold is None:
                    fl = []
                    fl.append(
                        sorted(
                            [_os.path.join(path, file) for file in _os.listdir(path)]
                        )
                    )
                elif fold in path.split("/")[-2]:
                    fl = sorted(
                        [_os.path.join(path, file) for file in _os.listdir(path)]
                    )
                else:
                    continue
        except Exception as exc:
            raise FileNotFoundError(
                f"Invalid Path: no data found for tn '{tn}'"
            ) from exc
    if len(fl) == 1:
        fl = fl[0]
    if key is not None:
        try:
            selected_list = []
            for file in fl:
                if key in file.split("/")[-1]:
                    selected_list.append(file)
        except TypeError as err:
            raise TypeError("'key' argument must be a string") from err
        fl = selected_list
    if len(fl) == 1:
        fl = fl[0]
    return fl


def load_psf(filepath: str):
    """
    Loads a PSF from a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file containing the PSF data.

    Returns
    -------
    psf_data : list of np.ndarray
        A list containing the PSF data arrays in the order:
        [psf_2d, psf_x, psf_y, psf_hr].
    """
    return PSFData(psf=filepath)


def load_psf_cube(tn: str) -> list["PSFData"]:
    """
    Load a series of PSF FITS files into a list of PSFData objects.

    Parameters
    ----------
    tn : str
        Tracking number to identify the folder containing the PSF FITS files.
    """
    fl = getFileList(tn)
    __ = fl.pop(-1)
    psf_cube = []
    for f in fl:
        psf_cube.append(load_psf(f))
    return psf_cube


def load_psf_calibration(tn: str) -> "PSFData":
    """
    Load the PSF calibration FITS file into a PSFData object.

    Parameters
    ----------
    tn : str
        Tracking number to identify the folder containing the PSF calibration FITS file.
    """
    fl = getFileList(tn)
    f = fl.pop(-1)
    psf_cal = load_psf(f)
    return psf_cal


def load_fits(filepath: str, return_header: bool = False, as_masked_array: bool = True):
    """
    Loads a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    return_header: bool
        Wether to return the header of the loaded fits file. Default is False.

    Returns
    -------
    fit : np.ndarray or np.ma.MaskedArray
        FITS file data.
    header : dict | fits.Header, optional
        The header of the loaded fits file.
    """
    with fits.open(filepath) as hdul:
        if len(hdul) == 1:
            fit = hdul[0]
        elif len(hdul) == 2 and as_masked_array:
            fit = hdul[0].data
            if len(hdul) > 1 and hasattr(hdul[1], "data"):
                mask = hdul[1].data.astype(bool)
                fit = _masked_array(fit, mask=mask)
        elif len(hdul) >= 2:
            fit = []
            for hdu in hdul:
                if hasattr(hdu, "data"):
                    fit.append(hdu.data)
                else:
                    fit.append(None)
            fit = tuple(fit)
        if return_header:
            header = hdul[0].header
            return fit, header
    return fit


def save_fits(
    filepath: str,
    data: _Array,
    overwrite: bool = True,
    header: dict[str, _Any] | fits.Header = None,
) -> None:
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    data : np.array
        Data to be saved.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    header : dict[str, any] | fits.Header, optional
        Header information to include in the FITS file. Can be a dictionary or
        a fits.Header object.
    """
    # Prepare the header
    if header is not None:
        header = _header_from_dict(header)
    # Save the FITS file
    if isinstance(data, _masked_array):
        fits.writeto(filepath, data.data, header=header, overwrite=overwrite)
        if hasattr(data, "mask"):
            fits.append(filepath, data.mask.astype(_uint8))
    else:
        fits.writeto(filepath, data, header=header, overwrite=overwrite)


def _header_from_dict(dictheader: dict[str, _Any | tuple[_Any, str]]) -> fits.Header:
    """
    Converts a dictionary to an astropy.io.fits.Header object.

    Parameters
    ----------
    dictheader : dict
        Dictionary containing header information. Each key should be a string,
        and the value can be a tuple of length 2, where the first element is the
        value and the second is a comment.

    Returns
    -------
    header : astropy.io.fits.Header
        The converted FITS header object.
    """
    if isinstance(dictheader, fits.Header):
        return dictheader
    header = fits.Header()
    for key, value in dictheader.items():
        if isinstance(value, tuple) and len(value) > 2:
            raise ValueError(
                "Header values must be a tuple of length 2 or less, "
                "where the first element is the value and the second is the comment."
                f"{value}"
            )
        else:
            header[key] = value
    return header


def get_kwargs(names: tuple[str], default: _Any, kwargs: dict[str, _Any]) -> _Any:
    """
    Gets a tuple of possible kwargs names for a variable and checks if it was
    passed, and in case returns it.

    Parameters
    ----------
    names : tuple
        Tuple containing all the possible names of a variable which can be passed
        as a **kwargs argument.
    default : any type
        The default value to assign the requested key if it doesn't exist.
    kwargs : dict
        The dictionary of variables passed as 'Other Parameters'.

    Returns
    -------
    key : value of the key
        The value of the searched key if it exists. If not, the default value will
        be returned.
    """
    possible_keys = names
    for key in possible_keys:
        if key in kwargs:
            return kwargs[key]
    return default


def newtn() -> str:
    """
    Returns a timestamp in a string of the format `YYYYMMDD_HHMMSS`.

    Returns
    -------
    str
        Current time in a string format.
    """
    return _time.strftime("%Y%m%d_%H%M%S")


#######################
##  OTHER UTILITIES  ##
#######################

import functools


def timer(func: callable) -> callable:
    """Decorator to time the execution of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = _time.perf_counter()
        result = func(*args, **kwargs)
        end_time = _time.perf_counter()
        h = (end_time - start_time) // 3600
        m = (end_time - start_time) % 3600 // 60
        s = (end_time - start_time) % 60
        print(f"Execution time: {int(h):02d}:{int(m):02d}:{s:.2f} (h:m:s)")
        return result

    return wrapper


def _findTracknum(tn: str, complete_path: bool = False) -> str | list[str]:
    """
    Search for the tracking number given in input within all the data path subfolders.

    Parameters
    ----------
    tn : str
        Tracking number to be searched.
    complete_path : bool, optional
        Option for wheter to return the list of full paths to the folders which
        contain the tracking number or only their names.

    Returns
    -------
    tn_path : list of str
        List containing all the folders (within the OPTData path) in which the
        tracking number is present, sorted in alphabetical order.
    """
    tn_path = []
    for fold in _os.listdir(simpath):
        search_fold = _os.path.join(simpath, fold)
        if not _os.path.isdir(search_fold):
            continue
        if tn in _os.listdir(search_fold):
            if complete_path:
                tn_path.append(_os.path.join(search_fold, tn))
            else:
                tn_path.append(fold)
    path_list = sorted(tn_path)
    if len(path_list) == 1:
        path_list = path_list[0]
    return path_list


#######################
## LOGGING UTILITIES ##
#######################

import logging as _l
import logging.handlers as _lh


class Logger:

    def __init__(self, level: int = _l.INFO) -> None:
        """The constructor"""
        self._l = self._set_up_logger(logging_level=level)

    def _set_up_logger(self, logging_level: int = _l.INFO) -> _l.Logger:
        """
        Set up a rotating file logger.

        This function configures a logger to write log messages to a file with
        rotation. The log file will be encoded in UTF-8 and will rotate when it
        reaches a specified size, keeping a specified number of backup files.

        Parameters
        ----------
        filename : str
            The path to the log file where log messages will be written.
        logging_level : int
            The logging level to set for the logger. This should be one of the
            logging level constants defined in the ``logging`` module (e.g.,
            ``DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50``).

        Notes
        -----
        - The log file will rotate when it reaches 10,000,000 bytes (10 MB).
        - Up to 3 backup log files will be kept.
        - The log format includes the timestamp, log level, logger name, and message.
        - The logger is configured at the root level, affecting all loggers in the application.
        - The handler will perform an initial rollover when set up.

        Examples
        --------
        .. code-block:: python

            set_up_logger('/path/to/logfile.log', logging.DEBUG)
        """
        FORMAT = "[%(levelname)s] %(name)s - %(message)s"
        formato = _l.Formatter(fmt=FORMAT)
        handler = _lh.RotatingFileHandler(
            filename="data/bincat.log",
            mode="a",
            maxBytes=10_000_000,
            backupCount=3,
            encoding="utf-8",
            delay=0,
        )
        root_logger = _l.getLogger()
        root_logger.setLevel(logging_level)
        handler.setFormatter(formato)
        handler.setLevel(logging_level)
        root_logger.addHandler(handler)
        handler.doRollover()
        return root_logger

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message at the specified level.

        Parameters
        ----------
        message : str
            The message to log.
        level : str, optional
            The logging level to use for the message. This should be one of the
            following strings: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. (can
            use lowercase too).
            The default is 'DEBUG'.

        Notes
        -----
        - The message will be logged using the logger configured by `set_up_logger`.
        - The message will be logged with the specified level.
        - If the specified level is not recognized, the message will be logged at the
        'DEBUG' level.
        """
        level = level.upper()
        if level == "DEBUG":
            self._l.debug(message)
        elif level == "INFO":
            self._l.info(message)
        elif level == "WARNING":
            self._l.warning(message)
        elif level == "ERROR":
            self._l.error(message)
        elif level == "CRITICAL":
            self._l.critical(message)
        else:
            self._l.debug(message)
            self._l.warning(f"Invalid log level '{level}'. Defaulting to 'DEBUG'.")


#################
##  PSF UTILS  ##
#################

import matplotlib.pyplot as _plt
from astropy.visualization import (
    ImageNormalize,
    MinMaxInterval,
    LogStretch,
)


def display_psf(
    psf: _xt.Optional[_xt.ArrayLike] = None,
    mode: str = "all",
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
        _l.log(e, level="ERROR")
        raise (e)
    if mode == "all":

        fig = _plt.figure(figsize=(8, 4))

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
            psf, cmap=cmap, aspect=aspect, extent=extent, origin=origin, **kwargs
        )
        ax1.axis("off")  # to hide axes

        # Right top: first plot (top half of right, 2/3 width)
        ax2 = _plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
        ax2.plot(psf_x, linewidth=2, color="tab:red")
        ax2.set_xticks(
            _xp.np.arange(0, int(psf_x.shape[0] + 1)),
            labels=[]
        )
        ax2.set_yticks(
            _xp.np.linspace(0, psf_x.max(), 4)
        )
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")
        ax2.grid(True, linestyle="--", alpha=0.85)

        # Right bottom: second plot (bottom half of right, 2/3 width)
        ax3 = _plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
        ax3.plot(psf_y, linewidth=2, color="tab:red")
        ax3.set_xticks(
            _xp.np.arange(0, int(psf_y.shape[0] + 1)),
            labels=[]
        )
        ax3.set_yticks(
            _xp.np.linspace(0, psf_y.max(), 4)
        )
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        ax3.grid(True, linestyle="--", alpha=0.85)

        fig.suptitle(
            "2D PSF" + " " * 25 + "1D Profiles: AL (up) | AC (down)",
            fontsize=14,
            weight="semibold",
        )
        _plt.tight_layout()
        _plt.show()
    elif mode == "2d":
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

        norm = ImageNormalize(
            vmin=_xp.np.nanmin(psf),
            vmax=_xp.np.nanmax(psf),
            stretch=LogStretch(500),
            interval=MinMaxInterval(),
        )
        normal = kwargs.pop("norm", norm)
        fig = _plt.figure()
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
        _plt.title("CCD PSF")
        _plt.xlabel("AL [px]")
        _plt.ylabel("AC [px]")
    else:
        fig = _plt.figure()
        _plt.ylabel("Normalized PSF")
        _plt.grid(linestyle="--")
        y = _xp.np.arange(len(psf_y)) - len(psf_y) // 2
        x = _xp.np.arange(len(psf_x)) - len(psf_x) // 2
        _plt.title(f"PSF in {mode} direction")
        if mode == "x":
            _plt.plot(x, psf_x)
            _plt.xlabel("AL [px]")
            _plt.xticks(_xp.np.arange(int(x.min()), int(x.max()) + 1))
        elif mode == "y":
            _plt.plot(y, psf_y)
            _plt.xlabel("AC [px]")
            _plt.xticks(_xp.np.arange(int(y.min()), int(y.max()) + 1))
        else:
            raise ValueError("Invalid mode. Use `all`, '2d', 'x', or 'y'.")
    _plt.show()
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


import dataclasses as _dc


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

#########################
## COMPUTING UTILITIES ##
#########################

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

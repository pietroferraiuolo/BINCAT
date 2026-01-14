# Lets try this way
from typing import Any as _Any
from ..core import root as _fn
from .psfutils import PSFData as _PSFData
from opticalib.ground import osutils as _optosu

_optosu._OPTDATA = _fn.SIMPATH

newtn = _optosu.newtn


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


def getFileList(tn: str, fold: str | None = None, key: str | None = None) -> list[str]:
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
        A key which identify specific files to return.

    Returns
    -------
    file_list : list of str
        A sorted list containing the paths to the found files.
    """
    return _optosu.getFileList(tn, fold=fold, key=key)


def load_fits(filepath: str, on_gpu: bool = False) -> _Any:
    """
    Wrapper for opticalib.ground.osutils.load_fits function.

    Parameters
    ----------
    filepath : str
        The path to the FITS file to be loaded.
    on_gpu : bool, optional
        Whether to load the data onto the GPU (default is False).

    Returns
    -------
    data : any type
        The data loaded from the FITS file.
    """
    return _optosu.load_fits(filepath, on_gpu=on_gpu)


def save_fits(
    filepath: str,
    data: _Any,
    overwrite: bool = True,
    header: dict[str, _Any] = None,
) -> None:
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        The path where the FITS file will be saved.
    data : ImageData | CubeData | MatrixLike | ArrayLike | Any
        The data to be saved in the FITS file.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists (default is True).
    header : dict | Header, optional
        The header information to be included in the FITS file (default is None).
    """
    return _optosu.save_fits(filepath, data, overwrite=overwrite, header=header)


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
    return _PSFData(psf=filepath)


def load_psf_cube(tn: str) -> list["_PSFData"]:
    """
    Load a series of PSF FITS files into a list of PSFData objects.

    Parameters
    ----------
    tn : str
        Tracking number to identify the folder containing the PSF FITS files.
    """
    fl = _optosu.getFileList(tn)
    __ = fl.pop(-1)
    psf_cube = []
    for f in fl:
        psf_cube.append(load_psf(f))
    return psf_cube


def load_psf_calibration(tn: str) -> "_PSFData":
    """
    Load the PSF calibration FITS file into a PSFData object.

    Parameters
    ----------
    tn : str
        Tracking number to identify the folder containing the PSF calibration FITS file.
    """
    fl = _optosu.getFileList(tn)
    f = fl.pop(-1)
    del fl
    psf_cal = load_psf(f)
    return psf_cal


def create_data_folder(basepath: str = _fn.BASE_DATA_PATH) -> str:
    """
    Creates a new data folder with a unique tracking number in the specified base path.
    
    Parameters
    ----------
    basepath : str, optional
        The base directory where the new tracking number folder will be created.
        Default is the BASE_DATA_PATH.
    
    Returns
    -------
    tn_path : str
        The path to the newly created tracking number folder.
    """
    import os

    tn = newtn()
    tn_path = os.path.join(basepath, tn)
    os.makedirs(tn_path, exist_ok=True)
    return tn_path


__all__ = [
    "load_fits",
    "save_fits",
    "load_psf",
    "load_psf_cube",
    "load_psf_calibration",
    "create_data_folder",
]

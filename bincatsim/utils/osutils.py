# Lets try this way
from typing import Any as _Any
from ..core import root as _fn
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


def load_psf(tn_or_fp: str, **kwargs: _Any):
    """
    Loads a PSF from a FITS file.

    Parameters
    ----------
    tn_or_fp : str
        Tracking number of a simulation folder or path to the FITS file 
        containing the PSF data.
    **kwargs : dict, optional
        Additional options passed to PSFCube when `tn_or_fp` is a tracking
        number. Supported keys are `load_mode`, `cache_size`, and `prefetch`.

    Returns
    -------
    psf : PSFData object | PSFCube object
        If `tn_or_fp` is a tracking number, a PSFCube object containing all the 
        PSFs data of a given simulation will be returned.
        If `tn_or_fp` is a path to a FITS file, a PSFData object containing the PSF data
        of the file will be returned.
    """
    from .psfutils import PSFData, PSFCube
    
    if _optosu.is_tn(tn_or_fp):
        return PSFCube(tn=tn_or_fp, **kwargs)
    
    return PSFData(psf=tn_or_fp)


def load_psf_calibration(tn: str):
    """
    Get the PSF calibration FITS filepath for a tracking number.

    Parameters
    ----------
    tn : str
        Tracking number to identify the folder containing the PSF calibration FITS file.

    Returns
    -------
    calib : PSFData
        The PSFData object containing the calibration data.
    """
    from .psfutils import PSFData
    
    calib = _optosu.getFileList(tn, key='calibration')
    if not isinstance(calib, str):
        raise ValueError(
            f"Expected exactly one calibration file for '{tn}', found {len(calib)}."
        )
    return PSFData(calib, tn=tn)


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


def getSimulationRecord():
    """
    Loads the simulation record CSV file into a pandas DataFrame.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame containing the simulation records.
    """
    import os
    import pandas as pd

    record_path = os.path.join(_fn.SIM_RECORD_FILE)
    if os.path.exists(record_path):
        df = pd.read_csv(record_path, index_col=0)
    else:
        df = pd.DataFrame()
    return df


__all__ = [
    "load_fits",
    "save_fits",
    "load_psf",
    "load_psf_calibration",
    "create_data_folder",
    "getSimulationRecord",
]
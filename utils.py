from astropy.io import fits as _fits
from numpy.typing import ArrayLike as _Array
from numpy.ma import MaskedArray as _masked_array
from numpy import uint8 as _uint8
from typing import Any as _Any


def load_fits(
    filepath: str, return_header: bool = False, as_hdul: bool = False
):
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
    with _fits.open(filepath) as hdul:
        if as_hdul:
            fit = hdul.copy()
        elif len(hdul) == 1:
            fit = hdul[0]
        else:
            fit = hdul[0].data
            if len(hdul) > 1 and hasattr(hdul[1], "data"):
                mask = hdul[1].data.astype(bool)
                fit = _masked_array(fit, mask=mask)
        if return_header:
            header = hdul[0].header
            return fit, header
    return fit


def save_fits(
    filepath: str,
    data: _Array,
    overwrite: bool = True,
    header: dict[str,_Any] | _fits.Header = None,
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
        _fits.writeto(filepath, data.data, header=header, overwrite=overwrite)
        if hasattr(data, "mask"):
            _fits.append(filepath, data.mask.astype(_uint8))
    else:
        _fits.writeto(filepath, data, header=header, overwrite=overwrite)


def _header_from_dict(
    dictheader: dict[str,_Any | tuple[_Any,str]]
) -> _fits.Header:
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
    if isinstance(dictheader, _fits.Header):
        return dictheader
    header = _fits.Header()
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

#######################
## LOGGING UTILITIES ##
#######################

import logging as _l
import logging.handlers as _lh

class Logger():
    
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
        FORMAT = " [%(levelname)s] %(name)s - %(message)s"
        formato = _l.Formatter(fmt=FORMAT)
        handler = _lh.RotatingFileHandler(
            filename="bincat.log",
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

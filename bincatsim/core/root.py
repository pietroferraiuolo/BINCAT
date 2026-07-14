import os

dn = os.path.dirname
ap = os.path.abspath
join = os.path.join

BASE_DATA_PATH: str = join(dn(dn(dn(ap(__file__)))), "data")
SIMPATH: str = join(BASE_DATA_PATH, "simulations")
PSF_DATA_PATH: str = join(SIMPATH, "PSFs")
PSF_FILE: str = join(PSF_DATA_PATH, "1062x2124_gpsf_T.fits")
OBS_DATA_PATH: str = join(SIMPATH, "observations")
BANDS_FILE: str = join(BASE_DATA_PATH, "bands.fits")
PASSBAND_FILE: str = join(BASE_DATA_PATH, "gaiaDR3passband.fits")
SIM_RECORD_FILE: str = join(SIMPATH, "simulations_record.csv")

def SIMULATION_PARAMETERS_PATH(tn: str) -> str:
    return join(OBS_DATA_PATH, tn, f"simulation_parameters.yaml")

def SIM_RECORD_FILE_V(version: int) -> str:
    return join(SIMPATH, f"simulations_record_v{version}.csv")
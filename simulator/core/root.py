import os

dn = os.path.dirname
ap = os.path.abspath
join = os.path.join

BASE_DATA_PATH: str = join(dn(dn(dn(ap(__file__)))), "data")
SIMPATH: str        = join(BASE_DATA_PATH, "simulations")
PSF_DATA_PATH: str  = join(SIMPATH, "PSFs")
OBS_DATA_PATH: str  = join(SIMPATH, "observations")
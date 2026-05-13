import gc
import numpy as np
import xupy as xp
from bincatsim.simulator import GaiaSimulator, CCD
from bincatsim.core.root import PSF_DATA_PATH
from astropy import units as u

try:
    xp.set_device(1) # Set to 2nd GPU (gandalf)
except Exception as e:
    pass

central = []
secondary = []
for c in range(5, 20):
    for s in range(c, min(20, c+4)):
        central.append(c)
        secondary.append(s)

couples = list(zip(central, secondary))

distances = np.arange(195, 221, 5) * u.mas
angle = 90 * u.deg

ccd = CCD(psf=PSF_DATA_PATH+'/1062x2124_gpsf.fits')


if __name__=='__main__':
    
    print(f"Starting bulk simulation: {len(distances)*len(couples)} simulations to run.\n")
    for D in distances:
        for it, (c, s) in enumerate(couples):
            print(
                f"Simulating observation {it+1}/{len(couples)}\nDistance={D} ; central={c},"
                f" secondary={s}", end="\n"
            )

            sim = GaiaSimulator(
                ccd=ccd,
                M1=c,
                M2=s,
                distance=D,
                angle=angle
            )

            tn = sim.observe()

            sim.save_simulation_parameters(tn)
            
            del sim
            gc.collect()

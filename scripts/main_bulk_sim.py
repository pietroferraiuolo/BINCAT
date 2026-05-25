import gc
import numpy as np
import xupy as xp
import bincatsim as bs
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

distances = np.arange(20, 221, 10) * u.mas
angle = 90 * u.deg

ccd = bs.CCD(
    psf=bs.paths.PSF_FILE,
    pixel_scale_x=177*(u.mas/u.pixel),
    pixel_scale_y=59*(u.mas/u.pixel)
)

def analyze_simulation(tn: str):
    amp, phase, almp, acmp = bs.processing.run_IPD_analysis(tn, mp_threshold=0.15)
    return {
        'gof_amp': amp,
        'gof_phase': phase,
        'al_multipeak': almp,
        'ac_multipeak': acmp
    }

if __name__=='__main__':
    
    print(f"Starting bulk simulation: {len(distances)*len(couples)} simulations to run.\n")
    for D in distances:
        for it, (c, s) in enumerate(couples):
            print(
                f"{it+1}/{len(couples)}: Distance={D} ; central={c},"
                f" secondary={s}", end="\n"
            )

            sim = bs.GaiaSimulator(
                ccd=ccd,
                M1=c,
                M2=s,
                distance=D,
                angle=angle
            )

            tn = sim.observe()
            resdict = analyze_simulation(tn)
            sim.update_record_file(tn, other_params=resdict)

            del sim
            gc.collect()

import gc
import numpy as np
import xupy as xp
import bincatsim as bs
from astropy import units as u

try:
    xp.set_device(1) # Set to 2nd GPU (gandalf)
except Exception as e:
    pass

# central = []
# secondary = []
# for c in range(5, 9):
#     for s in range(c, min(9, c+4)):
#         central.append(c)
#         secondary.append(s)

# couples = list(zip(central, secondary))

central = []
secondary = []
c = 7
for s in range(c, min(11, c+4)):
    central.append(c)
    secondary.append(s)

couples = list(zip(central, secondary))

distances = np.arange(138, 221, 2) * u.mas
angle = 90 * u.deg

ccd = bs.CCD(
    psf=bs.paths.PSF_FILE,
    pixel_scale_x=177*(u.mas/u.pixel),
    pixel_scale_y=59*(u.mas/u.pixel)
)

def analyze_simulation(tn: str):
    ipd = bs.IPD(tn)
    ipd()
    return {
        'gof_amp': ipd.gof_amp,
        'gof_phase': ipd.gof_phase,
        'al_multipeak': ipd.frac_multipeak,
        'ac_multipeak': 0.0,
        'delta_m': None,
        'frac_badfit': ipd.frac_badfit,
        'chi2_threshold': ipd._chi2_threshold,
        'phi_threshold': ipd._phi_threshold
    }

if __name__=='__main__':
    
    print(f"Starting bulk simulation: {len(distances)*len(couples)} simulations to run.\n")
    k=1
    for D in distances:
        for it, (c, s) in enumerate(couples):
            print(
                f"[{k}/{len(distances)*len(couples)}] - ({it+1}/{len(couples)}) : Distance={D} ; central={c},"
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
            resdict['delta_m'] = s - c
            sim.update_record_file(tn, other_params=resdict)

            del sim
            gc.collect()
            k += 1

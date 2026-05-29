import os
import gc
import numpy as np
import xupy as xp
import bincatsim as bs
from astropy import units as u
from bincatsim.utils.read_config import resolve_bulk_simulation_config

try:
    xp.set_device(1) # Set to 2nd GPU (gandalf)
except Exception as e:
    pass

parfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
config = resolve_bulk_simulation_config(parfile)

sm = int(config['starting_magnitude'])
dm = int(config['delta_mag'])

max = sm + dm

central = []
secondary = []
for c in range(sm, max + 1):
    for s in range(c, min(max, c + (dm + 1))):
        central.append(c)
        secondary.append(s)

couples = list(zip(central, secondary))


distances = config['distances']
angle = config['angle']
shot_noise = config['shot_noise']
ron = config['ron']

ccd = bs.CCD(
    psf=bs.paths.PSF_FILE,
    pixel_scale_x=177*(u.mas/u.pixel),
    pixel_scale_y=59*(u.mas/u.pixel)
)

band_dict = {
    'wavelength'    : ccd._passbands['lambda'],
    'transmission'  : ccd._passbands['G'],
    'name'          : ccd._bands['band'][0].lower(),
    'zero_point'    : ccd._bands['zero_point'][0]
}

def analyze_simulation(tn: str):
    ipd = bs.IPD(tn)
    ipd(epsilon=1e-3)
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
    
    try:
        print(f"Starting bulk simulation: {len(distances)*len(couples)} simulations to run.\n")
        k=1
        for D in distances:
            for it, (c, s) in enumerate(couples):
                print(
                    f"[{k}/{len(distances)*len(couples)}] - ({it+1}/{len(couples)}) : Distance={D} ; central={c},"
                    f" secondary={s}", end="\n"
                )
                
                C = bs.Star(
                    magnitude=c,
                    type_or_temp=config.get(
                        'central_star_temperature',
                        config.get('central_spectral_type')
                    ),
                    band=config.get('band', 'gaia_g')
                )
                S = bs.Star(
                    magnitude=s,
                    type_or_temp=config.get(
                        'companion_star_temperature',
                        config.get('companion_spectral_type')
                    ),
                    band=config.get('band', 'gaia_g')
                )

                sim = bs.GaiaSimulator(
                    ccd=ccd,
                    central_star=C,
                    companion_star=S,
                    distance=D,
                    angle=angle,
                )

                tn = sim.observe(
                    shot_noise=shot_noise,
                    read_out_noise=ron
                )

                resdict = analyze_simulation(tn)
                resdict['delta_m'] = s - c
                sim.update_record_file(tn, other_params=resdict)

                del sim
                gc.collect()
                k += 1
                
    except KeyboardInterrupt as e:
        from bincatsim.core.root import OBS_DATA_PATH
        import os
        
        fold_to_delete = os.path.join(OBS_DATA_PATH, tn)
        if os.path.exists(fold_to_delete):
            import shutil
            shutil.rmtree(fold_to_delete)
            print(f"Deleted incomplete simulation folder: {fold_to_delete}")
        print("Simulation interrupted.")
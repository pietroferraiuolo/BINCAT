import os
import bincatsim as bs
from astropy import units as u

parfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
config = bs.utils.read_config.resolve_single_simulation_config(parfile)

ccd = bs.CCD(
    psf=bs.paths.PSF_DATA_PATH+'/1062x2124_gpsf_T.fits',
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

def resolve_simulation_parameters():
    distance = config['distance']
    angle = config['angle']
    shot_noise = config['shot_noise']
    ron = config['ron']

    C = bs.Star(
        magnitude=config['M1'],
        type_or_temp=config.get(
            'central_star_temperature',
            config.get('central_spectral_type')
        ),
        band=config.get('band', 'gaia_g')
    )

    S = bs.Star(
        magnitude=config['M2'],
        type_or_temp=config.get(
            'companion_star_temperature',
            config.get('companion_spectral_type')
        ),
        band=config.get('band', 'gaia_g')
    )
    return {
        'central_star': C,
        'companion_star': S,
        'distance': distance,
        'angle': angle,
        'shot_noise': shot_noise,
        'ron': ron,
    }

if __name__ == "__main__":

    params = resolve_simulation_parameters()    
    sn = params.pop('shot_noise')
    ron = params.pop('ron')

    sim = bs.GaiaSimulator(
        ccd=ccd,
        **params
    )

    tn = sim.observe(
        read_out_noise=ron,
        shot_noise=sn
    )

    params = analyze_simulation(tn)
    params['delta_m'] = config['M2'] - config['M1']

    sim.update_record_file(tn, other_params=params)
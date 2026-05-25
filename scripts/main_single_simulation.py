import argparse
from bincatsim.simulator import GaiaSimulator, CCD
from bincatsim.core.root import PSF_DATA_PATH
from astropy import units as u


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate binary star observations with Gaia"
    )
    parser.add_argument(
        "--mag1", "-m1", type=float, help="Magnitude of the central star", required=True
    )
    parser.add_argument(
        "--mag2", "-m2", type=float, help="Magnitude of the secondary star", required=True
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        help="Angular separation of the binary system in milliarcseconds [mas]",
        required=True,
    )
    parser.add_argument(
        "--angle",
        "-a",
        type=float,
        help="Position angle of the secondary star relative to the primary [deg] (defaults to 360)",
        default=360.0,
        required=False,
    )
    return parser

def resolve_args() -> dict[str, u.Quantity | float]:
    parser = build_parser()
    args = parser.parse_args()
    return {
        "M1": args.mag1,
        "M2": args.mag2,
        "distance": args.distance * u.mas,
        "angle": args.angle * u.deg,
    }


if __name__ == "__main__":

    params = resolve_args()

    sim = GaiaSimulator(
        ccd=CCD(
            psf=PSF_DATA_PATH+'/1062x2124_gpsf_T.fits',
            psf_pixel_scale_x = 177*u.mas,
            psf_pixel_scale_y = 59*u.mas
        ),
        **params
    )

    tn = sim.observe()
    
    sim.update_record_file(tn)
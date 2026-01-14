import argparse
from bincatsim.simulator import GaiaSimulator, CCD
from bincatsim.core.root import PSF_DATA_PATH
from astropy import units as u


def main(M1: float, M2: float, distance: int, angle: float, ccd: "CCD" = None):
    if ccd is None:
        ccd = CCD(
            psf=PSF_DATA_PATH + "/1062x2124_gpsf.fits",
            pixel_scale_x=177 * u.mas,
            pixel_scale_y=59 * u.mas,
        )

    bs = GaiaSimulator(ccd=ccd, M1=M1, M2=M2, distance=distance, angle=angle)

    bs.observe(ccd)


if __name__ == "__main__":

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

    args = parser.parse_args()

    main(M1=args.mag1, M2=args.mag2, distance=args.distance, angle=args.angle)

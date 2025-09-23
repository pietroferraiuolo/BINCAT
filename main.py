from simulator import BinarySystem, CCD

if __name__ == "__main__":
    bs = BinarySystem(M1=7, M2=7, distance=200, shape=(4500,4500))
    ccd = CCD(psf="data/simulations/PSFs/20250922_2_gaia_psf.fits")

    tn = bs.observe(ccd)
    print(tn)
    
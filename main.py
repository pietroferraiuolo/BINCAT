from simulator import BinarySystem, CCD

if __name__ == "__main__":
    ccd = CCD(psf="data/simulations/PSFs/20250923_3_gaia_psf.fits")
    bs = BinarySystem(ccd=ccd, M1=7, M2=7, distance=200)

    tn = bs.observe(ccd)
    print(tn)
    
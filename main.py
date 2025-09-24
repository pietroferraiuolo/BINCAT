from simulator import BinarySystem, CCD

def main():
    ccd = CCD(psf="data/simulations/PSFs/1062x2124_gpsf.fits")
    bs = BinarySystem(ccd=ccd, M1=7, M2=7, distance=200)

    tn = bs.observe(ccd)
    print(tn)
    
    
if __name__ == "__main__":
    main()
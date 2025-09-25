from simulator import BinarySystem, CCD

def main(M1, M2, distance):
    ccd = CCD(psf="data/simulations/PSFs/1062x2124_gpsf.fits")
    bs = BinarySystem(ccd=ccd, M1=M1, M2=M2, distance=distance)

    tn = bs.observe(ccd)
    print(tn)
    
    
if __name__ == "__main__":
    main()
import time
import argparse
import bincatsim as bs
import pandas as pd
from tqdm import tqdm

def run_ipd(tn: str):
    ipd = bs.ipd_.IPD(tn=tn)
    ipd()
    return ipd.gof_amp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--row", type=int, default=0, help="Row index to start from")
    parser.add_argument('-w', '--wait-time', type=int, default=0, help='Wait time in seconds before starting')
    args = parser.parse_args()
    
    record_file = "/home/pietrof/NAS/BINCAT/data/simulations/simulations_record_v1.csv"
    df = pd.read_csv(record_file)
    sim_dataset = pd.read_csv(record_file)

    time.sleep(args.wait_time)  # Wait for the specified time before starting the processing
    try:
        for idx, (_, row) in enumerate(tqdm(sim_dataset.iterrows(), total=sim_dataset.shape[0])):
            if idx < args.row:
                continue
            tn = row['TN']
            gof_amp = run_ipd(tn)
            df.loc[df['TN'] == tn, ['gof_amp']] = gof_amp
    except KeyboardInterrupt:
        print("\nInterrupt received. Saving progress...")
    
    df.to_csv(record_file, index=False)

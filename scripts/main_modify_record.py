import argparse
import bincatsim as bs
import pandas as pd
from tqdm import tqdm

def run_ipd(tn: str):
    ipd = bs.ipd_.IPD(tn=tn)
    ipd()
    return ipd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--row", type=int, default=0, help="Row index to start from")
    parser.add_argument("-v", "--version", type=str, default='', help="Version of the record file to use (e.g., v1, v2)")
    
    args = parser.parse_args()

    version = '' if args.version == '' else '_v'+args.version
    
    record_file = f"/home/pietrof/NAS/BINCAT/data/simulations/simulations_record{version}.csv"
    df = pd.read_csv(record_file)
    sim_dataset = pd.read_csv(record_file)

    try:
        for idx, (_, row) in enumerate(tqdm(sim_dataset.iterrows(), total=sim_dataset.shape[0])):
            if idx < args.row:
                continue
            tn = row['TN']
            ipd_result = run_ipd(tn)
            for key, attr in zip(
                ['gof_amp', 'gof_phase', 'al_multipeak', 'frac_badfit', 'chi2_threshold', 'phi_threshold'],
                ['gof_amp', 'gof_phase', 'frac_multipeak', 'frac_badfit', '_chi2_threshold', '_phi_threshold']
                ):
                df.loc[df['TN'] == tn, [key]] = getattr(ipd_result, attr)

    except KeyboardInterrupt:
        print("\nInterrupt received. Saving progress...")
    
    df.to_csv(record_file, index=False)
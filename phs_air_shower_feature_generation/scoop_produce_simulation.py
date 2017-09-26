"""
Call with 'python -m scoop --hostfile scoop_hosts.txt'

Usage: phs.extract.air-shower.features.sim --sim_dir=DIR --out_dir=DIR

Options:
    --sim_dir=DIR   The input phs/sim/ directory with the '.phs.jsonl' runs
    --out_dir=DIR   The output directory
"""
import docopt
import scoop
from glob import glob
from os.path import join
from os.path import exists
from os.path import basename
from os.path import dirname
from os import makedirs
import photon_stream as ps
import phs_air_shower_feature_generation as psfg


def extract_features(job): 
    makedirs(dirname(job['feature_path']), exist_ok=True)
    try:
        psfg.extract_from_simulation(
            path=job['input_path'], 
            out_path=job['feature_path']
        )
    except:
        return 0
    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        sim_dir = arguments['--sim_dir']
        out_dir = arguments['--out_dir']

        jobs = []
        for input_path in glob(join(obs_dir,'*','*.phs.jsonl.gz')):
            feature_path = join(out_dir, basename(input_path).split('.')[0]+'.sim.features.msg')
            if not exists(feature_path):
                jobs.append({'input_path': input_path, 'feature_path': feature_path})

        job_return_codes = list(scoop.futures.map(extract_features, jobs))

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()

"""
Call with 'python -m scoop --hostfile scoop_hosts.txt'

Usage: phs.extract.air-shower.features.sim --sim_block_dir=DIR --out_dir=DIR

Options:
    --sim_block_dir=DIR   The input phs/sim/block directory (e.g. gamma_gustav/ ) with the '.phs.jsonl' runs
    --out_dir=DIR         The output directory
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
    triggered, thrown = psfg.extract_from_simulation(
        phs_path=job['in_path'], 
    )
    psfg.write_simulation_extraction(
        triggered=triggered, thrown=thrown,
        out_path=job['out_path']
    )
    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        sim_block_dir = arguments['--sim_block_dir']
        out_dir = arguments['--out_dir']

        jobs = []
        for in_path in glob(join(sim_block_dir,'*.phs.jsonl.gz')):
            out_path = join(out_dir, basename(in_path).split('.')[0]+'.ft.msg')
            if not exists(out_path):
                jobs.append({'in_path': in_path, 'out_path': out_path})

        job_return_codes = list(scoop.futures.map(extract_features, jobs))

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()

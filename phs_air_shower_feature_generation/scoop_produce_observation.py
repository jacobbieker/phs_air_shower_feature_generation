"""
Call with 'python -m scoop --hostfile scoop_hosts.txt'

Usage: phs.extract.air-shower.features.obs --obs_dir=DIR --out_dir=DIR

Options:
    --obs_dir=DIR   The input phs/obs/ directory with the '.phs.jsonl' runs
    --out_dir=DIR   The outpub binary directory
"""
import docopt
import scoop
from glob import glob
from os.path import join
from os.path import exists
from os.path import basename
from os.path import dirname
from os import makedirs
import shutil
import tempfile
import photon_stream as ps
import fact
import phs_air_shower_feature_generation as psfg
 
 
def extract_features(job):
    makedirs(dirname(job['out_path']), exist_ok=True)
    psfg.extract_from_observation(
        path=job['in_path'],
        out_path=job['out_path']
    )
    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        obs_dir = arguments['--obs_dir']
        out_dir = arguments['--out_dir']

        jobs = []
        for in_path in glob(join(obs_dir,'*','*','*','*.phs.jsonl.gz')):
            p = fact.path.parse(in_path)
            out_path = fact.path.tree_path(p['night'], p['run'], out_dir, '.ft.msg')
            if not exists(out_path):
                jobs.append({'in_path': in_path, 'out_path': out_path})

        job_return_codes = list(scoop.futures.map(extract_features, jobs))

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()

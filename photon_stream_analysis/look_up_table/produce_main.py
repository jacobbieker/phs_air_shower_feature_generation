"""
Call with 'python -m scoop --hostfile scoop_hosts.txt'

Usage: phs.lut.produce --sim_block_dir=DIR --out_dir=DIR

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
import photon_stream_analysis as psa


def wrapp_produce_lut_run(job): 
    psa.look_up_table.produce.simulation_run(
        phs_path=job['phs_path'],
        out_path=job['out_path']
    )
    return 1


def main():
    try:
        arguments = docopt.docopt(__doc__)
        sim_block_dir = arguments['--sim_block_dir']
        out_dir = arguments['--out_dir']
        tmp_dir = '.'+out_dir+'.tmp'
        
        # MAP
        # ---
        jobs = []
        for phs_path in glob(join(sim_block_dir,'*.phs.jsonl.gz')):
            out_path = join(tmp_dir, basename(phs_path).split('.')[0]+'.lut')
            if not exists(out_path):
                jobs.append({'phs_path': phs_path, 'out_path': out_path})

        job_return_codes = list(scoop.futures.map(extract_features, jobs))

        # REDUCE
        # ------
        out_paths = []
        for job in jobs:
            if exists(job['out_path']):
                out_paths.append(job['out_path'])
        psa.look_up_table.produce.concatenate(out_paths, out_dir)

    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()

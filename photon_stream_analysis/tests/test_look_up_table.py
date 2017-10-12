import numpy as np
import photon_stream_analysis as psa
import pkg_resources
import os
import tempfile

phs_path = pkg_resources.resource_filename(
    'photon_stream_analysis',
    os.path.join('tests','resources','014884.phs.jsonl.gz')
)


def test_production_scenario(out_dir):
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix='psa_') as tmp:
            run_production(out_dir=tmp)
    else:
        os.makedirs(out_dir, exist_ok=True)
        run_production(out_dir=out_dir)


def run_production(out_dir):
    out_path = os.path.join(out_dir, 'lut')
    psa.look_up_table.produce.simulation_run(phs_path, out_path)

    L = psa.look_up_table.LookUpTable(out_path)
    assert len(L.energy) == 471
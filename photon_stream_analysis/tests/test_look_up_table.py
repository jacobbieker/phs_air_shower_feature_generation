import numpy as np
import photon_stream_analysis as psa
import pkg_resources
import os
import tempfile

phs_path = pkg_resources.resource_filename(
    'photon_stream_analysis',
    os.path.join('tests','resources','014884.phs.jsonl.gz')
)


def test_production():

    with tempfile.TemporaryDirectory(prefix='psa') as tmp:
        out_path = os.path.join(tmp, 'lut')
        psa.look_up_table.produce.simulation_run(phs_path, out_path)

        L = psa.look_up_table.LookUpTable(out_path)
        assert len(L.energy) == 471
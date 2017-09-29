import numpy as np
import photon_stream_analysis as psa
import pkg_resources
import os
import tempfile

mmcs_corsika_path = pkg_resources.resource_filename(
    'photon_stream_analysis',
    os.path.join('tests','resources','014884.ch.gz')
)

phs_path = pkg_resources.resource_filename(
    'photon_stream_analysis',
    os.path.join('tests','resources','014884.phs.jsonl.gz')
)


def test_from_simulation():
    triggered, thrown = psa.extract.from_simulation(
        phs_path=phs_path,
        mmcs_corsika_path=mmcs_corsika_path
    )

    assert np.all(triggered.extraction == 0)

    with tempfile.TemporaryDirectory(prefix='psa') as tmp:
        out_path = os.path.join(tmp, '014884.ft.msg')
        psa.extract.write_simulation_extraction(
            triggered=triggered,
            thrown=thrown,
            out_path=out_path
        )

        assert os.path.exists(out_path)
        assert os.path.exists(out_path + '.thrown')
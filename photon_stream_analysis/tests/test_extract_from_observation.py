import numpy as np
import photon_stream_analysis as psa
import pkg_resources
import os
import tempfile

phs_path = pkg_resources.resource_filename(
    'photon_stream_analysis',
    os.path.join('tests','resources','20170119_229_pass4_100events.phs.jsonl.gz')
)


def test_from_observation():
    triggered = psa.extract.from_observation(
        phs_path=phs_path,
    )

    assert np.all(triggered.extraction == 0)

    with tempfile.TemporaryDirectory(prefix='psa') as tmp:
        out_path = os.path.join(tmp, '20170119_229.ft.msg')
        psa.extract.write_observation_extraction(
            triggered=triggered,
            out_path=out_path
        )

        assert os.path.exists(out_path)
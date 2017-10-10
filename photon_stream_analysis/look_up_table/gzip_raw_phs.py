import numpy as np
import gzip
import io


def raw_phs_to_raw_phs_gz(raw_phs):
    ssz = io.BytesIO()
    h = gzip.open(ssz, 'wb')
    h.write(raw_phs.tobytes())
    h.close()
    ssz.seek(0)
    return ssz.read()


def raw_phs_gz_to_raw_phs(raw_phs_gz):
    h = gzip.open(io.BytesIO(raw_phs_gz), 'rb')
    raw_phs_str = h.read()
    h.close()
    return np.fromstring(raw_phs_str, dtype=np.uint8)
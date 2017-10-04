import numpy as np
from array import array
import gzip
import io
import photon_stream as ps

VERSION = 1


def similarity(ims1, ims2):
    i1 = np.sqrt((ims1**2).sum())
    i2 = np.sqrt((ims2**2).sum())
    i = np.sort([i1,i2])[-1]
    diff = np.sqrt(
        ((ims1 - ims2)**2).sum()
    )
    if diff == 0:
        return 1.0
    else:
        return i/diff


def raw_phs_to_image_sequence(raw_phs):
    image_sequence = np.zeros(
        shape=(
            ps.io.magic_constants.NUMBER_OF_TIME_SLICES,
            ps.io.magic_constants.NUMBER_OF_PIXELS
        ),
        dtype=np.int16,
    )
    pixel_chid = 0
    for s in raw_phs:
        if s == ps.io.binary.LINEBREAK[0]:
            pixel_chid += 1
        else:
            image_sequence[
                s - ps.io.magic_constants.NUMBER_OF_TIME_SLICES_OFFSET_AFTER_BEGIN_OF_ROI, 
                pixel_chid
            ] += 1 

    return image_sequence



def time_series_and_mask_to_raw_phs(time_series, mask, raw_phs):
    i = 0
    j = 0
    for time_serie in time_series:
        for photon in time_serie:
            if mask[i]:
                raw_phs[j] = photon
                j +=1
            i += 1
        raw_phs[j] = ps.io.binary.LINEBREAK[0]
        j += 1
    return raw_phs


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



lut = {
    'run': array('L'),
    'event': array('L'),
    'reuse': array('L'),

    'energy': array('f'),
    'theta': array('f'),
    'phi': array('f'),
    'impact_x': array('f'),
    'impact_y': array('f'),
    'particle': array('f'),
    'hight_of_first_interaction': array('f'),

    'number_photons': array('f'),
    'cog_camera_x': array('f'),
    'cog_camera_y': array('f'),

    'raw_phs_gz': [],
}


def read(path):
    with gzip.open(path, 'rb') as fin:
        return _read_lut_from_file(fin)

def write(lut, path):
    with gzip.open(path, 'wb') as fout:
        return _append_lut_to_file(lut, fout)   


def _append_lut_to_file(lut, fout):
    fout.write(np.uint64(VERSION).tobytes())
    number_events = len(lut['run'])

    fout.write(np.uint64(number_events).tobytes())

    fout.write(np.array(lut['run'], dtype=np.uint32).tobytes())
    fout.write(np.array(lut['event'], dtype=np.uint32).tobytes())
    fout.write(np.array(lut['reuse'], dtype=np.uint32).tobytes())

    fout.write(np.array(lut['energy'], dtype=np.float32).tobytes())
    fout.write(np.array(lut['theta'], dtype=np.float32).tobytes())
    fout.write(np.array(lut['phi'], dtype=np.float32).tobytes())
    fout.write(np.array(lut['impact_x'], dtype=np.float32).tobytes())
    fout.write(np.array(lut['impact_y'], dtype=np.float32).tobytes())
    fout.write(np.array(lut['particle'], dtype=np.float32).tobytes()) 
    fout.write(np.array(lut['hight_of_first_interaction'], dtype=np.float32).tobytes())

    raw_phs_gz_lens = [len(e) for e in lut['raw_phs_gz']]
    raw_phs_gz_lens = np.array(raw_phs_gz_lens, dtype=np.uint32)
    fout.write(raw_phs_gz_lens.tobytes())

    for raw_phs_gz in lut['raw_phs_gz']:
        fout.write(raw_phs_gz)


def _read_lut_from_file(fin):
    fs = np.fromstring
    version = fs(fin.read(8), dtype=np.uint64, count=1)[0]
    assert version == VERSION

    number_events = int(fs(fin.read(8), dtype=np.uint64, count=1)[0])
    print(number_events)
    lut = {}

    lut['run'] = array('L', fs(fin.read(number_events*4), dtype=np.uint32))
    lut['event'] = array('L', fs(fin.read(number_events*4), dtype=np.uint32))
    lut['reuse'] = array('L', fs(fin.read(number_events*4), dtype=np.uint32))

    lut['energy'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['theta'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['phi'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['impact_x'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['impact_y'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['particle'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))
    lut['hight_of_first_interaction'] = array('f', fs(fin.read(number_events*4), dtype=np.float32))

    raw_phs_gz_lens = fs(fin.read(number_events*4), dtype=np.uint32)
    lut['raw_phs_gz'] = []
    for raw_phs_gz_len in raw_phs_gz_lens:
        lut['raw_phs_gz'].append(
            fin.read(raw_phs_gz_len)
        )
    return lut


events = ps.SimulationReader('photon_stream/photon_stream/tests/resources/011014.phs.jsonl.gz')

def append_events_to_lut(events, lut):
    for event in events:
        st = event.simulation_truth

        lut['run'].append(st.run)
        lut['event'].append(st.event)
        lut['reuse'].append(st.reuse)

        lut['energy'].append(st.air_shower.energy)
        lut['theta'].append(st.air_shower.theta)
        lut['phi'].append(st.air_shower.phi)

        lut['impact_x'].append(st.air_shower.impact_x(st.reuse))
        lut['impact_y'].append(st.air_shower.impact_y(st.reuse))
        lut['particle'].append(st.air_shower.particle)

        lut['hight_of_first_interaction'].append(
            st.air_shower.hight_of_first_interaction
        )

        cluster = ps.PhotonStreamCluster(event.photon_stream)
        mask = cluster.labels >= 0
        time_series = event.photon_stream.time_lines
        number_photons = mask.sum()
        number_pixels = len(event.photon_stream.time_lines)

        raw_phs = np.zeros(
            number_photons + number_pixels, 
            dtype=np.uint8,
        )

        raw_phs = time_series_and_mask_to_raw_phs(time_series, mask, raw_phs)
        raw_phsz = raw_phs_to_raw_phs_gz(raw_phs)

        lut['raw_phs_gz'].append(raw_phsz)
    return lut
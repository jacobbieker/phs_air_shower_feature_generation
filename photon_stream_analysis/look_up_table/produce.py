import os
import photon_stream as ps
from os.path import join
import numpy as np

from .structure import head
from .gzip_raw_phs import raw_phs_to_raw_phs_gz
from .. import features


def rrr(event):
    cluster = ps.PhotonStreamCluster(event.photon_stream)
    mask = cluster.labels >= 0
    number_photons = mask.sum()
    raw_phs = np.zeros(
        number_photons + ps.io.magic_constants.NUMBER_OF_PIXELS, 
        dtype=np.uint8,
    )
    raw_phs = ps.representations.masked_raw_phs(mask, event.photon_stream.raw)
    raw_phsz = raw_phs_to_raw_phs_gz(raw_phs)

    evt = {}
    evt['raw_phs_gz'] = raw_phsz
    evt['number_photons'] = number_photons
    ellipse = features.extract_ellipse(cluster.xyt[mask])
    evt['cog_cx_pap'] = ellipse['center'][0]
    evt['cog_cy_pap'] = ellipse['center'][1]
    evt['cluster'] = cluster
    return evt


def simulation_run(phs_path, out_path, mmcs_corsika_path=None):
    os.makedirs(out_path, exist_ok=True)

    raw_phs_gzs = []
    files = []
    k = {}
    for i in range(len(head)):
        name = head[i][0]
        files.append(open(join(out_path, name), 'wb'))
        k[name] = i

    def write(key, value):
        files[k[key]].write(head[k[key]][1](value).tobytes())

    for event in ps.SimulationReader(phs_path, mmcs_corsika_path=mmcs_corsika_path):
        try:
            st = event.simulation_truth
            evt = rrr(event)
        except Exception as e:
            print(e)
            print(event)
            with open(join(out_path, 'error.log'), "at") as fout:
                fout.write(str(e)+'\n')
                fout.write(str(event)+'\n')
            continue

        write('run', st.run)
        write('event', st.event)
        write('reuse', st.reuse)

        write('telescope_az', np.deg2rad(event.az)) # w.r.t. MARS-CERES
        write('telescope_zd', np.deg2rad(event.zd)) # w.r.t. MARS-CERES

        source_az_wrt_corsika = st.air_shower.phi
        source_az = source_az_wrt_corsika + st.air_shower.raw_corsika_event_header[93-1]

        source_zd_wrt_corsika = st.air_shower.theta
        source_zd = source_zd_wrt_corsika

        write('source_zd', source_zd)
        write('source_az', source_az)

        write('energy', st.air_shower.energy)
        write('impact_x', st.air_shower.impact_x(st.reuse))
        write('impact_y', st.air_shower.impact_y(st.reuse))
        write('particle', np.uint16(np.round(st.air_shower.particle)))
        write('hight_of_first_interaction', st.air_shower.hight_of_first_interaction)

        write('number_photons', evt['number_photons'])
        write('cog_cx_pap', evt['cog_cx_pap'])
        write('cog_cy_pap', evt['cog_cy_pap'])
        raw_phs_gzs.append(evt['raw_phs_gz'])

    for f in files:
        f.close()

    raw_phs_gz_lens = [len(e) for e in raw_phs_gzs]
    raw_phs_gz_lens = np.array(raw_phs_gz_lens, dtype=np.uint32)
    with open(join(out_path, 'phs_gz_lens'), 'wb') as fout:
        fout.write(raw_phs_gz_lens.tobytes())

    with open(join(out_path, 'phs_gz'), 'wb') as fout:
        for raw_phs_gz in raw_phs_gzs:
            fout.write(raw_phs_gz)



def concatenate(lut_paths, out_path):
    os.makedirs(out_path, exist_ok=True)
    for lut_path in lut_paths:
        for i in range(len(head)):
            name = head[i][0]
            with open(join(out_path, name), "ab") as fo, open(join(lut_path, name), "rb") as fi:
                fo.write(fi.read())
        with open(join(out_path, 'phs_gz_lens'), "ab") as fo, open(join(lut_path, 'phs_gz_lens'), "rb") as fi:
            fo.write(fi.read())
        with open(join(out_path, 'phs_gz'), "ab") as fo, open(join(lut_path, 'phs_gz'), "rb") as fi:
            fo.write(fi.read())


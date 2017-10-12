import os
import photon_stream as ps
from os.path import join
import numpy as np

from .structure import head
from .gzip_raw_phs import raw_phs_to_raw_phs_gz
from .. import features
from ..transformations import corsika_impact_to_ceres_impact
from ..transformations import corsika_az_zd_to_ceres_az_zd


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
    evt['cluster'] = cluster

    # Air-Shower features
    evt['number_photons'] = number_photons
    ellipse = features.extract_ellipse(cluster.xyt[mask])
    evt['ellipse_cog_x'] = ellipse['center'][0]
    evt['ellipse_cog_y'] = ellipse['center'][1]
    evt['ellipse_ev0_x'] = ellipse['ev0'][0]
    evt['ellipse_ev0_y'] = ellipse['ev0'][1]
    evt['ellipse_std0'] = ellipse['std0']
    evt['ellipse_std1'] = ellipse['std1']

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

        az_offset_between_magnetic_and_geographic_north = st.air_shower.raw_corsika_event_header[93-1]
        az_offset_between_corsika_and_ceres = - np.pi + az_offset_between_magnetic_and_geographic_north
        
        source_az, source_zd = corsika_az_zd_to_ceres_az_zd(
            source_az_wrt_corsika=st.air_shower.phi,
            source_zd_wrt_corsika=st.air_shower.theta,
            az_offset_between_corsika_and_ceres=az_offset_between_corsika_and_ceres,
        )

        write('source_az', source_az) # w.r.t. MARS-CERES
        write('source_zd', source_zd) # w.r.t. MARS-CERES
        
        impact_x, impact_y = corsika_impact_to_ceres_impact(
            impact_x_wrt_corsika=st.air_shower.impact_x(st.reuse),
            impact_y_wrt_corsika=st.air_shower.impact_y(st.reuse),
            az_offset_between_corsika_and_ceres=az_offset_between_corsika_and_ceres,
        )

        write('impact_x', impact_x)
        write('impact_y', impact_y)

        write('energy', st.air_shower.energy)
        write('particle', np.uint16(np.round(st.air_shower.particle)))
        write('hight_of_first_interaction', st.air_shower.hight_of_first_interaction)

        # Air-Shower features
        write('number_photons', evt['number_photons'])
        write('ellipse_cog_x', evt['ellipse_cog_x'])
        write('ellipse_cog_y', evt['ellipse_cog_y'])
        write('ellipse_ev0_x', evt['ellipse_ev0_x'])
        write('ellipse_ev0_y', evt['ellipse_ev0_y'])
        write('ellipse_std0', evt['ellipse_std0'])
        write('ellipse_std1', evt['ellipse_std1'])

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


import numpy as np
from array import array
import gzip
import io
import photon_stream as ps
import photon_stream_analysis as psa

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

d2r = np.deg2rad

import os
from os.path import join


def show_point_cloud(pcl):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    add_point_cloud_2_ax(pcl=pcl, ax=ax)

def add_ring_2_ax(x,y,z,r, ax, color='k', line_width=1.0):
    p = Circle((x, y), r, edgecolor=color, facecolor='none', lw=line_width)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")


def add_point_cloud_2_ax(pcl, ax, color='b'):
    pcl[:,2] *= 1e9
    min_time = pcl[:,2].min()
    max_time = pcl[:,2].max()

    phs_cls = ps.PhotonStream()
    fovR = phs_cls.geometry['fov_radius']
    add_ring_2_ax(x=0.0, y=0.0, z=min_time, r=fovR, ax=ax)
    ax.set_xlim(-fovR, fovR)
    ax.set_ylim(-fovR, fovR)
    ax.set_zlim(min_time, max_time)
    ax.set_xlabel('x/deg')
    ax.set_ylabel('y/deg')
    ax.set_zlabel('t/ns')
    ax.scatter(
        pcl[:,0],
        pcl[:,1],
        pcl[:,2],
        lw=0,
        alpha=0.075,
        s=55.,
        c=color
    )



head = [
    ('run', np.uint16),
    ('event', np.uint16),
    ('reuse', np.uint16),

    ('azimuth', np.float16),
    ('zenith', np.float16),

    ('number_photons', np.float16),
    ('cog_cx_pap', np.float16),
    ('cog_cy_pap', np.float16),

    ('energy', np.float16),
    ('theta', np.float16),
    ('phi', np.float16),
    ('impact_x', np.float16),
    ('impact_y', np.float16),
    ('particle', np.float16),
    ('hight_of_first_interaction', np.float16),
]

sorted_keys = [
    ('number_photons', np.float16),
    ('cog_cx_pap', np.float16),
    ('cog_cy_pap', np.float16),   
]

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



def raw_phs_to_point_cloud(raw_phs, cx, cy):
    number_photons = len(raw_phs) - ps.io.magic_constants.NUMBER_OF_PIXELS
    cloud = np.zeros(shape=(number_photons,3))
    pixel_chid = 0
    p = 0
    for s in raw_phs:
        if s == ps.io.binary.LINEBREAK[0]:
            pixel_chid += 1
        else:
            cloud[p,0] = cx[pixel_chid]
            cloud[p,1] = cy[pixel_chid]
            cloud[p,2] = s*ps.io.magic_constants.TIME_SLICE_DURATION_S
            p += 1
    return cloud


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


def rrr(event):
    evt = {}
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

    evt['raw_phs_gz'] = raw_phsz

    evt['number_photons'] = number_photons
    air_shower_photons_ns = cluster.xyt[mask]
    air_shower_photons = air_shower_photons_ns.copy()
    fov_radius = event.photon_stream.geometry['fov_radius']
    air_shower_photons[:,0:2] *= (fov_radius*2.0)

    ellipse = psa.features.extract_ellipse(air_shower_photons)
    evt['cog_cx_pap'] = d2r(ellipse['center'][0])
    evt['cog_cy_pap'] = d2r(ellipse['center'][1])
    return evt



def run2lut(phs_path, out_path):
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

    for event in ps.SimulationReader(phs_path):
        st = event.simulation_truth

        write('run', st.run)
        write('event', st.event)
        write('reuse', st.reuse)

        write('azimuth', d2r(event.az))
        write('zenith', d2r(event.zd))

        write('energy', st.air_shower.energy)
        write('theta', d2r(st.air_shower.theta))
        write('phi', d2r(st.air_shower.phi))
        write('impact_x', st.air_shower.impact_x(st.reuse))
        write('impact_y', st.air_shower.impact_y(st.reuse))
        write('particle', st.air_shower.particle)
        write('hight_of_first_interaction', st.air_shower.hight_of_first_interaction)

        evt = rrr(event)
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





class Lut():
    def __init__(self, path):

        for i in range(len(head)):
            name = head[i][0]
            with open(join(path, name), 'rb') as fi:
                setattr(self, name, np.fromstring(fi.read(), dtype=head[i][1]))
        
        self.number_events = len(self.energy)
        
        for i in range(len(head)):
            name = head[i][0]
            assert len(getattr(self, name)) == self.number_events

        with open(join(path, 'phs_gz_lens'), 'rb') as fi:
            self._raw_phs_gz_lens = np.fromstring(fi.read(), dtype=np.uint32)
        
        with open(join(path, 'phs_gz'), 'rb') as fi:
            self._raw_phs_gz = []
            for raw_phs_gz_len in self._raw_phs_gz_lens:
                self._raw_phs_gz.append(fi.read(raw_phs_gz_len))

        self.init_sorted_keys()
        phs_cls = ps.PhotonStream()
        self._cx = phs_cls.geometry['x_angle']
        self._cy = phs_cls.geometry['y_angle']


    def init_sorted_keys(self):
        for i in range(len(sorted_keys)):
            name = sorted_keys[i][0]
            values = getattr(self, name)
            aso = np.argsort(values).astype(np.uint32)
            setattr(self, 'argsort_'+name, aso)
            setattr(self, 'sorted_'+name, values[aso].astype(np.float16))


    def idx_for_number_photons_within(self, min_val, max_val):
        return self._idx_for_attribute_within('number_photons', min_val, max_val)

    def idx_for_cog_cx_pap_within(self, min_val, max_val):
        return self._idx_for_attribute_within('cog_cx_pap', min_val, max_val)

    def idx_for_cog_cy_pap_within(self, min_val, max_val):
        return self._idx_for_attribute_within('cog_cy_pap', min_val, max_val)


    def _idx_for_attribute_within(self, key, min_val, max_val):
        sorted_values = getattr(self, 'sorted_'+key)
        argsort_values = getattr(self, 'argsort_'+key)
        ll = np.searchsorted(sorted_values, min_val)
        ul = np.searchsorted(sorted_values, max_val)
        return argsort_values[np.arange(ll, ul)]


    def raw_phs(self, index):
        return raw_phs_gz_to_raw_phs(self._raw_phs_gz[index])


    def image_sequence(self, index):
        return raw_phs_to_image_sequence(self.raw_phs(index))


    def point_cloud(self, index):
        return raw_phs_to_point_cloud(self.raw_phs(index), cx=self._cx, cy=self._cy)



def not_unique(list_of_index_arrays):
    l = len(list_of_index_arrays)
    idxs = np.concatenate(list_of_index_arrays)
    u = np.unique(idxs, return_counts=True)
    return u[0][u[1] == l]

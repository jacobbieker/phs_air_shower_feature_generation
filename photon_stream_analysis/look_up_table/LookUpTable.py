import numpy as np
import os
from os.path import join
from .structure import head 
from .structure import sorted_keys 
import photon_stream as ps
from .gzip_raw_phs import raw_phs_gz_to_raw_phs
from ..transformations import ray_local_system_to_principal_aperture_plane_system


class LookUpTable():
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
        return ps.representations.raw_phs_to_image_sequence(self.raw_phs(index))


    def image(self, index):
        return ps.representations.raw_phs_to_image(self.raw_phs(index))


    def point_cloud(self, index):
        return ps.representations.raw_phs_to_point_cloud(
            self.raw_phs(index), 
            cx=ps.GEOMETRY.x_angle,
            cy=ps.GEOMETRY.y_angle
        )

    def pap(self, index):
        return ray_local_system_to_principal_aperture_plane_system(
            impact_x=self.impact_x[index],
            impact_y=self.impact_y[index],
            source_az=self.source_az[index],
            source_zd=self.source_zd[index],
            telescope_az=self.telescope_az[index],
            telescope_zd=self.telescope_zd[index],
        )






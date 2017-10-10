import numpy as np
import photon_stream as ps
from .produce import rrr
from .distance_metric import difference_image


def find_matches(event_features, lut, cog_angular_distance=np.deg2rad(0.175)):
    evt = event_features

    ul = evt['number_photons']*1.05
    ll = evt['number_photons']*0.95
    valid_num_ph = lut.idx_for_number_photons_within(ll, ul)

    ul = evt['cog_cx_pap'] + cog_angular_distance
    ll = evt['cog_cx_pap'] - cog_angular_distance
    valid_cog_cx_pap = lut.idx_for_cog_cx_pap_within(ll, ul)

    ul = evt['cog_cy_pap'] + cog_angular_distance
    ll = evt['cog_cy_pap'] - cog_angular_distance
    valid_cog_cy_pap = lut.idx_for_cog_cy_pap_within(ll, ul)

    return not_unique([valid_num_ph, valid_cog_cx_pap, valid_cog_cy_pap])


def not_unique(list_of_index_arrays):
    l = len(list_of_index_arrays)
    idxs = np.concatenate(list_of_index_arrays)
    u = np.unique(idxs, return_counts=True)
    return u[0][u[1] == l]


def match(event, lut):
    event_features = rrr(event)
    match_candidates = find_matches(event_features, lut)

    image = ps.representations.raw_phs_to_image(event.photon_stream.raw)

    for candidate in match_candidates:
        ccandidate_image = lut.image(candidate)
        image_diff = difference_image(image, ccandidate_image)
        print(image_diff)




import numpy as np
import photon_stream as ps
from .produce import rrr
from .distance_metric import difference_image
from .distance_metric import difference_image_sequence
from ..transformations import particle_ray_from_corsika_to_principal_aperture_plane
from ..transformations import ceres_azimuth_rad_to_corsika_azimuth_rad

def not_unique(list_of_index_arrays):
    l = len(list_of_index_arrays)
    idxs = np.concatenate(list_of_index_arrays)
    u = np.unique(idxs, return_counts=True)
    return u[0][u[1] == l]


def match_candidates_based_on_coarse_features(
    event_features, 
    lut, 
    absolute_cog_radius=np.deg2rad(0.2),
    relative_number_photons_radius=0.05
):
    evt = event_features

    ul = evt['number_photons']*(1.0 + relative_number_photons_radius)
    ll = evt['number_photons']*(1.0 - relative_number_photons_radius)
    valid_num_ph = lut.idx_for_number_photons_within(ll, ul)

    ul = evt['cog_cx_pap'] + absolute_cog_radius
    ll = evt['cog_cx_pap'] - absolute_cog_radius
    valid_cog_cx_pap = lut.idx_for_cog_cx_pap_within(ll, ul)

    ul = evt['cog_cy_pap'] + absolute_cog_radius
    ll = evt['cog_cy_pap'] - absolute_cog_radius
    valid_cog_cy_pap = lut.idx_for_cog_cy_pap_within(ll, ul)

    matches_within_cog_rectangle = not_unique(
        [valid_num_ph, valid_cog_cx_pap, valid_cog_cy_pap]
    )

    cog_cx_pap_candidates = lut.cog_cx_pap[matches_within_cog_rectangle]
    cog_cy_pap_candidates = lut.cog_cy_pap[matches_within_cog_rectangle]
    rel_cog_cx = cog_cx_pap_candidates - evt['cog_cx_pap']
    rel_cog_cy = cog_cy_pap_candidates - evt['cog_cy_pap']

    within_radial_distance = (rel_cog_cx**2 + rel_cog_cy**2) <= absolute_cog_radius**2

    return matches_within_cog_rectangle[within_radial_distance]


def match_candidates_based_on_image(
    raw_phs, 
    lut, 
    match_candidates, 
    max_difference=0.4
):
    image = ps.representations.raw_phs_to_image(raw_phs)
    diffs = np.zeros(len(match_candidates))

    for i in range(len(match_candidates)):
        candidate_image = lut.image(match_candidates[i])
        diffs[i] = difference_image(image, candidate_image)

    valid = diffs < max_difference
    return match_candidates[valid], diffs[valid]


def match_candidates_based_on_image_sequence(
    raw_phs, 
    lut, 
    match_candidates, 
    max_difference=0.4
): 
    image_sequence = ps.representations.raw_phs_to_image_sequence(raw_phs)
    diffs = np.zeros(len(match_candidates))

    for i in range(len(match_candidates)):
        candidate_image_sequence = lut.image_sequence(match_candidates[i])
        diffs[i] = difference_image_sequence(
            image_sequence, 
            candidate_image_sequence
        )

    valid = diffs < max_difference
    return match_candidates[valid], diffs[valid]


def match(
    event_features, 
    raw_phs, 
    lut,
    coarse_match_absolute_cog_radius=np.deg2rad(0.2),
    coarse_match_relative_number_photons_radius=0.05,
    imgae_match_max_difference=0.5,
    imgae_sequence_match_max_difference=0.8,
):
    prop = {
        'number_coarse_matches': 0,
        'number_image_matches': 0,
        'number_image_sequence_matches': 0,
    }

    coarse_match_candidates = match_candidates_based_on_coarse_features(
        event_features=event_features, 
        lut=lut,
        absolute_cog_radius=coarse_match_absolute_cog_radius,
        relative_number_photons_radius=coarse_match_relative_number_photons_radius
    )

    if len(coarse_match_candidates) == 0:
        return prop
    prop['number_coarse_matches'] = len(coarse_match_candidates)

    image_match_candidates, image_diffs = match_candidates_based_on_image(
        raw_phs=raw_phs, 
        lut=lut, 
        match_candidates=coarse_match_candidates, 
        max_difference=imgae_match_max_difference
    )

    if len(image_match_candidates) == 0:
        return prop
    prop['number_image_matches'] = len(image_match_candidates)

    # best image match
    # ----------------
    best_match = image_match_candidates[np.argmin(image_diffs)]

    x_pap, y_pap, cx_pap, cy_pap = particle_ray_from_corsika_to_principal_aperture_plane(
        corsika_impact_x=lut.impact_x[best_match],
        corsika_impact_y=lut.impact_y[best_match],
        corsika_phi=lut.phi[best_match],
        corsika_theta=lut.theta[best_match],
        telescope_azimuth_ceres=ceres_azimuth_rad_to_corsika_azimuth_rad(lut.azimuth[best_match]),
        telescope_zenith_ceres=lut.zenith[best_match],
    )
    prop['bim_x_pap'] = x_pap
    prop['bim_y_pap'] = y_pap
    prop['bim_cx_pap'] = cx_pap
    prop['bim_cy_pap'] = cy_pap
    prop['bim_energy'] = lut.energy[best_match]
    prop['bim_best_match'] = best_match
    prop['bim_gammaness'] = 1 - np.min(image_diffs)


    # best image sequence match
    # -------------------------
    
    image_sequence_match_candidates, image_sequence_diffs = match_candidates_based_on_image_sequence(
        raw_phs=raw_phs, 
        lut=lut, 
        match_candidates=image_match_candidates, 
        max_difference=imgae_sequence_match_max_difference
    )

    if len(image_sequence_match_candidates) == 0:
        return prop
    prop['number_image_sequence_matches'] = len(image_sequence_match_candidates)

    best_match = image_sequence_match_candidates[np.argmin(image_sequence_diffs)]
    x_pap, y_pap, cx_pap, cy_pap = particle_ray_from_corsika_to_principal_aperture_plane(
        corsika_impact_x=lut.impact_x[best_match],
        corsika_impact_y=lut.impact_y[best_match],
        corsika_phi=lut.phi[best_match],
        corsika_theta=lut.theta[best_match],
        telescope_azimuth_ceres=ceres_azimuth_rad_to_corsika_azimuth_rad(lut.azimuth[best_match]),
        telescope_zenith_ceres=lut.zenith[best_match],
    )
    prop['bims_x_pap'] = x_pap
    prop['bims_y_pap'] = y_pap
    prop['bims_cx_pap'] = cx_pap
    prop['bims_cy_pap'] = cy_pap
    prop['bims_energy'] = lut.energy[best_match]
    prop['bims_best_match'] = best_match
    prop['bims_gammaness'] = 1 - np.min(image_diffs)


   
    weights = 1.0 - image_sequence_diffs
    weights /= weights.sum()

    energies = lut.energy[image_sequence_match_candidates]
    prop['weighted_energy'] = np.dot(energies, weights)
    prop['median_energy'] = np.median(energies)
    
    """
    impact_xs = np.median(lut.impact_x[image_sequence_match_candidates])
    impact_ys = np.median(lut.impact_y[image_sequence_match_candidates])
    phis = np.median(lut.phi[image_sequence_match_candidates])
    thetas = np.median(lut.theta[image_sequence_match_candidates])
    azimuths = np.median(lut.azimuth[image_sequence_match_candidates])
    zeniths = np.median(lut.zenith[image_sequence_match_candidates])


    x_pap, y_pap, cx_pap, cy_pap = particle_ray_from_corsika_to_principal_aperture_plane(
        corsika_impact_x=impact_xs,
        corsika_impact_y=impact_ys,
        corsika_phi=phis,
        corsika_theta=thetas,
        telescope_azimuth_ceres=azimuths, 
        telescope_zenith_ceres=zeniths,
    )
    prop['x_pap'] = x_pap
    prop['y_pap'] = y_pap
    prop['cx_pap'] = cx_pap
    prop['cy_pap'] = cy_pap
    """


    return prop


def crossvalidate_from_look_up_table(lut, index):
    raw_phs = lut.raw_phs(index)
    event_features = {
        'number_photons': lut.number_photons[index],
        'cog_cx_pap': lut.cog_cx_pap[index],
        'cog_cy_pap': lut.cog_cy_pap[index],
    }
    return event_features, raw_phs


def run_crossvalidation_between_two_lut(lut1, lut2, number_events):
    props = []
    if number_events > lut1.number_events:
        number_events = lut1.number_events
    for index in range(number_events):
        event_features, raw_phs = crossvalidate_from_look_up_table(lut1, index)
        prop = match(event_features, raw_phs, lut2)
        
        prop['true_energy'] = lut1.energy[index]
        x_pap, y_pap, cx_pap, cy_pap = particle_ray_from_corsika_to_principal_aperture_plane(
            corsika_impact_x=lut1.impact_x[index],
            corsika_impact_y=lut1.impact_y[index],
            corsika_phi=lut1.phi[index],
            corsika_theta=lut1.theta[index],
            telescope_azimuth_ceres=ceres_azimuth_rad_to_corsika_azimuth_rad(lut1.azimuth[index]), 
            telescope_zenith_ceres=lut1.zenith[index],
        )
        prop['true_x_pap'] = x_pap
        prop['true_y_pap'] = y_pap
        prop['true_cx_pap'] = cx_pap
        prop['true_cy_pap'] = cy_pap


        props.append(prop)
    return props
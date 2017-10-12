import numpy as np
from .HomTra import HomTra


def corsika_impact_to_ceres_impact(
    impact_x_wrt_corsika,
    impact_y_wrt_corsika,
    az_offset_between_corsika_and_ceres,
):
    az = az_offset_between_corsika_and_ceres
    caz = np.cos(az)
    saz = np.sin(az)

    impact_x_wrt_ceres = caz*impact_x_wrt_corsika - saz*impact_y_wrt_corsika
    impact_y_wrt_ceres = saz*impact_x_wrt_corsika + caz*impact_y_wrt_corsika

    return impact_x_wrt_ceres, impact_y_wrt_ceres


def corsika_az_zd_to_ceres_az_zd(
    source_az_wrt_corsika,
    source_zd_wrt_corsika,
    az_offset_between_corsika_and_ceres,
):
    source_az_wrt_ceres = source_az_wrt_corsika + az_offset_between_corsika_and_ceres
    source_zd_wrt_ceres = source_zd_wrt_corsika
    return source_az_wrt_ceres, source_zd_wrt_ceres


def H_COR2PAP(
    telescope_az, 
    telescope_zd
):
    H = HomTra()
    H.set_rotation_tait_bryan_angles(
        Rx=0,
        Ry=-telescope_zd,
        Rz=-telescope_az
    )
    H.set_translation([0,0,0])
    return H.inverse()


def corsica_trajectory_2_ray(
    impact_x,
    impact_y,
    azimuth,
    zenith_distance
):
    support = np.array([impact_x, impact_y, 0])
    H = HomTra()
    H.set_rotation_tait_bryan_angles(Rx=0, Ry=-zenith_distance, Rz=-azimuth)
    H.set_translation([0,0,0])
    direction = H.transformed_orientation([0,0,1])
    return support, direction


def transform_ray(H, support, direction):
    return H.transformed_position(support), H.transformed_orientation(direction)



def ray_from_corsika_2_pap(
    trajectory_support,
    trajectory_pointing,
    telescope_azimuth_ceres, 
    telescope_zenith_ceres, 
):
    H = H_COR2PAP(telescope_azimuth_ceres, telescope_zenith_ceres)
    return transform_ray(H, trajectory_support, trajectory_pointing)



def ray_xy_plane_intersection(support, direction):
    v = -support[2]/direction[2]
    intersection = support + v*direction
    return intersection


def ray_local_system_to_principal_aperture_plane_system(
    impact_x,
    impact_y,
    source_az,
    source_zd,
    telescope_az, 
    telescope_zd,
):
    support, direction = corsica_trajectory_2_ray(
        impact_x=impact_x,
        impact_y=impact_y,
        azimuth=source_az,
        zenith_distance=source_zd,
    )

    direction /= np.linalg.norm(direction)

    corsika2pap = H_COR2PAP(
        telescope_az=telescope_az, 
        telescope_zd=telescope_zd
    )

    support_pap, direction_pap = transform_ray(
        corsika2pap,
        support=support,
        direction=direction
    )

    direction_pap /= np.linalg.norm(direction_pap)

    impact_pap = ray_xy_plane_intersection(
        support=support_pap, 
        direction=direction_pap,
    )

    impact_x_pap = impact_pap[0]
    impact_y_pap = impact_pap[1]

    cx_pap = direction_pap[0]
    cy_pap = direction_pap[1]

    return impact_x_pap, impact_y_pap, cx_pap, cy_pap


def direction_2_azimuth_zenith_distance(direction):
    d = direction
    zenith_distance = np.arccos(d[2])
    azimuth = np.arctan2(d[1], d[0])
    return azimuth, zenith_distance
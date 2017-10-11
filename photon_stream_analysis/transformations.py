import numpy as np
from .HomTra import HomTra

# COR frame
# PAP frame
# CAM frame


def ceres_azimuth_rad_to_corsika_azimuth_rad(ceres_az_rad):            
    return ceres_az_rad - 3.0 + 2*np.pi # works well
    #return ceres_az_rad + np.pi # works not so well


def ceres_zenith_rad_to_corsika_zenith_rad(ceres_zd_rad):            
    return ceres_zd_rad


def H_COR2PAP(
    telescope_azimuth_ceres, 
    telescope_zenith_ceres
):
    H = HomTra()
    H.set_rotation_tait_bryan_angles(
        Rx=0,
        Ry=-telescope_zenith_ceres,
        Rz=-telescope_azimuth_ceres
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


def particle_ray_from_corsika_to_principal_aperture_plane(
    corsika_impact_x,
    corsika_impact_y,
    corsika_phi,
    corsika_theta,
    telescope_azimuth_ceres, 
    telescope_zenith_ceres,
):
    support, direction = corsica_trajectory_2_ray(
        impact_x=corsika_impact_x,
        impact_y=corsika_impact_y,
        azimuth=corsika_phi,
        zenith_distance=corsika_theta,
    )

    direction /= np.linalg.norm(direction)

    corsika2pap = H_COR2PAP(
        telescope_azimuth_ceres=telescope_azimuth_ceres, 
        telescope_zenith_ceres=telescope_zenith_ceres
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
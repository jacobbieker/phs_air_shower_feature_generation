import photon_stream_analysis as psa
import numpy as np
d2r = np.deg2rad
r2d = np.rad2deg


def test_corsika_trajectory_2_ray():
    supp, dire = psa.transformations.corsica_trajectory_2_ray(
        impact_x=0,
        impact_y=0,
        azimuth=d2r(0),
        zenith_distance=d2r(0),       
    )
    assert supp[0] == 0
    assert supp[1] == 0
    assert supp[2] == 0

    assert dire[0] == 0
    assert dire[1] == 0
    assert dire[2] == 1


def test_corsika_trajectory_2_ray_constant_support():
    for x in np.linspace(-100,100, 11):
        for y in np.linspace(-100,100, 11):
            for az in np.linspace(-180,180, 11):
                for zd in np.linspace(0,90, 11):
                    supp, dire = psa.transformations.corsica_trajectory_2_ray(
                        impact_x=x,
                        impact_y=y,
                        azimuth=d2r(az),
                        zenith_distance=d2r(zd),       
                    )
                    assert supp[0] == x
                    assert supp[1] == y
                    assert supp[2] == 0

                    assert np.isclose(np.linalg.norm(dire), 1.0)


def test_corsika_trajectory_2_ray_direction_1():
    supp, dire = psa.transformations.corsica_trajectory_2_ray(
        impact_x=0,
        impact_y=0,
        azimuth=d2r(0),
        zenith_distance=d2r(90),       
    )
    assert np.isclose(dire[0], 1.0)
    assert np.isclose(dire[1], 0.0)
    assert np.isclose(dire[2], 0.0)


def test_corsika_trajectory_2_ray_direction_2():
    supp, dire = psa.transformations.corsica_trajectory_2_ray(
        impact_x=0,
        impact_y=0,
        azimuth=d2r(90),
        zenith_distance=d2r(90),       
    )
    assert np.isclose(dire[0], 0.0)
    assert np.isclose(dire[1], 1.0)
    assert np.isclose(dire[2], 0.0)


def test_corsika_trajectory_2_ray_direction_3():
    supp, dire = psa.transformations.corsica_trajectory_2_ray(
        impact_x=0,
        impact_y=0,
        azimuth=d2r(180),
        zenith_distance=d2r(90),       
    )
    assert np.isclose(dire[0], -1.0)
    assert np.isclose(dire[1], 0.0)
    assert np.isclose(dire[2], 0.0)


def test_corsika_trajectory_2_ray_direction_4():
    supp, dire = psa.transformations.corsica_trajectory_2_ray(
        impact_x=0,
        impact_y=0,
        azimuth=d2r(270),
        zenith_distance=d2r(90),       
    )
    assert np.isclose(dire[0], 0.0)
    assert np.isclose(dire[1], -1.0)
    assert np.isclose(dire[2], 0.0)


def test_corsika_trajectory_2_pap_1():
    supp, dire = psa.transformations.ray_from_corsika_2_pap(
        trajectory_support=[0,0,0],
        trajectory_pointing=[0,0,1],
        telescope_azimuth_ceres=d2r(0), 
        telescope_zenith_ceres=d2r(0),      
    )
    assert np.isclose(dire[0], 0.0)
    assert np.isclose(dire[1], 0.0)
    assert np.isclose(dire[2], 1.0)


def test_corsika_trajectory_2_pap_2():
    for az in np.linspace(0,360,11):
        supp, dire = psa.transformations.ray_from_corsika_2_pap(
            trajectory_support=[0,0,0],
            trajectory_pointing=[0,0,1],
            telescope_azimuth_ceres=d2r(az), 
            telescope_zenith_ceres=d2r(0),      
        )
        assert np.isclose(dire[0], 0.0)
        assert np.isclose(dire[1], 0.0)
        assert np.isclose(dire[2], 1.0)


def test_corsika_trajectory_2_pap_3():
    for az in np.linspace(0,360,11):
        supp, dire = psa.transformations.ray_from_corsika_2_pap(
            trajectory_support=[0,0,0],
            trajectory_pointing=[0,0,1],
            telescope_azimuth_ceres=d2r(az), 
            telescope_zenith_ceres=d2r(90),      
        )
        assert np.isclose(dire[0], -1.0)
        assert np.isclose(dire[1], 0.0)
        assert np.isclose(dire[2], 0.0)


def test_xy_plane_intersection_1():
    s = np.array([0,0,0])
    d = np.array([0,0,1])
    intersec = psa.transformations.ray_xy_plane_intersection(
        support=s,
        direction=d
    )
    assert np.isclose(intersec[0], 0.0)
    assert np.isclose(intersec[1], 0.0)
    assert np.isclose(intersec[2], 0.0)


def test_xy_plane_intersection_2():
    s = np.array([0,0,1])
    d = np.array([0,1,1])
    intersec = psa.transformations.ray_xy_plane_intersection(
        support=s,
        direction=d
    )
    assert np.isclose(intersec[0], 0.0)
    assert np.isclose(intersec[1], -1.0)
    assert np.isclose(intersec[2], 0.0)


def test_xy_plane_intersection_3():
    s = np.array([1,0,1])
    d = np.array([0,1,1])
    intersec = psa.transformations.ray_xy_plane_intersection(
        support=s,
        direction=d
    )
    assert np.isclose(intersec[0], 1.0)
    assert np.isclose(intersec[1], -1.0)
    assert np.isclose(intersec[2], 0.0)
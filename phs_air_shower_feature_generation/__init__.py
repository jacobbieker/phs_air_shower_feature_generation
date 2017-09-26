import photon_stream as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil

PHYSICS_TRIGGER = 4

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def eigenvectors2rot_matrix(ev0, ev1):
    r = np.array([ev0, ev1])
    if (r == r.T).sum() == 4:
        r[:,1] *= -1
    return r


def extract_leakage(air_shower_photons, edge=0.1):
    d = np.linalg.norm(air_shower_photons[:,0:2], axis=1)
    leaky = d > 0.5*(1.0 - edge)
    return leaky.sum()

def extract_ellipse(air_shower_photons):
    asp = air_shower_photons
    asp_xy = asp[:,0:2]

    center = np.mean(asp_xy.T, axis=1)
    cov_mat = np.cov(asp_xy.T)
    eig_vars, eig_vecs = np.linalg.eig(cov_mat)
    eig_std = np.sqrt(eig_vars)

    order = np.argsort(eig_std)
    order = np.flip(order, axis=0)
    eig_std = eig_std[order]
    eig_vecs = eig_vecs[:,order]

    return {
        'center': center,
        'ev0': eig_vecs[:,0],
        'ev1': eig_vecs[:,1],
        'std0': eig_std[0], 
        'std1': eig_std[1],   
    }


def transform_into_ellipse_frame(air_shower_photons, ellipse):
    asp = air_shower_photons
    el_xyt = asp.copy()
    # translate
    el_xyt[:,0:2] -= ellipse['center']
    # rotate
    ev0 = ellipse['ev0']
    ev1 = ellipse['ev1']
    rot_matrix = eigenvectors2rot_matrix(ev0, ev1).T
    el_xyt[:,0:2] = np.dot(rot_matrix, el_xyt[:,0:2].T).T
    return el_xyt


def time_poly_fit(air_shower_photons_ElFrame, deg=1):
    photon_main_axis = air_shower_photons_ElFrame[:,0]
    photon_arrival_time = air_shower_photons_ElFrame[:,2]
    return np.polyfit(photon_main_axis, photon_arrival_time, deg=deg)


def head_tail_ratio_along_main_axis(air_shower_photons_ElFrame):
    h = np.histogram(air_shower_photons_ElFrame[:,0], bins=2)[0]
    return h[0]/h[1]


def reject_early_or_late_clusters(cluster):
    cluster_arrival_times = np.zeros(cluster.number)
    cluster_sizes = np.zeros(cluster.number)
    for c in range(cluster.number):
        cluster_sizes[c] = (cluster.labels==c).sum()
        cluster_arrival_times[c] = cluster.xyt[cluster.labels==c,2].mean()
    order = np.argsort(cluster_sizes)
    order = np.flip(order, axis=0)
    cluster_arrival_times = cluster_arrival_times[order]
    rejected_clusters = []
    for c in range(cluster.number):
        delay = cluster_arrival_times[c] - cluster_arrival_times[0]
        if np.abs(delay) > 0.75:
            rejected_clusters.append(c)
    accepted_cluster = cluster
    for c in rejected_clusters:
        accepted_cluster.labels[accepted_cluster.labels == c] = -1
        accepted_cluster.number -= 1
    return accepted_cluster


def extract_features(photon_stream, cluster):
    fov_diameter = 2.0*photon_stream.geometry['fov_radius']

    air_shower_photons = cluster.xyt[cluster.labels >= 0]
    asp = air_shower_photons

    leakage = extract_leakage(asp)
    ellipse = extract_ellipse(asp)
    asp_ElFrame = transform_into_ellipse_frame(asp, ellipse)
    time_gradient = time_poly_fit(asp_ElFrame)[0]
    head_tail_ratio = head_tail_ratio_along_main_axis(asp_ElFrame)

    """
    if head_tail_ratio < 1:
        ellipse['ev0'] *= -1
        ellipse['ev1'] *= -1
        head_tail_ratio = 1/head_tail_ratio
        time_gradient *= -1
    """

    return {
        'number_photons': air_shower_photons.shape[0],
        'cog_cx': ellipse['center'][0]*fov_diameter,
        'cog_cy': ellipse['center'][1]*fov_diameter,
        'main_axis_cx': ellipse['ev0'][0],
        'main_axis_cy': ellipse['ev0'][1],
        'length': ellipse['std0']*fov_diameter,
        'width': ellipse['std1']*fov_diameter,
        'time_gradient': time_gradient,
        'head_tail_ratio': head_tail_ratio,
        'leakage': leakage,
    }



def cut_and_extract_features(photon_stream, cluster):
    f = {}
    try:
        cluster = reject_early_or_late_clusters(cluster)
        f = extract_features(photon_stream=photon_stream, cluster=cluster)
        f['extraction'] = 0
    except:
        f['extraction'] = 1
    return f


def extract_from_simulation(path, out_path, mmcs_corsika_path=None):
    event_list = ps.SimulationReader(path, mmcs_corsika_path=mmcs_corsika_path)
    features = []
    for event in event_list:
        cluster = ps.PhotonStreamCluster(event.photon_stream)
        f = cut_and_extract_features(photon_stream=event.photon_stream, cluster=cluster)
        f['type'] = ps.io.binary.SIMULATION_EVENT_TYPE_KEY
        f['az'] = np.deg2rad(event.az)
        f['zd'] = np.deg2rad(event.zd)

        f['run'] = event.simulation_truth.run
        f['event'] = event.simulation_truth.event
        f['reuse'] = event.simulation_truth.reuse

        f['particle'] = event.simulation_truth.air_shower.particle
        f['energy'] = event.simulation_truth.air_shower.energy
        f['theta'] = event.simulation_truth.air_shower.theta
        f['phi'] = event.simulation_truth.air_shower.phi
        f['impact_x'] = event.simulation_truth.air_shower.impact_x(event.simulation_truth.reuse)
        f['impact_y'] = event.simulation_truth.air_shower.impact_y(event.simulation_truth.reuse)
        f['starting_altitude'] = event.simulation_truth.air_shower.starting_altitude
        f['hight_of_first_interaction'] = event.simulation_truth.air_shower.hight_of_first_interaction
        features.append(f)

    passed_trigger = pd.DataFrame(features)
    passed_trigger.to_msgpack(out_path+'.part')
    shutil.move(out_path+'.part', out_path)

    no_trigger = pd.DataFrame(event_list.thrown_events())
    no_trigger.to_msgpack(out_path+'.no_trigger.part')
    shutil.move(out_path+'.no_trigger.part', out_path+'.no_trigger')


def extract_from_observation(path, out_path):
    event_list = ps.EventListReader(path)
    features = []
    for event in event_list:
        if event.observation_info.trigger_type == PHYSICS_TRIGGER:
            cluster = ps.PhotonStreamCluster(event.photon_stream)
            f = cut_and_extract_features(photon_stream=event.photon_stream, cluster=cluster)
            f['type'] = ps.io.binary.OBSERVATION_EVENT_TYPE_KEY
            f['az'] = np.deg2rad(event.az)
            f['zd'] = np.deg2rad(event.zd)

            f['night'] = event.observation_info.night
            f['run'] = event.observation_info.run
            f['event'] = event.observation_info.event

            f['time'] = event.observation_info._time_unix_s + event.observation_info._time_unix_us/1e6
            features.append(f)

    df = pd.DataFrame(features)
    df.to_msgpack(out_path+'.part')
    shutil.move(out_path+'.part', out_path)




"""
event_list = ps.EventListReader('014884.phs.jsonl.gz')


fl = []
for event in event_list:
    cluster = ps.PhotonStreamCluster(event.photon_stream)

    cluster = reject_early_or_late_clusters(cluster)

    f = extract_features(photon_stream=event.photon_stream, cluster=cluster)
    fl.append(f)
    print(event.simulation_truth.event)



many_clusters = [
306,
566,
660,
843,
2500,]
"""


def plot_air_shower_photons_xy(air_shower_photons):
    asp = air_shower_photons
    plt.plot(asp[:,0], asp[:,1], 'o')

def plot_air_shower_photons_xt(air_shower_photons):
    asp = air_shower_photons
    plt.plot(asp[:,0], asp[:,2], 'o')

def plot_air_shower_photons_yt(air_shower_photons):
    asp = air_shower_photons
    plt.plot(asp[:,1], asp[:,2], 'o')


def plot_ellipse(ellipse):
    c = ellipse['center']
    ev0 = ellipse['ev0']
    ev1 = ellipse['ev1']
    std0 = ellipse['std0']
    std1 = ellipse['std1']

    plt.plot(c[0], c[1], 'ro')

    ax0 = ev0*std0
    ax1 = ev1*std1

    plt.plot(
        [c[0], c[0] + ax0[0]], 
        [c[1], c[1] + ax0[1]], 
        'r'
    )
    plt.plot(
        [c[0], c[0] + ax1[0]], 
        [c[1], c[1] + ax1[1]], 
        'g'
    )

    plt.axes().set_aspect('equal')

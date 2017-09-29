import photon_stream as ps
import numpy as np
import pandas as pd
import shutil

from . import features
from . import reject

PHYSICS_TRIGGER = 4


def _raw_features(photon_stream, cluster):
    fov_diameter = 2.0*photon_stream.geometry['fov_radius']

    air_shower_photons = cluster.xyt[cluster.labels >= 0]
    asp = air_shower_photons

    leakage = features.extract_leakage(asp)
    ellipse = features.extract_ellipse(asp)
    asp_ElFrame = features.transform_into_ellipse_frame(asp, ellipse)
    time_gradient = features.time_poly_fit(asp_ElFrame)[0]
    head_tail_ratio = features.head_tail_ratio_along_main_axis(asp_ElFrame)

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


def raw_features(photon_stream, cluster):
    """
    Exception code handling to encode the reason for a failed feature extraction
    """
    f = {}
    try:
        f = _raw_features(photon_stream=photon_stream, cluster=cluster)
        f['extraction'] = 0
    except:
        f['extraction'] = 1
    return f


def from_simulation(phs_path, mmcs_corsika_path=None):
    event_list = ps.SimulationReader(phs_path, mmcs_corsika_path=mmcs_corsika_path)
    features = []
    for event in event_list:
        cluster = ps.PhotonStreamCluster(event.photon_stream)
        cluster = reject.early_or_late_clusters(cluster)
        f = raw_features(photon_stream=event.photon_stream, cluster=cluster)
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

    triggered = _reduce_to_32_bit(pd.DataFrame(features))
    triggered = _reduce_to_32_bit(triggered)

    thrown = pd.DataFrame(event_list.thrown_events())
    thrown = _reduce_to_32_bit(thrown)

    return triggered, thrown


def write_simulation_extraction(triggered, thrown, out_path):
    triggered.to_msgpack(out_path+'.part')
    shutil.move(out_path+'.part', out_path)

    thrown.to_msgpack(out_path+'.thrown.part')
    shutil.move(out_path+'.thrown.part', out_path+'.thrown')


def from_observation(phs_path):
    event_list = ps.EventListReader(phs_path)
    features = []
    for event in event_list:
        if event.observation_info.trigger_type == PHYSICS_TRIGGER:
            cluster = ps.PhotonStreamCluster(event.photon_stream)
            cluster = reject.early_or_late_clusters(cluster)
            f = raw_features(photon_stream=event.photon_stream, cluster=cluster)
            f['type'] = ps.io.binary.OBSERVATION_EVENT_TYPE_KEY
            f['az'] = np.deg2rad(event.az)
            f['zd'] = np.deg2rad(event.zd)

            f['night'] = event.observation_info.night
            f['run'] = event.observation_info.run
            f['event'] = event.observation_info.event

            f['time'] = event.observation_info._time_unix_s + event.observation_info._time_unix_us/1e6
            features.append(f)

    triggered = pd.DataFrame(features)
    triggered = _reduce_to_32_bit(triggered)

    return triggered


def write_observation_extraction(triggered, out_path):
    triggered.to_msgpack(out_path+'.part')
    shutil.move(out_path+'.part', out_path)  


def _reduce_to_32_bit(df):
    for key in df.keys():
        if df[key].dtype == 'float64':
            df[key] = df[key].astype(np.float32)
        if df[key].dtype == 'int64':
            df[key] = df[key].astype(np.int32)
    return df
import numpy as np
import photon_stream as ps


def eigenvectors2rot_matrix(ev0, ev1):
    r = np.array([ev0, ev1])
    if (r == r.T).sum() == 4:
        r[:,1] *= -1
    return r


def extract_leakage(air_shower_photons, edge_ratio=0.1):
    d = np.linalg.norm(air_shower_photons[:,0:2], axis=1)
    leaky = d > ps.GEOMETRY.fov_radius*(1.0 - edge_ratio)
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
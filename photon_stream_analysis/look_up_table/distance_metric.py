import numpy as np
import photon_stream as ps


def difference_image_sequence(ims1, ims2):
    sum1 = (ims1**2).sum()
    sum2 = (ims2**2).sum()
    sum12 = sum1 + sum2
    diff_sum = ((ims1 - ims2)**2).sum()    
    return diff_sum/sum12



"""
def make_neighborhood_matrix(max_distance=np.deg2rad(0.2)):
    x = ps.GEOMETRY.x_angle
    y = ps.GEOMETRY.y_angle
    n = len(x)
    max_distance2 = max_distance**2
    N = np.zeros(shape=(n,n), dtype=np.bool)

    for p in range(n):
        for q in range(n):
            dist2 = (x[p] - x[q])**2 + (y[p] - y[q])**2
            if dist2 < max_distance2:
                N[p,q] = True
    return N


def neighborhood_list(neighborhood_matrix):
    assert neighborhood_matrix.shape[0] == neighborhood_matrix.shape[1]
    n = neighborhood_matrix.shape[0]
    M = []
    idx = np.arange(n)
    for i in range(n):
        M.append(idx[neighborhood_matrix[:,i]])
    return M


def pixel_index(raw):
    idx = np.zeros(ps.io.magic_constants.NUMBER_OF_PIXELS, dtype=np.uint32)
    count = np.zeros(ps.io.magic_constants.NUMBER_OF_PIXELS, dtype=np.uint32)
    pixel_chid = 0
    count_per_pixel = 0
    for i, symbol in enumerate(raw):
        if symbol == ps.io.binary.LINEBREAK:
            idx[pixel_chid] = i
            count[pixel_chid] = count_per_pixel
            pixel_chid += 1
            count_per_pixel = 0
        else:
            count_per_pixel += 1

    idx[1:] = idx[0:ps.io.magic_constants.NUMBER_OF_PIXELS-1]+1
    return idx, count


def matching_distance(raw1, raw2, neighbor_list, deg_over_s=0.35e9, max_dist_deg=0.2):
    cx = ps.GEOMETRY.x_angle
    cy = ps.GEOMETRY.y_angle
    sd = ps.io.magic_constants.TIME_SLICE_DURATION_S
    rad_o_s = np.deg2rad(deg_over_s)
    maxD2 = np.deg2rad(max_dist_deg)**2

    pixidx1, pixcnt1 = pixel_index(raw1)
    pixidx2, pixcnt2 = pixel_index(raw2)
    
    C = 0

    pixel1 = 0
    for s1 in raw1:
        if s1 == ps.io.binary.LINEBREAK:
            pixel1 += 1
        else:
            for pixel2 in neighbor_list[pixel1]:
                start = pixidx2[pixel2]
                end = start + pixcnt2[pixel2]
                for idx2 in np.arange(start, end):
                    s2 = raw2[idx2]

                    x1 = cx[pixel1]
                    y1 = cy[pixel1]
                    t1 = s1*sd*rad_o_s
                    
                    x2 = cx[pixel2]
                    y2 = cy[pixel2]
                    t2 = s2*sd*rad_o_s    
                    D2 = (x1-x2)**2 + (y1-y2)**2 + (t1-t2)**2
                    if D2 < maxD2:
                        C += 1

    return C
"""






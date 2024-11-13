import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import *
import math
from utils import geo_utils
import matplotlib.pyplot as plt

SCALLOP_MAX_SIZE = 0.15

def get_avg_plane_normal(poly):
    NUM_ITTS = 100
    norm_vecs = np.zeros((NUM_ITTS, 3), dtype=np.float64)
    for itt in range(NUM_ITTS):
        pts = poly[np.random.randint(poly.shape[0], size=3)]
        norm_vec = np.cross(pts[0]-pts[1], pts[0]-pts[2])
        norm_vec *= np.sign(norm_vec[2])
        norm_vecs[itt] = norm_vec / (np.linalg.norm(norm_vec) + 1e-64)
    avg_norm_vec = np.mean(norm_vecs, axis=0)
    return avg_norm_vec / np.linalg.norm(avg_norm_vec)

def get_rot_3pt(pts):
    x_vec = pts[0]-pts[1]
    z_vec = np.cross(x_vec, pts[0]-pts[2])
    y_vec = np.cross(z_vec, x_vec)
    mat = np.stack([x_vec, y_vec, z_vec]).T
    mat /= np.linalg.norm(mat, axis=0) + 1e-64
    return mat

def convolve_z(poly, conv_size=21):
    KERNEL = np.ones((conv_size,), dtype=np.float64) / conv_size
    CONV_OFFSET = conv_size // 2
    polyz_extended = np.concatenate([poly[-CONV_OFFSET:, 2], poly[:, 2], poly[:CONV_OFFSET, 2]])
    polyz_convolved = np.convolve(polyz_extended, KERNEL, 'valid')
    poly[:, 2] = polyz_convolved
    return poly

def plane_ransac_filter(poly, max_num_itts=200, threshold_dist=0.002, frac_thresh_cutoff=0.8, scale=1.0, flatten=False):
    highest_consensus_frac = 0
    best_fit_polygon = poly.copy()
    center_pnt = np.mean(poly, axis=0)
    for itt in range(max_num_itts):
        rand_idxs = np.random.randint(poly.shape[0], size=3)
        rand_pnts = poly[rand_idxs]
        rot_mat = get_rot_3pt(rand_pnts)
        if np.linalg.matrix_rank(rot_mat) != 3:
            continue
        inv_mat = rot_mat.T
        poly_principal = np.matmul(inv_mat, (poly - center_pnt).T).T
        good_vert_mask = np.abs(poly_principal[:, 2]) < (threshold_dist / scale)
        consensus_frac = good_vert_mask.sum() / poly.shape[0]
        if consensus_frac > highest_consensus_frac:
            highest_consensus_frac = consensus_frac
            if flatten:
                poly_principal[:, 2] = 0.0
            else:
                poly_principal[np.logical_not(good_vert_mask), 2] = 0.0
            best_fit_polygon = (np.matmul(rot_mat, poly_principal.T).T + center_pnt)
            if highest_consensus_frac > frac_thresh_cutoff:
                break
    return best_fit_polygon

FOV_MUL = 1.2

def pnt_in_cam_fov(pnt, cam_fov, tol_deg=0):
    tol_rad = math.radians(tol_deg)
    vec_zy = pnt * np.array([0, 1, 1])
    vec_zy /= np.linalg.norm(vec_zy) + 1e-9
    vec_zx = pnt * np.array([1, 0, 1])
    vec_zx /= np.linalg.norm(vec_zx) + 1e-9
    angle_x = math.acos(vec_zx[2])
    angle_y = math.acos(vec_zy[2])
    angle_thresh = FOV_MUL * cam_fov / 2
    return angle_x < angle_thresh[0] + tol_rad and angle_y < angle_thresh[1] + tol_rad

def in_camera_fov(polygon_cam, cam_fov, tol_deg=10):
    polygon_center = np.mean(polygon_cam, axis=1)
    return pnt_in_cam_fov(polygon_center, cam_fov, tol=tol_deg)

PC_MUL = 1.9
def polygon_PCA_width(polygon):
    pc_vecs, pc_lengths, center_pnt = pca(polygon)
    pc_lengths = np.sqrt(pc_lengths) * PC_MUL
    scaled_pc_lengths = pc_lengths * 2
    return scaled_pc_lengths[0]

def polygon_max_width(polygon):
    scallop_vert_mat = np.repeat(polygon[None], polygon.shape[0], axis=0)
    scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
    return np.max(scallop_vert_dists)

def cluster_avg_polygon(cluster, polygon_scores):
    cluster_linstrings = [LineString(np.concatenate([poly, poly[:1]], axis=0)) for poly in cluster]
    c = np.concatenate(cluster, axis=0).mean(axis=0)[:2]
    avg_points = []
    radial_errs = []
    for theta in np.linspace(0, 2*np.pi, num=100):
        ray = LineString([c, c + 0.2 * np.array([np.sin(theta), np.cos(theta)])])
        intersection_pnt_sum = np.zeros((3,))
        score_sum = 1e-32
        for poly, score in zip(cluster_linstrings, polygon_scores):
            int_pnts = ray.intersection(poly)
            if isinstance(int_pnts, Point):
                intersection_pnt_sum += score * np.array(int_pnts.coords[0])
                score_sum += score
        avg_points.append(intersection_pnt_sum / score_sum)
    return np.array(avg_points)

def calc_cluster_widths(polygon_clusters, mode=None):
    cluster_widths_l = []
    if mode == 'max':
        sz_func = polygon_max_width
    elif mode == 'pca':
        sz_func = polygon_PCA_width
    else:
        sz_func = lambda poly: (polygon_max_width(poly) + polygon_PCA_width(poly)) / 2
    for cluster in tqdm(polygon_clusters):
        cluster_poly_width = np.mean([sz_func(poly) for poly in cluster])
        cluster_poly_width = min(float(cluster_poly_width), SCALLOP_MAX_SIZE)
        cluster_widths_l.append(cluster_poly_width)
    return cluster_widths_l

def get_next_seed_index(mask_arr):
    for i, val in enumerate(mask_arr):
        if val == True:
            return i

def rnn_clustering(point_groups, rnn_distance):
    unclustered_mask = np.ones((len(point_groups),)).astype(bool)
    neighbourhood_mask = unclustered_mask.copy()
    centers = np.array([np.mean(poly[:, :2], axis=0) for poly in point_groups])
    cluster_indexes = []
    while any(unclustered_mask):
        seed_center = centers[get_next_seed_index(unclustered_mask)]
        for i in range(2):
            unclst_dists = np.linalg.norm(centers - seed_center, axis=1)
            neighbourhood_mask = (unclst_dists < rnn_distance) * unclustered_mask
            seed_center = np.mean(centers[neighbourhood_mask], axis=0)
        neighbour_idxs = np.where(neighbourhood_mask)[0]
        cluster_indexes.append(neighbour_idxs)
        unclustered_mask[neighbour_idxs] = False
    return cluster_indexes

def UpsamplePoly(polygon, num=10):
    poly_ext = np.append(polygon, [polygon[0, :]], axis=0)
    up_poly = []
    for idx in range(poly_ext.shape[0]-1):
        int_points = np.linspace(poly_ext[idx], poly_ext[idx+1], num=num)
        up_poly.extend(int_points)
    return np.array(up_poly)

def Project2Img(points, cam_mtx, dist):
    result = []
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    if len(points) > 0:
        result, _ = cv2.projectPoints(points, rvec, tvec,
                                      cam_mtx, dist)
    return np.squeeze(result, axis=1)

def undistort_pixels(pixels, cam_mtx, dist):
    pix_ud = np.array([])
    if len(pixels) > 0:
        pix_ud = cv2.undistortPoints(np.expand_dims(pixels.astype(np.float32), axis=1), cam_mtx, dist, P=cam_mtx)
    return np.squeeze(pix_ud, axis=1)

def pca(points):
    u = np.mean(points, axis=0)
    cov_mat = np.cov(points - u, rowvar=False)
    assert not np.isnan(cov_mat).any() and not np.isinf(cov_mat).any() and check_symmetric(cov_mat)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sort_indices]
    pc_lengths = eig_vals[sort_indices]
    return eig_vecs, pc_lengths, u

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def remove_outliers(pnts, radius):
    mean_distances = np.linalg.norm(pnts - np.mean(pnts, axis=1)[:, None], axis=0)
    inlier_mask = mean_distances < radius
    return pnts[:, inlier_mask]

def polyline_dist_thresh(pnt, polyline, thresh):
    for seg_idx in range(len(polyline)-1):
        line_seg = [polyline[seg_idx+1], polyline[seg_idx]]
        dist = pnt2lineseg_dist(pnt, line_seg)
        if dist < thresh:
            return True
    return False

def get_poly_arr_2d(poly):
    return np.array(poly.exterior.coords)[:, :2]

def get_local_poly_arr(poly_gps):
    poly_arr_2d = get_poly_arr_2d(poly_gps)
    poly_arr_m = geo_utils.convert_gps2local(poly_arr_2d[0], poly_arr_2d)
    return poly_arr_m

def get_local_poly_arr_3D(poly_gps):
    poly_arr_m = geo_utils.convert_gps2local(poly_gps[0], poly_gps)
    return poly_arr_m

def get_poly_area_m2(poly_gps):
    return Polygon(get_local_poly_arr(poly_gps)).area

def pnt2lineseg_dist(point, line):
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
            np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
            np.linalg.norm(unit_line)
    )

    diff = (
            (norm_unit_line[0] * (point[0] - line[0][0])) +
            (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        return 10.0
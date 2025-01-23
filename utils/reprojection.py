import numpy as np
from utils import polygon_functions as spf
from utils import geo_utils

CAM_PROXIMITY_THRESH = 0.3  # m
ELEV_MEAN_PROX_THRESH = 0.05

def CamToChunk(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def CamPixToRay(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]


gcs2ccs = lambda pnt: geo_utils.geocentric_to_geodetic(pnt[0], pnt[1], pnt[2])

def PixToGeodedic(polygon_ud, vert_elevations, cam_mtx, cam_q44, chunk_transform):
    scallop_poly_cam = CamPixToRay(polygon_ud.T, cam_mtx) * vert_elevations.T
    scallop_polygon_chunk = CamToChunk(scallop_poly_cam, cam_q44)
    scallop_polygon_geocentric = TransformPoints(scallop_polygon_chunk, chunk_transform)
    scallop_polygon_geodetic = np.apply_along_axis(gcs2ccs, 1, scallop_polygon_geocentric.T)[:, :2]
    return scallop_polygon_geodetic

def reproject_polygon(pixels, cam_mtx, cam_dist, cam_q44, depth_map, chunk_scale, chunk_transform):
    # Undistort polygon vertices
    scallop_polygon_ud = spf.undistort_pixels(pixels, cam_mtx, cam_dist).astype(np.int32)
    vert_elevations = depth_map[scallop_polygon_ud[:, 1], scallop_polygon_ud[:, 0]]
    # Threshold out points close to camera
    # TODO: check depth img coverage around pixel coordinates
    # TODO: estimate depth values where the depth image is invalid - ransac method or take neighbouring points??
    valid_indices = np.where(vert_elevations > (CAM_PROXIMITY_THRESH / chunk_scale))
    if len(valid_indices[0]) < 10:
        return None
    vert_elevation_mean = np.mean(vert_elevations[valid_indices])
    valid_indices = np.where(
        np.abs(vert_elevations - vert_elevation_mean) < (ELEV_MEAN_PROX_THRESH / chunk_scale))
    if len(valid_indices[0]) < 10:
        return None
    vert_elevations = vert_elevations[valid_indices]
    scallop_polygon_ud = scallop_polygon_ud[valid_indices]
    return PixToGeodedic(scallop_polygon_ud, vert_elevations, cam_mtx, cam_q44, chunk_transform)
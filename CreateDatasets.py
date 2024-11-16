import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
from detectron2.structures import BoxMode
from utils import polygon_functions as spf
from utils import vpz_utils, tiff_utils, geo_utils, file_utils
import geopandas as gpd
import pickle
import json
from tqdm import tqdm
import glob
import math
from shapely.geometry import *

DISPLAY = False
WAITKEY = 0

CAM_COV_THRESHOLD = 0.02

CAM_SCALLOP_DIST_THRESH = 2.0
CAM_SCALLOP_Z_THRESH = 2.0

PIX_EDGE_THRESH_CORNER = 10

BOUNDARY_BUFFER_DIST_M = 0.1

IMG_SHAPE = (2056, 2464)
IMG_RS_MOD = 2
CNN_INPUT_SHAPE = (IMG_SHAPE[0] // IMG_RS_MOD, IMG_SHAPE[1] // IMG_RS_MOD)
NON_SCALLOP_DOWNSAMPLE = 50  # 10

PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_annotation_log.txt'

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]


def create_dataset(dirname):
    print(f"\n--------------- Processing {dirname} Annotations ---------------")
    data_dir = PROCESSED_BASEDIR + dirname
    dataset_name = 'dataset-' + dirname[:-1]
    file_utils.ensure_dir_exists(data_dir + dataset_name, clear=not DISPLAY)

    if DISPLAY:
        cv2.namedWindow("Annotated img", cv2.WINDOW_GUI_NORMAL)

    print("Loading Chunk Telemetry")
    with open(data_dir + "chunk_telemetry.pkl", "rb") as pkl_file:
        chunk_telem = pickle.load(pkl_file)
        chunk_scale = chunk_telem['0']['scale']
        chunk_transform = chunk_telem['0']['transform']
        chunk_inv_transform = np.linalg.inv(chunk_transform)

    print("Loading Camera Telemetry")
    with open(data_dir + "camera_telemetry.pkl", "rb") as pkl_file:
        camera_telem = pickle.load(pkl_file)

    print("Importing shapes from gpkgs and .vpz")
    # TODO: check for annotations overlap??
    # TODO: check for same file and
    shape_layers = []
    # shape_fpaths = glob.glob(data_dir + 'shapes_ann/*Poly*.gpkg')
    # for shape_path in shape_fpaths:
    #     shape_gdf = gpd.read_file(shape_path)
    #     shapes_label = shape_path.split('/')[-1].split('.')[0]
    #     print(f"Importing {shapes_label} from files")
    #     shape_layers.append(shape_gdf)
    shape_layers_vpz = vpz_utils.get_shape_layers_gpd(data_dir, data_dir.split('/')[-2] + '.vpz')
    include_polygons = []
    exclude_polygons = []
    for layer_label, shape_layer in shape_layers_vpz:
        if layer_label in ['Include Areas', 'Exclude Areas']:
            inc_bound = layer_label == 'Include Areas'
            boundary_list = include_polygons if inc_bound else exclude_polygons
            for poly in shape_layer.geometry:
                offset_poly = poly.buffer(BOUNDARY_BUFFER_DIST_M / 111e3 * (-1 if inc_bound else 1), join_style=2)
                boundary_list.extend(offset_poly.geoms if isinstance(offset_poly, MultiPolygon) else [offset_poly])
        if 'poly' in layer_label.lower():
            shape_layers.append(shape_layer)

    # test_write_gdf = gpd.GeoDataFrame({'NAME': 'test', 'geometry': include_polygons}, geometry='geometry')
    # test_write_gdf.to_file('/csse/users/tkr25/Desktop/include_shrunk.geojson', driver='GeoJSON')

    if len(shape_layers) == 0:
        print(f"No annotation shape layers of files found! Ignoring {dirname}")
        return

    print("Initialising DEM Reader")
    dem_obj = tiff_utils.DEM(data_dir + 'geo_tiffs/')
    # Transformer from geographic/geodetic coordinates to geocentric
    ccs2gcs = lambda pnt: geo_utils.geodetic_to_geocentric(pnt[1], pnt[0], pnt[2])
    gcs2ccs = lambda pnt: geo_utils.geocentric_to_geodetic(pnt[0], pnt[1], pnt[2])
    print("Getting 3D polygons using DEMs")
    polygons_chunk = []
    for shape_gdf in shape_layers:
        for i, row in shape_gdf.iterrows():
            if isinstance(row.geometry, Polygon):
                poly_2d = np.array(row.geometry.exterior.coords)[:, :2]
                if poly_2d.shape[0] < 5:
                    continue
                poly_2d = spf.UpsamplePoly(poly_2d, 100 // poly_2d.shape[0])
                poly_3d = dem_obj.poly3d_from_dem(poly_2d)
                poly_3d_conv = spf.convolve_z(poly_3d.copy())
                poly_3d_geocentric = np.apply_along_axis(ccs2gcs, 1, poly_3d_conv)
                poly_3d_chunk = TransformPoints(poly_3d_geocentric.T, chunk_inv_transform).T
                poly_3d_chunk_flt = spf.plane_ransac_filter(poly_3d_chunk, scale=chunk_scale, flatten=True)
                # poly_center = poly_3d_chunk.mean(axis=0)
                # poly_3d_chunk = poly_3d_chunk - poly_center
                # poly_3d_chunk_flt = poly_3d_chunk_flt - poly_center
                # poly_geo = np.apply_along_axis(ccs2gcs, 1, poly_3d)
                # poly_chunk = TransformPoints(poly_geo.T, chunk_inv_transform).T - poly_center
                # ax = plt.axes(projection='3d')
                # ax.plot3D(poly_chunk[:, 0], poly_chunk[:, 1], poly_chunk[:, 2], 'r')
                # ax.plot3D(poly_3d_chunk[:, 0], poly_3d_chunk[:, 1], poly_3d_chunk[:, 2], 'g')
                # ax.plot3D(poly_3d_chunk_flt[:, 0], poly_3d_chunk_flt[:, 1], poly_3d_chunk_flt[:, 2], 'b')
                # # ax.scatter3D(poly_3d_chunk_filt[:, 0], poly_3d_chunk_filt[:, 1], poly_3d_chunk_filt[:, 2], 'b')
                # BOX_MINMAX = [-0.07 / chunk_scale, 0.07 / chunk_scale]
                # ax.auto_scale_xyz(BOX_MINMAX, BOX_MINMAX, BOX_MINMAX)
                # plt.show()
                polygons_chunk.append(poly_3d_chunk_flt)
    print("Number of valid scallop polygons in chunk: {}".format(len(polygons_chunk)))

    img_id = 0
    skip_num = 0
    label_dict = []
    for cam_label, cam_telem in tqdm(camera_telem.items()):
        cam_quart = cam_telem['q44']
        cam_quart_inv = np.linalg.inv(cam_quart)
        cam_cov = cam_telem['loc_cov33']
        xyz_cov_mean = cam_cov[(0, 1, 2), (0, 1, 2)].mean()
        # Check camera accuracy

        print("Cam COV xyz:", xyz_cov_mean * chunk_scale**2)

        if xyz_cov_mean > CAM_COV_THRESHOLD / chunk_scale**2:
            continue

        # Check camera center is inside include area and outside all exclude areas (with buffer)
        camloc_chunk = cam_quart[:3, 3][:, None]
        camloc_geocentric = TransformPoints(camloc_chunk, chunk_transform)
        camloc_geodetic_pnt = Point(np.apply_along_axis(gcs2ccs, 1, camloc_geocentric.T)[0, :2])
        if len(include_polygons) and not any(camloc_geodetic_pnt.within(inc_poly) for inc_poly in include_polygons):
            continue
        if any(camloc_geodetic_pnt.within(exc_poly) for exc_poly in exclude_polygons):
            continue

        height, width = cam_telem['shape']
        img_path_rel = cam_telem['cpath']
        img_path = data_dir + img_path_rel
        camMtx = cam_telem['cam_mtx']
        camMtx[:2, :] /= IMG_RS_MOD
        camDist = cam_telem['cam_dist']
        cam_fov = cam_telem['cam_fov']

        # img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
        # img_cam_und = cv2.undistort(img, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)

        # Deletion of image regions not in orthomosaic
        # valid_pixel_mask = np.zeros_like(img)
        # row_indices, col_indices = np.indices((img_shape[0], img_shape[1]))[:, ::20, ::20]
        # pnts_uv = np.array([col_indices.flatten(), row_indices.flatten(), col_indices.size*[1]], dtype=np.float32)
        # cam_rays = np.matmul(np.linalg.inv(camMtx), pnts_uv)
        # cam_pnts = TransformPoints(cam_rays, cam_quart)
        # for cam_pnt in cam_pnts.transpose():
        #     cam_origin = Metashape.Vector(cam_quart[:3, 3])
        #     ray_pnt = Metashape.Vector(cam_pnt)
        #     closest_intersection = chunk_densepoints.pickPoint(cam_origin, ray_pnt)
        #     if closest_intersection is not None:
        #         if (cam_origin - closest_intersection).norm() < 10:
        #             img_pixels = cam.project(closest_intersection)
        #             if img_pixels is not None:
        #                 valid_pix_int = np.array(img_pixels, dtype=np.int).reshape((2,))
        #                 cv2.circle(valid_pixel_mask, tuple(valid_pix_int), 100, (1, 1, 1), -1)
        # img_cam_und_roi *= valid_pixel_mask

        objs = []
        display_polygon_l = []
        display_bxs = []
        scallop_in_img = False
        for scallop_id, polygon in enumerate(polygons_chunk):
            polygon_center_cam = TransformPoints(polygon.mean(axis=0)[:, None], cam_quart_inv).T[0]
            too_far_from_cam = abs(polygon_center_cam[2]) > 1.5 / chunk_scale
            if not spf.pnt_in_cam_fov(polygon_center_cam, cam_fov, tol_deg=0) or too_far_from_cam:
                continue
            polygon_cam = TransformPoints(polygon.T, cam_quart_inv)
            pix_coords = spf.Project2Img(polygon_cam, camMtx, camDist).astype(int).T
            valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < width) *
                                                      (pix_coords[1, :] >= 0) * (pix_coords[1, :] < height))][:, 0, :]
            if valid_pix_coords.shape[1] < 3:
                continue
            scallop_in_img = True

            # Fill in corner detections where it would cut
            is_edge = np.less(valid_pix_coords, PIX_EDGE_THRESH_CORNER) + \
                      np.greater(valid_pix_coords, np.array([[width - PIX_EDGE_THRESH_CORNER],
                                                             [height - PIX_EDGE_THRESH_CORNER]]))
            needs_corner = np.logical_xor(is_edge[0, :-1], is_edge[0, 1:]) * \
                           np.logical_xor(is_edge[1, :-1], is_edge[1, 1:]) * \
                           np.logical_xor(is_edge[0, :-1], is_edge[1, :-1])
            if np.any(needs_corner):
                corner_index = np.argmax(needs_corner) + 1
                corner_pix = valid_pix_coords[:, corner_index]
                corner_pix = np.round(corner_pix / np.array([width, height])) * np.array([width, height])
                valid_pix_coords = np.insert(valid_pix_coords, corner_index, corner_pix, axis=1)

            display_polygon_l.append(valid_pix_coords.transpose())
            x_min, y_min = np.min(valid_pix_coords, axis=1).tolist()
            x_max, y_max = np.max(valid_pix_coords, axis=1).tolist()
            display_bxs.append([[x_min, y_min], [x_max, y_max]])
            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [valid_pix_coords.transpose().tolist()],
                "category_id": 0,
                "id": scallop_id,
                "iscrowd": 0,
                "name": 'scallop',
            }
            objs.append(obj)

        if skip_num < NON_SCALLOP_DOWNSAMPLE and not scallop_in_img:
            skip_num += 1
            continue
        skip_num = 0

        img_fn = cam_label + '.jpeg'
        img_ds_fpath = data_dir + dataset_name + '/' + img_fn
        cimg = cv2.imread(img_path)
        cimg_rs = cv2.resize(cimg, CNN_INPUT_SHAPE[::-1])
        cv2.imwrite(img_ds_fpath, cimg_rs)

        record = {}
        record["file_name"] = img_fn
        record["height"] = height // IMG_RS_MOD
        record["width"] = width // IMG_RS_MOD
        record["image_id"] = img_id
        img_id += 1
        record["annotations"] = objs
        label_dict.append(record)

        if DISPLAY:  # and len(display_polygon_l):
            drawing = cimg_rs
            for polygon in display_polygon_l:
                cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=1)
            for box_pt1, box_pt2 in display_bxs:
                cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 1)
            cv2.imshow("Annotated img", drawing)
            key = cv2.waitKey(WAITKEY)
            if key == ord('b'):
                break
            if key == ord('q'):
                exit(0)

    if not DISPLAY:
        print("Saving annotations json...")
        with open(data_dir + dataset_name + "/scallop_dataset.json", 'w') as fp:
            json.dump(label_dict, fp)

    return True


if __name__ == '__main__':
    datasets_created = []
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for dir_line in data_dirs[24:]:
        if 'STOP' in dir_line:
            break
        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line:
            continue
        # if not 'second done' in dir_line.lower():
        #     continue
        data_dir = dir_line.split(' ')[0][:13] + '/'
        ret = create_dataset(data_dir)
        if ret:
            datasets_created.append(data_dir[:-1])

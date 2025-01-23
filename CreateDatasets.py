import numpy as np
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

POLY_DEM_UPSAMPLE = 100

CAM_COV_THRESHOLD = 0.01
CAM_SCALLOP_DIST_THRESH = 2.0
BOUNDARY_BUFFER_DIST_M = 0.1

IMG_SHAPE = (2056, 2464)
IMG_RS_MOD = 2
CNN_INPUT_SHAPE = (IMG_SHAPE[0] // IMG_RS_MOD, IMG_SHAPE[1] // IMG_RS_MOD)
NON_SCALLOP_DOWNSAMPLE = 50  # 10

FRAME_POLY_SHPLY = Polygon([[0, 0], [CNN_INPUT_SHAPE[1]-1, 0], [CNN_INPUT_SHAPE[1]-1, CNN_INPUT_SHAPE[0]-1],
                            [0, CNN_INPUT_SHAPE[0]-1]])  # XY

PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
ANN_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_annotation_log.txt'

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
                # TODO: replace with img section masking
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
                poly_2d = spf.UpsamplePoly(poly_2d, max(1, POLY_DEM_UPSAMPLE // poly_2d.shape[0]))
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
                if poly_3d_chunk_flt.shape[0] > 3:
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
        if xyz_cov_mean > CAM_COV_THRESHOLD / chunk_scale**2:
            # print("Bad camera covariance, skipping.")
            continue
        height = cam_telem['shape'][0] // IMG_RS_MOD
        width = cam_telem['shape'][1] // IMG_RS_MOD
        img_path_rel = cam_telem['cpath']
        img_path = data_dir + img_path_rel
        camMtx = cam_telem['cam_mtx']
        camMtx[:2, :] /= IMG_RS_MOD
        camDist = cam_telem['cam_dist']
        cam_fov = cam_telem['cam_fov']

        # TODO: replace this with image region masking...
        # Check camera center is inside include area and outside all exclude areas (with buffer)
        camloc_chunk = cam_quart[:3, 3][:, None]
        camloc_geocentric = TransformPoints(camloc_chunk, chunk_transform)
        camloc_geodetic_pnt = Point(np.apply_along_axis(gcs2ccs, 1, camloc_geocentric.T)[0, :2])
        if len(include_polygons) and not any(camloc_geodetic_pnt.within(inc_poly) for inc_poly in include_polygons):
            continue
        if any(camloc_geodetic_pnt.within(exc_poly) for exc_poly in exclude_polygons):
            continue

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
            distance_to_cam = np.linalg.norm(polygon[::10].mean(axis=0) - cam_quart[:3, 3])
            if distance_to_cam > CAM_SCALLOP_DIST_THRESH / chunk_scale:
                continue
            polygon_cam = TransformPoints(polygon.T, cam_quart_inv)
            pix_coords = spf.Project2Img(polygon_cam, camMtx, camDist).astype(np.int32)
            pix_poly_shply = Polygon(pix_coords).buffer(0)
            if not pix_poly_shply.intersects(FRAME_POLY_SHPLY):
                continue
            intersection_geoms = pix_poly_shply.intersection(FRAME_POLY_SHPLY)
            if isinstance(intersection_geoms, Polygon):
                intersection_geoms = [intersection_geoms]
            elif isinstance(intersection_geoms, GeometryCollection) or isinstance(intersection_geoms, MultiPolygon):
                intersection_geoms = [geom for geom in intersection_geoms.geoms if isinstance(geom, Polygon)]
            elif isinstance(intersection_geoms, MultiLineString):
                intersection_geoms = [Polygon(line_str) for line_str in intersection_geoms.geoms if len(line_str.coords) > 2]
            elif isinstance(intersection_geoms, LineString) and len(intersection_geoms.coords) > 2:
                intersection_geoms = [Polygon(intersection_geoms)]
            else:
                continue

            for valid_poly in intersection_geoms:
                valid_pix_coords = np.array(valid_poly.exterior.coords, dtype=np.int32)
                scallop_in_img = True
                display_polygon_l.append(np.concatenate([valid_pix_coords, valid_pix_coords[:1]]))
                x_min, y_min = np.min(valid_pix_coords, axis=0).tolist()
                x_max, y_max = np.max(valid_pix_coords, axis=0).tolist()
                display_bxs.append([[x_min, y_min], [x_max, y_max]])
                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [valid_pix_coords.tolist()],
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
        if not DISPLAY:
            cv2.imwrite(img_ds_fpath, cimg_rs)

        record = {}
        record["file_name"] = img_fn
        record["height"] = height
        record["width"] = width
        record["image_id"] = img_id
        img_id += 1
        record["annotations"] = objs
        label_dict.append(record)

        if DISPLAY and len(display_polygon_l) > 0:
            drawing = cimg_rs
            for polygon in display_polygon_l:
                cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
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
    with open(ANN_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for dir_line in data_dirs:
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

    print(datasets_created)
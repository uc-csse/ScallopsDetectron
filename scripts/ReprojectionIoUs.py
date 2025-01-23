import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import polygon_functions as spf
from utils import vpz_utils, tiff_utils, geo_utils, reprojection
import pickle
import json
from tqdm import tqdm
from shapely.geometry import *

DISPLAY_HISTS = True
DISPLAY_POLYS = True
if DISPLAY_POLYS:
    cv2.namedWindow("Annotated Dataset Img", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Annotated Dataset DImg", cv2.WINDOW_GUI_NORMAL)
    plt.ion()
    plt.show()

PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
ANN_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_annotation_log.txt'

IMG_RS_MOD = 2
EDGE_LIMIT_PIX = 10
OUTLIER_RADIUS = 0.1
CAM_PROXIMITY_THRESH = 0.3  # m
ELEV_MEAN_PROX_THRESH = 0.05


def CamToChunk(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]

def CamPixToRay(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))


gcs2ccs = lambda pnt: geo_utils.geocentric_to_geodetic(pnt[0], pnt[1], pnt[2])
ccs2gcs = lambda pnt: geo_utils.geodetic_to_geocentric(pnt[1], pnt[0], pnt[2])

def iou_hist(ious_list):
    plt.figure()
    counts, bins = np.histogram(ious_list, bins=100)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title("IoU of re-projected dataset labels" +
              f"\nTotal Reprojected Count = {len(ious_list)}" +
              f"\nAverage IoU = {round(np.mean(ious_list), 2)}")
    plt.ylabel("Count")
    plt.xlabel("IoU")
    plt.show()

def pixel_gps_IoU(dirname):
    print(f"\n--------------- Processing {dirname} Annotation IoU ---------------")
    data_dir = PROCESSED_BASEDIR + dirname

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
    shape_layers = []
    shape_layers_vpz = vpz_utils.get_shape_layers_gpd(data_dir, data_dir.split('/')[-2] + '.vpz')
    for layer_label, shape_layer in shape_layers_vpz:
        if 'poly' in layer_label.lower():
            shape_layers.append(shape_layer)
    if len(shape_layers) == 0:
        print(f"No annotation shape layers or files found! Ignoring {dirname}")
        return
    assert len(shape_layers) == 1
    print("Initialising DEM Reader")
    dem_obj = tiff_utils.DEM(data_dir + 'geo_tiffs/')
    # Get annotation scallop centers
    ann_scallop_polys = []
    for i, row in shape_layers[0].iterrows():
        if isinstance(row.geometry, Polygon):
            poly_geodedic_2d = np.array(row.geometry.exterior.coords)[:, :2]
            ann_scallop_polys.append(poly_geodedic_2d)
    ann_scallop_centers = np.array([p.mean(axis=0) for p in ann_scallop_polys])

    # Load created dataset
    with open(PROCESSED_BASEDIR + dirname + "dataset-" + dirname + "/scallop_dataset.json") as ds_file:
        dataset_json = json.load(ds_file)

    reproj_scallop_ious = []
    for img_entry in tqdm(dataset_json):
        img_label = img_entry['file_name'].split('/')[-1][:-5]
        frame_telem = camera_telem[img_label]
        height, width = img_entry['height'], img_entry['width']
        cam_mtx = frame_telem['cam_mtx']
        cam_mtx[:2, :] /= IMG_RS_MOD
        cam_dist = frame_telem['cam_dist']
        cam_q44 = frame_telem['q44']
        dimg_path = frame_telem['dpath']
        depth_img_path = data_dir + dimg_path
        col_img_path = data_dir + "dataset-" + dirname + img_entry['file_name']
        if '.npy' in dimg_path:
            img_depth_np = np.load(PROCESSED_BASEDIR + dimg_path)
        else:
            img_depth_u16 = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
            if img_depth_u16 is None:
                continue
            scale = float(depth_img_path.split('/')[-1].split('_')[-1][:-4])
            img_depth_np = scale * img_depth_u16.astype(np.float32)

        anns = img_entry['annotations']
        for ann in anns:
            scallop_ann = np.array(ann['segmentation'][0])  # xy
            max_pix_inds = np.max(scallop_ann, axis=0)
            min_pix_inds = np.min(scallop_ann, axis=0)
            if any(min_pix_inds < 0) or max_pix_inds[0] >= width or max_pix_inds[1] >= height:
                print("\nError! Ann indexes outside image bounds!")
                print("Min:", min_pix_inds, "Max:", max_pix_inds, "Width:", width, "Height:", height)
                continue
            if (any(min_pix_inds < EDGE_LIMIT_PIX) or
                    max_pix_inds[0] >= width - EDGE_LIMIT_PIX or max_pix_inds[1] >= height - EDGE_LIMIT_PIX):
                continue

            scallop_polygon_geodetic = reprojection.reproject_polygon(scallop_ann, cam_mtx, cam_dist, cam_q44,
                                                                      img_depth_np, chunk_scale, chunk_transform)
            if scallop_polygon_geodetic is None:
                reproj_scallop_ious.append(0.0)
                print("\nNo valid depth points!")
                continue

            scallop_center_gps = np.mean(scallop_polygon_geodetic, axis=0)
            # print(scallop_center_chunk)
            distances = np.linalg.norm(ann_scallop_centers - scallop_center_gps, axis=1)
            dist_args = np.argsort(distances)[:5]
            min_dist_idx = dist_args[0]
            ortho_scallop_match = ann_scallop_polys[min_dist_idx]

            scallop_img_local = geo_utils.convert_gps2local(scallop_center_gps, scallop_polygon_geodetic)
            scallop_ortho_local = geo_utils.convert_gps2local(scallop_center_gps, ortho_scallop_match)

            distance = distances[min_dist_idx] * 111e3
            scallop_img_local_shply = Polygon(scallop_img_local).buffer(0)
            scallop_ortho_local_shply = Polygon(scallop_ortho_local)
            if scallop_img_local_shply.is_valid and scallop_img_local_shply.intersects(scallop_ortho_local_shply):
                intersection = scallop_img_local_shply.intersection(scallop_ortho_local_shply).area
                union = scallop_img_local_shply.union(scallop_ortho_local_shply).area
                scallop_iou = intersection / union
            else:
                scallop_iou = 0.0
            reproj_scallop_ious.append(scallop_iou)

            if DISPLAY_POLYS and scallop_iou < 0.8 and len(scallop_ann):
                print("\nDistance:", round(distance, 4), "IoU:", round(scallop_iou, 2))
                plt.cla()
                plt.plot(scallop_img_local[:, 0], scallop_img_local[:, 1])
                plt.plot(scallop_ortho_local[:, 0], scallop_ortho_local[:, 1])
                plt.legend(["Dataset Scallop re-projected", "Annotated Ortho Scallop"])
                plt.draw()
                plt.pause(0.001)
                cimg = cv2.imread(col_img_path)
                dimg_u8 = (20 * np.clip(3 * (img_depth_np * chunk_scale - 0.8), 0, 1.0) * 255).astype(np.uint8)
                cdimg = cv2.applyColorMap(dimg_u8, cv2.COLORMAP_JET)
                scallop_ann_ud = spf.undistort_pixels(scallop_ann, cam_mtx, cam_dist).astype(np.int32)
                cv2.polylines(cdimg, [scallop_ann_ud], False, (0, 0, 0), thickness=2)
                cv2.polylines(cimg, [scallop_ann], False, (0, 255, 0), thickness=1)
                cv2.imshow("Annotated Dataset Img", cimg)
                cv2.imshow("Annotated Dataset DImg", cdimg)
                cv2.waitKey()
    if DISPLAY_HISTS:
        iou_hist(reproj_scallop_ious)

    return reproj_scallop_ious


if __name__ == '__main__':
    with open(ANN_DIRS_FILE, 'r') as anns_file:
        data_dirs = anns_file.readlines()
    reproj_ious = []
    for dir_line in data_dirs:
        if 'STOP' in dir_line:
            break
        if len(dir_line) == 1 or '#' in dir_line:
            continue
        data_dir = dir_line.split(' ')[0][:13] + '/'
        ret_ious = pixel_gps_IoU(data_dir)
        reproj_ious.extend(ret_ious)
    iou_hist(reproj_ious)
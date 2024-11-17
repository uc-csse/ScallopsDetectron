import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pathlib, os
from detectron2.config import get_cfg
from utils import VTKPointCloud as PC, polygon_functions as spf
from utils import geo_utils, file_utils
import vtk
import time
from datetime import datetime
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pickle

NUM_INFERENCE_THREADS = 4

IMG_RS_MOD = 2

MASK_PNTS_SUB = 200

EDGE_LIMIT_PIX = 150 // IMG_RS_MOD
OUTLIER_RADIUS = 0.1
CAM_PROXIMITY_THRESH = 0.3  # m
ELEV_MEAN_PROX_THRESH = 0.05

CAM_SPACING_THRESH = 0.1

IMSHOW = False
VTK = False
WAITKEY = 0
YAPPI_PROFILE = False

OUTPUT_FOV_SHAPES = True

SHAPE_CRS = "EPSG:4326"

if YAPPI_PROFILE:
    import yappi
    yappi.start()


def CamToChunk(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def CamPixToRay(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))

def draw_scaled_axes(img, axis_vecs, axis_scales, origin, cam_mtx):
    points = np.concatenate([np.multiply(axis_vecs, np.repeat(axis_scales[:, None], 3, axis=1)) +
                             np.repeat(origin[None, :], 3, axis=0), origin[None, :]], axis=0)
    axis_points, _ = cv2.projectPoints(points, np.zeros((1, 3)), np.zeros((1, 3)), cam_mtx, None)
    axis_points = axis_points.astype(int)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (255, 0, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (0, 0, 255), 3)


PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'

MODEL_PATH = "/csse/research/CVlab/processed_bluerov_data/training_outputs/fifth/"  #   # "/local/ScallopMaskRCNNOutputs/HR+LR LP AUGS/"

cfg = get_cfg()
cfg.NUM_GPUS = 1
cfg.set_new_allowed(True)
cfg.merge_from_file(MODEL_PATH + 'config.yml')
model_paths = [str(path) for path in pathlib.Path(MODEL_PATH).glob('*.pth')]
model_paths.sort()
print(f"Loading from {model_paths[-1].split('/')[-1]}")
cfg.MODEL.WEIGHTS = os.path.join(model_paths[-1])

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.TEST.DETECTIONS_PER_IMAGE = 100
cfg.TEST.AUG.ENABLED = False
# cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
# cfg.TEST.AUG.MAX_SIZE = 4000
# cfg.TEST.AUG.FLIP = False
# cfg.TEST.PRECISE_BN.ENABLED = False
# cfg.TEST.PRECISE_BN.NUM_ITER = 200
predictor = DefaultPredictor(cfg)


if IMSHOW:
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Labelled sub image", cv2.WINDOW_NORMAL)
if VTK:
    pnt_cld = PC.VtkPointCloud(pnt_size=1)
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(1000, 1000)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    vtk_axes = vtk.vtkAxesActor()
    axes_transform_np = np.eye(4)
    axes_matrix = vtk.vtkMatrix4x4()
    vtk_axes.SetUserMatrix(axes_matrix)
    ren.AddActor(vtk_axes)
    ren.AddActor(pnt_cld.vtkActor)
    iren.Initialize()

def log(str):
    with open(PROCESSED_BASEDIR + 'inference_log.txt', 'a') as lf:
        lf.write(str + '\n')


def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]


def run_inference(base_dir, dirname):
    recon_dir = base_dir + dirname + '/'
    print(f"\n ----- Running CNN on {dirname} -----")

    with open(recon_dir + "chunk_telemetry.pkl", "rb") as pkl_file:
        chunk_telem = pickle.load(pkl_file)
        chunk_scale = chunk_telem['0']['scale']
        chunk_transform = chunk_telem['0']['transform']
        # print(chunk_telem['0']['geoccs'])
        # print(chunk_telem['0']['geogcs'])

    with open(recon_dir + "camera_telemetry.pkl", "rb") as pkl_file:
        camera_telem = pickle.load(pkl_file)

    prediction_geometries = []
    prediction_markers = []
    prediction_labels = []
    gcs2ccs = lambda pnt: geo_utils.geocentric_to_geodetic(pnt[0], pnt[1], pnt[2])
    cnt = 0
    sensor_keys = []
    cam_fov_polys = []
    prev_cam_loc = np.array((3,), dtype=np.float64)
    for cam_label, cam_telem in tqdm(camera_telem.items()):
        cnt += 1
        sensor_key = cam_label.split('-')[0]
        if sensor_key not in sensor_keys:
            sensor_keys.append(sensor_key)
        cam_quart = cam_telem['q44']
        cam_cov = cam_telem['loc_cov33']
        xyz_cov_mean = cam_cov[(0, 1, 2), (0, 1, 2)].mean()
        cam_loc = cam_quart[:3, 3]
        dist_since_last = np.linalg.norm((cam_loc - prev_cam_loc)[:2])
        if dist_since_last < CAM_SPACING_THRESH / chunk_scale:
            continue

        # cam_pos_score = 1.0 - min(0.9, chunk_scale**2 * xyz_cov_mean / CAM_COV_THRESHOLD)
        # print(chunk_scale)
        # print("COV mean: ", chunk_scale**2 * xyz_cov_mean)
        # print(cam_pos_score)
        prev_cam_loc = cam_loc

        img_shape = cam_telem['shape']
        cimg_path = cam_telem['cpath']

        img = cv2.imread(recon_dir + cimg_path)
        rs_size = (img_shape[1] // IMG_RS_MOD, img_shape[0] // IMG_RS_MOD)
        img_rs = cv2.resize(img, rs_size)

        outputs = predictor(img_rs)
        instances = outputs["instances"].to("cpu")
        masks = instances._fields['pred_masks']
        bboxes = instances._fields['pred_boxes']
        scores = instances._fields['scores']

        dimg_path = cam_telem['dpath']
        camMtx = cam_telem['cam_mtx']
        camMtx[:2, :] /= IMG_RS_MOD
        camDist = cam_telem['cam_dist']
        cam_fov = cam_telem['cam_fov']

        depth_img_path = recon_dir + dimg_path
        if '.npy' in dimg_path:
            img_depth_np = np.load(recon_dir + dimg_path)
        else:
            img_depth_u16 = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
            if img_depth_u16 is None:
                continue
            scale = float(depth_img_path.split('/')[-1].split('_')[-1][:-4])
            img_depth_np = scale * img_depth_u16.astype(np.float32)

        img_depth_np = cv2.resize(img_depth_np, rs_size)
        # img_depth_np = cv2.blur(img_depth_np, ksize=(11, 11))

        edge_box = (EDGE_LIMIT_PIX, EDGE_LIMIT_PIX, rs_size[0]-EDGE_LIMIT_PIX, rs_size[1]-EDGE_LIMIT_PIX)

        if OUTPUT_FOV_SHAPES:
            fov_rect = np.array([[0, 0], [rs_size[0], 0], [rs_size[0], rs_size[1]], [0, rs_size[1]]])
            fov_rect_ud = spf.undistort_pixels(fov_rect, camMtx, camDist).astype(np.int32)
            depth_img_sample = img_depth_np[::100, ::100]
            avg_z = depth_img_sample[np.where(depth_img_sample > 0)].mean()
            fov_rect_cam = CamPixToRay(fov_rect_ud.T, camMtx) * avg_z
            fov_rect_chunk = CamToChunk(fov_rect_cam, cam_quart)
            fov_rect_geocentric = TransformPoints(fov_rect_chunk, chunk_transform)
            fov_rect_geodetic = np.apply_along_axis(gcs2ccs, 1, fov_rect_geocentric.T)
            cam_fov_polys.append(Polygon(fov_rect_geodetic[:, :2]))

        if len(masks) == 0:
            continue

        if IMSHOW:
            v = Visualizer(img_rs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            v = v.draw_instance_predictions(instances)
            out_image = v.get_image()[:, :, ::-1].copy()
            depth_display = (255*(img_depth_np - np.min(img_depth_np)) / np.max(img_depth_np)).astype(np.uint8)
            depth_display = np.repeat(depth_display[:, :, None], 3, axis=2)

        for mask, box, score in list(zip(masks, bboxes, scores)):
            mask_pnts = np.array(np.where(mask))[::-1].transpose()
            scallop_centre, radius = cv2.minEnclosingCircle(mask_pnts)
            scallop_centre = np.array(scallop_centre, dtype=int)
            if edge_box[0] <= scallop_centre[0] <= edge_box[2] and edge_box[1] <= scallop_centre[1] <= edge_box[3]:
                mask_np = mask.numpy()[:, :, None].astype(np.uint8)
                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                scallop_polygon = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])][:, 0]
                # Clip number of vertices in polygon to 50->100
                # scallop_polygon = scallop_polygon[::(1 + scallop_polygon.shape[0] // 100)]
                if IMSHOW:
                    cv2.circle(out_image, (scallop_centre[0], scallop_centre[1]), int(radius), color=(0, 255, 0), thickness=2)
                    cv2.drawContours(depth_display, contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

                # TODO: check all undistortion code and images distortion

                # Undistort polygon vertices
                scallop_polygon_ud = spf.undistort_pixels(scallop_polygon, camMtx, camDist).astype(np.int32)
                vert_elevations = img_depth_np[scallop_polygon_ud[:, 1], scallop_polygon_ud[:, 0]]
                # Threshold out points close to camera
                valid_indices = np.where(vert_elevations > (CAM_PROXIMITY_THRESH / chunk_scale))
                if len(valid_indices[0]) < 10:
                    continue
                vert_elevation_mean = np.mean(vert_elevations[valid_indices])
                valid_indices = np.where(np.abs(vert_elevations - vert_elevation_mean) < (ELEV_MEAN_PROX_THRESH / chunk_scale))
                if len(valid_indices[0]) < 10:
                    continue
                vert_elevations = vert_elevations[valid_indices]
                scallop_polygon_ud = scallop_polygon_ud[valid_indices]
                scallop_poly_cam = CamPixToRay(scallop_polygon_ud.T, camMtx) * vert_elevations.T

                num_mask_pixs = len(mask_pnts)
                if num_mask_pixs < MASK_PNTS_SUB:
                    continue
                mask_pnts_mod = max(1, num_mask_pixs // MASK_PNTS_SUB)
                mask_pnts_sub = mask_pnts[::mask_pnts_mod]

                vert_elevations = img_depth_np[mask_pnts_sub[:, 1], mask_pnts_sub[:, 0]]
                mask_pnts_ud = spf.undistort_pixels(mask_pnts_sub, camMtx, camDist)
                scallop_pnts_cam = CamPixToRay(mask_pnts_ud.T, camMtx) * vert_elevations.T
                scallop_pnts_cam = spf.remove_outliers(scallop_pnts_cam, OUTLIER_RADIUS / chunk_scale)
                if scallop_pnts_cam.shape[1] < 10:
                    continue

                scallop_polygon_chunk = CamToChunk(scallop_poly_cam, cam_quart)
                scallop_polygon_geocentric = TransformPoints(scallop_polygon_chunk, chunk_transform)
                scallop_polygon_geodetic = np.apply_along_axis(gcs2ccs, 1, scallop_polygon_geocentric.T)
                scallop_polygon_shapely = Polygon(scallop_polygon_geodetic).simplify(tolerance=0.001 / 111e3, preserve_topology=True)
                prediction_geometries.append(scallop_polygon_shapely)
                prediction_markers.append(Point(np.mean(scallop_polygon_geodetic, axis=0)))
                prediction_labels.append(str(round(score.item(), 2)))

                if IMSHOW and scallop_pnts_cam.shape[1] > 1:
                    pc_vecs, pc_lengths, center_pnt = spf.pca(scallop_pnts_cam.T)
                    MUL = 1.9
                    pc_lengths = np.sqrt(pc_lengths) * MUL
                    scaled_pc_lengths = pc_lengths * chunk_scale * 2
                    width_scallop_circle = 2 * chunk_scale * scallop_pnts_cam[2, :].mean() * radius / camMtx[0, 0]
                    cv2.putText(out_image, str(round(scaled_pc_lengths[0], 3)), tuple(scallop_centre + np.array([20, -10])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(out_image, str(round(width_scallop_circle, 3)), tuple(scallop_centre + np.array([20, 30])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                    # draw_scaled_axes(out_image, pc_vecs, pc_lengths, center_pnt, camMtx)

                if VTK:
                    pnt_cld.setPoints(scallop_pnts_cam.T - center_pnt, np.array([1, 1, 1] * scallop_pnts_cam.shape[1]).T)
                    axes_transform_np[:3, :3] = np.multiply(pc_vecs, np.repeat(pc_lengths[:, None], 3, axis=1)).T
                    axes_matrix.DeepCopy(axes_transform_np.ravel())
                    vtk_axes.Modified()
                    iren.Render()
                    iren.Start()

        if IMSHOW:
            # print("Image inference time: {}s".format(time.time()-start_time))
            cv2.rectangle(out_image, edge_box[:2], edge_box[2:], (0, 0, 255), thickness=1)
            cv2.imshow("Input image", img_rs)
            cv2.imshow("Labelled sub image", out_image)
            cv2.imshow("Depth", depth_display)
            key = cv2.waitKey(WAITKEY)
            if key == ord('q'):
                exit(0)

    if len(sensor_keys) != 2:
        log(f"{recon_dir} has {len(sensor_keys)} sensors in camera_telem.pkl!")

    if YAPPI_PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()
        exit(0)

    file_utils.ensure_dir_exists(recon_dir + 'shapes_pred', clear=True)
    shapes_fn = recon_dir + 'shapes_pred/Pred_' + datetime.now().strftime("%d%m%y_%H%M")
    shapes_fn_3d = shapes_fn + '_3D.gpkg'
    gdf_3D = gpd.GeoDataFrame({'geometry': prediction_geometries, 'NAME': prediction_labels},
                              geometry='geometry', crs=SHAPE_CRS)
    gdf_3D.to_file(shapes_fn_3d)
    # markers_fn_3d = shapes_fn + '_3D_markers.gpkg'
    # gdf_3D = gpd.GeoDataFrame({'geometry': prediction_markers, 'NAME': prediction_labels},
    #                           geometry='geometry', crs=SHAPE_CRS)
    # gdf_3D.to_file(markers_fn_3d)

    # Save shapes in 2D also
    gdf = gpd.read_file(shapes_fn_3d)
    new_geo = []
    for polygon in gdf.geometry:
        if polygon.has_z:
            assert polygon.geom_type == 'Polygon'
            lines = [xy[:2] for xy in list(polygon.exterior.coords)]
            new_geo.append(Polygon(lines))
    gdf.geometry = new_geo
    gdf.to_file(shapes_fn + '_2D.gpkg')

    if OUTPUT_FOV_SHAPES:
        cam_fov_gdf = gpd.GeoDataFrame({'geometry': cam_fov_polys, 'NAME': ''},
                                        geometry='geometry', crs=SHAPE_CRS)
        cam_fov_gdf.to_file(recon_dir + 'shapes_pred/cam_fov_rects_2d.gpkg')


if __name__ == "__main__":
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    data_dirs = data_dirs[20:]
    for dir_line in data_dirs:
        if 'STOP' in dir_line:
            break
        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line:
            continue
        data_dir = dir_line[:13]

        # Process this directory
        run_inference(PROCESSED_BASEDIR, data_dir)

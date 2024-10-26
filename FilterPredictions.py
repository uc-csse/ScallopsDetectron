# import Metashape
# print("Metashape version {}".format(Metashape.version))
from utils import VTKPointCloud as PC, polygon_functions as spf
from utils import geo_utils
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import re
import glob
from shapely.geometry import Polygon
import geopandas as gpd

# TODO: make single filtered polygon per cluster
# TODO: display valid/invalid polygon examples
# TODO: weighted confidence function?
# TODO: outlier removal


PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'


SAVE_PLT = True
DISPLAY = False
OUTPUT_SHAPES = True
SAVE_SHAPES_2D = True

SCALE_MUL = 1.0
CLUSTER_CNT_THRESH = 3

def process_dir(recon_dir):
    shape_file_3D = glob.glob(recon_dir + 'shapes_pred/*3D.gpkg')
    assert len(shape_file_3D) == 1
    scallops_gpd = gpd.read_file(shape_file_3D[0])
    scallop_polys_gps = []
    for detection in scallops_gpd.itertuples():
        poly_lonlatz = np.array(detection.geometry.exterior.coords)
        label = detection.NAME
        conf = float(eval(label)['conf'])
        scallop_polys_gps.append([poly_lonlatz, conf])

    datum = scallop_polys_gps[0][0][0]
    scallop_polys_local = [[geo_utils.convert_gps2local(datum, p), c] for p, c in scallop_polys_gps]

    polygons_d = {"detections": scallop_polys_local}

    for key, polygon_detections in polygons_d.items():
        print("Analysing {}...".format(key))

        print("Filtering {} polygons...".format(len(polygon_detections)))
        scallop_polygons, invalid_polygons = spf.filter_polygon_detections(polygon_detections)
        # scallop_polygons = [p[0] for p in polygon_detections]
        print("Filtered down to {} polygons".format(len(scallop_polygons)))

        print("RNN clustering polygons...")
        polygon_clusters, labels = spf.polygon_rnn_clustering(scallop_polygons, ["labels"]*len(scallop_polygons))
        print("Num clusters: {}".format(len(polygon_clusters)))

        print("Filtering clusters...")
        polygon_clusters, invalid_clusters = spf.filter_clusters(polygon_clusters)
        print("Filtered down to {} clusters".format(len(polygon_clusters)))

        print("Combining cluster masks...")
        filtered_polygons = [clust[0] for clust in polygon_clusters]

        print("Reducing filtered polygon complexity")
        filtered_polygons = [poly[::(1 + len(poly) // 40)] for poly in filtered_polygons]

        filtered_polygons = [geo_utils.convert_local2gps(datum, p) for p in filtered_polygons]

        print("Calculating cluster sizes...")
        scallop_sizes = spf.calc_cluster_widths(polygon_clusters, mode='max')

        if OUTPUT_SHAPES:
            shapes_fn = recon_dir + 'shapes_pred/' + key + '_Filtered_2D.gpkg'
            geometry = [Polygon(poly[:, :2]) for poly in filtered_polygons]
            names = [str(round(i, 4)) for i in scallop_sizes]
            polygons_2d_gpd = gpd.GeoDataFrame({'NAME': names, 'geometry': geometry}, geometry='geometry')
            polygons_2d_gpd.to_file(shapes_fn)


if __name__ == "__main__":
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for dir_line in data_dirs:
        if 'STOP' in dir_line:
            break
        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line:
            continue
        data_dir = dir_line.split(' ')[0][:13] + '/'

        # Process this directory
        process_dir(PROCESSED_BASEDIR + data_dir)

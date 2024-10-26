from utils import vpz_utils, file_utils
import geopandas as gp
from shapely.geometry import *


PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
TO_REPAIR_FP = PROCESSED_BASEDIR + '240713-104835/shapes_ann/JRC_polygon.gpkg'
OUTPUT_FP = PROCESSED_BASEDIR + '240713-104835/shapes_ann/JRC2_polygon.gpkg'

if __name__ == '__main__':
    shape_file = gp.read_file(TO_REPAIR_FP)
    new_geometries = []
    new_labels = []
    for i, row in shape_file.iterrows():
        if isinstance(row.geometry, Polygon):
            new_geometries.append(row.geometry)
            new_labels.append(row.NAME)

    new_write_gdf = gp.GeoDataFrame({'NAME': new_labels, 'geometry': new_geometries}, geometry='geometry')
    new_write_gdf.to_file(OUTPUT_FP)

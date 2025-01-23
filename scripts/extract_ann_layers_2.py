from utils import vpz_utils, file_utils
import glob
import geopandas as gp
from tqdm import tqdm


PROCESSED_BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'
DIRS_FILE = PROCESSED_BASE_DIR + 'dirs_annotation_log.txt'
OUT_DIR = PROCESSED_BASE_DIR + 'annotation_shape_files/'


if __name__ == '__main__':

    file_utils.ensure_dir_exists(OUT_DIR, clear=True)
    with open(DIRS_FILE, 'r') as f:
        dirs_list = f.readlines()

    for dir_entry in tqdm(dirs_list):
        if len(dir_entry) == 1 or '#' in dir_entry or 'STOP' in dir_entry:
            continue
        dir_name = dir_entry[:13]
        dir_full = PROCESSED_BASE_DIR + dir_name + '/'

        shape_layers_gpd = vpz_utils.get_shape_layers_gpd(dir_full, dir_name + '.vpz')
        for label, shape_layer in shape_layers_gpd:
            if '.gpkg' in label:
                label = label[:-5]
            label += '.geojson'
            if 'poly' in label.lower():
                shape_layer.to_file(OUT_DIR + dir_name + '_UC_' + label)
            if 'live' in label.lower():
                shape_layer.to_file(OUT_DIR + dir_name + '_' + label)
        file_utils.SetFolderPermissions(OUT_DIR)

from utils import vpz_utils
import os
import json
from shapely import Point
from tqdm import tqdm
import re
import glob


DATA_BASEDIR = '/csse/research/CVlab/bluerov_data/'
PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_processing_todo.txt'

INCLUDE_TIFFS = False

if __name__ == "__main__":
    with open(DONE_DIRS_FILE, 'r') as f:
        dirs_list = f.readlines()

    geo_tiff_paths = []
    site_tags = {'NAME': [], 'geometry': [], 'Depth': [], 'Altitude': [], 'T.Heading': [], 'ROV LOG': []}
    for dir_entry in tqdm(dirs_list):
        if len(dir_entry) == 1 or '#' in dir_entry or 'STOP' in dir_entry:
            continue
        dir_name = dir_entry[:13]
        dir_full = PROCESSED_BASEDIR + dir_name + '/'

        if INCLUDE_TIFFS:
            lr_tif_fp = dir_full + 'geo_tiffs/chunk0-ortho-lr.tif'
            if os.path.isfile(lr_tif_fp):
                geo_tiff_paths.append(lr_tif_fp)
            else:
                print(f"Site {dir_name} has no LR ortho!")

        if os.path.isfile(dir_full + "scan_metadata.json"):
            with open(dir_full + "scan_metadata.json", 'r') as meta_doc:
                metadata = json.load(meta_doc)
        else:
            raise Exception(f"Site {dir_name} has no JSON metadata!")

        site_tags['geometry'].append(Point(metadata['lonlat']))
        site_tags['NAME'].append(metadata['NAME'] + '|' + dir_name)

        for key in ['Depth', 'Altitude', 'T.Heading']:
            site_tags[key].append(metadata[key])

        site_tags['ROV LOG'].append(';'.join(metadata['ROV LOG']) if 'ROV LOG' in metadata else '')

    print(site_tags['ROV LOG'])

    vpz_utils.write_vpz_file('/csse/research/CVlab/processed_bluerov_data/all_sites_lr.vpz', site_tags, geo_tiff_paths)
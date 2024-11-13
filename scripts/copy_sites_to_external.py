import os
import shutil
import subprocess
from utils import file_utils
from tqdm import tqdm

# Paired sites
# FOLDER_LIST = ['240713-134608', '240620-135134', '240714-113449', '240615-144558', '240617-080551', '240617-132136',
#                '240714-084519', '240713-104835', '240714-140552', '240618-090121', '240616-082046', '240621-082509',
#                '240617-112604', '240616-132704', '240616-105315', '240713-082638', '240618-115508']
# To groom:
FOLDER_LIST = ['240627-163723', '240620-123219', '240629-100813', '240630-072145', '240628-080625', '240629-161409',
               '240605-092733', '240625-110542', '240625-115533', '240629-093514', '240629-092634', '240625-125205',
               '240619-104223', '240710-132415', '240604-133813', '240711-093045', '240711-131748', '240710-142912',
               '240712-111940', '240716-130445', '240603-151729', '240710-110235']
FOLDER_LIST = FOLDER_LIST[:18]

BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'
DST_DIR = '/media/tkr25/ExtremeSSD/data/'
# DST_DIR = '/csse/users/tkr25/Desktop/detection_shape_files/'

if __name__ == '__main__':
    for cp_folder in tqdm(FOLDER_LIST):
        print(f"Copying {cp_folder}")
        src_path = BASE_DIR + cp_folder
        dst_path = DST_DIR + cp_folder
        file_utils.ensure_dir_exists(dst_path, clear=False)
        # subprocess.call(["rsync", '-rtv', src_path + '/geo_tiffs', dst_path])
        subprocess.call(["rsync", '-rtv', src_path + '/shapes_pred', dst_path])
        subprocess.call(["rsync", '-rtv', src_path + '/shapes_ann', dst_path])
        # subprocess.call(["rsync", '-t', src_path + f'/{cp_folder}.vpz', dst_path])

        # subprocess.call(["rsync", '-rtv', src_path + '/shapes_pred', dst_path])

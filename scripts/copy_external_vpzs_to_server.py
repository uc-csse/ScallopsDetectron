import subprocess
from tqdm import tqdm

FOLDER_LIST = ['240627-163723', '240620-123219', '240629-100813', '240630-072145', '240628-080625', '240629-161409',
               '240605-092733', '240625-110542', '240625-115533', '240629-093514', '240629-092634', '240625-125205',
               '240619-104223', '240710-132415', '240604-133813']
FOLDER_LIST = FOLDER_LIST[14:]

DST_DIR = '/csse/research/CVlab/processed_bluerov_data/'
SRC_DIR = '/media/tim/ExtremeSSD/data/'

if __name__ == '__main__':
    for cp_folder in tqdm(FOLDER_LIST):
        print(f"Copying {cp_folder}.vpz to server")
        src_path = SRC_DIR + cp_folder
        dst_path = 'tkr25@linux.cosc.canterbury.ac.nz:' + DST_DIR + cp_folder
        subprocess.call(["scp", src_path + f'/{cp_folder}.vpz', dst_path])

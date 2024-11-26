import os, glob, pickle


PROCESSED_BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASE_DIR + 'dirs_done.txt'


if __name__ == '__main__':
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for dir_line in data_dirs:
        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line or 'STOP' in dir_line:
            continue
        data_dir = dir_line[:13]
        dir_path = PROCESSED_BASE_DIR + data_dir + '/'

        with open(dir_path + "camera_telemetry.pkl", "rb") as pkl_file:
            camera_telem = pickle.load(pkl_file)
        if len(camera_telem) < 100:
            print(f'Problem with {data_dir}: {len(camera_telem)} entries in cam telem!')

        has_preds = os.path.isdir(dir_path + 'shapes_pred') and glob.glob(dir_path + 'shapes_pred/*.gpkg')
        if not has_preds:
            print(f'Problem with {data_dir}: no predictions')
        has_telem = os.path.isfile(dir_path + 'camera_telemetry.pkl') and os.path.isfile(dir_path + 'chunk_telemetry.pkl')
        if not has_telem:
            print(f'Problem with {data_dir}: no telems')
        has_json = os.path.isfile(dir_path + 'scan_metadata.json')

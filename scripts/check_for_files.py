import os, glob


PROCESSED_BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASE_DIR + 'dirs_done.txt'


if __name__ == '__main__':
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for dir_line in data_dirs:
        if 'STOP' in dir_line:
            break
        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line or 'STOP' in dir_line:
            continue
        data_dir = dir_line[:13]
        dir_path = PROCESSED_BASE_DIR + data_dir + '/'

        if os.path.isdir(dir_path + 'shapes_pred') and glob.glob(dir_path + 'shapes_pred/*.gpkg'):
            continue
        else:
            print(f'Problem with {data_dir}!')

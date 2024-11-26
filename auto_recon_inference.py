import ReconstructionInference
import FilterPredictions
import CalculateScallopStatistics

PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"

TODO_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_processing_todo.txt'
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_processing_done.txt'

EDIT_TXTS = True


if __name__ == '__main__':
    idx = 0
    while True:
        # Pop first directory from to-do file
        with open(TODO_DIRS_FILE, 'r') as todo_file:
            data_dirs = todo_file.readlines()
        if idx >= len(data_dirs):
            break
        dir_line = data_dirs.pop(idx)
        if 'STOP' in dir_line:
            break
        if EDIT_TXTS:
            # Append line to done txt file
            with open(DONE_DIRS_FILE, 'a') as check_file:
                check_file.write(dir_line)
            # Erase from to-do list
            with open(TODO_DIRS_FILE, 'w') as todo_file:
                todo_file.writelines(data_dirs)
        else:
            idx += 1

        # Check if this is a valid directory that needs processing
        if len(dir_line) == 1 or '#' in dir_line:
            continue
        data_dir = dir_line[:13]

        # ReconstructionInference.run_inference(PROCESSED_BASEDIR, data_dir)
        try:
            FilterPredictions.process_dir(PROCESSED_BASEDIR, data_dir)
            CalculateScallopStatistics.process_dir(PROCESSED_BASEDIR, data_dir)
        except Exception as e:
            print(e)

        # break
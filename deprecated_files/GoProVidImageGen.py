import cv2
import pathlib as P
import time
import subprocess as sp
import numpy as np
import struct
import matplotlib.pyplot as plt
import pathlib, os
import threading

GOPRO_VIDEO_FOLDER = '/home/cosc/research/CVlab/ROV Gopro Footage/Scallops/121'  #'/home/cosc/research/CVlab/General ROV Footage/Marlborough Sounds - Mussel Farm/GOPRO' # '/local/HarvesterTowGopro' #
START_TIMES = ['00:05:00', '00:00:00', '00:00:00']
END_TIMES = ['00:08:52', '00:08:52', '00:03:52']
START_IDX = 2
EXTRACT_FPS = 5

IMAGE_WRITE_DIR = '/local/ScallopReconstructions/gopro_121/'

SUB_DIR = 'imgs/'
SAVE_IMGS = True
EXTRACT_DATA = False
IMSHOW = False
CENTER_CROP_MUL = 0.75
FRAME_SHAPE = (2160, 3840, 3)

if not pathlib.Path(IMAGE_WRITE_DIR).exists():
    os.mkdir(IMAGE_WRITE_DIR)
if not pathlib.Path(IMAGE_WRITE_DIR+'imgs/').exists():
    os.mkdir(IMAGE_WRITE_DIR+'imgs/')

if EXTRACT_DATA:
    first_file = list(P.Path(GOPRO_VIDEO_FOLDER).iterdir())[0]
    telem_cmnd = [ "ffmpeg",
                   '-i', first_file,
                   '-f', 'image2pipe',
                   '-r', str(EXTRACT_FPS),
                   '-codec', 'copy',
                   '-map', '0:3', '-']
    telem_pipe = sp.Popen(telem_cmnd, stdout=sp.PIPE, bufsize=10**8)
    cnt = 0
    labels = ["ACCL", "DEVC", "DVID", "DVNM", "EMPT", "GPRO", "GPS5", "GPSF", "GPSP", "GPSU", "GYRO", "HD5.", "SCAL", "SIUN",
                "STRM", "TMPC", "TSMP", "UNIT", "TICK", "STNM", "ISOG", "SHUT", "TYPE", "FACE", "FCNM", "ISOE", "WBAL", "WRGB",
                "MAGN", "STMP", "STPS", "SROT", "TIMO", "UNIF", "MTRX", "ORIN", "ALLD", "ORIO"]
    labels_bytes = [label.encode('utf-8') for label in labels]
    telem_buff = b''
    prev_label = None
    prev_label_cnt = 0
    data_buff = []
    while telem_pipe.stdout.readable() and cnt < 100000000:
        telem_buff += telem_pipe.stdout.read(1000)
        cnt += 1000
        print("Byte cnt: {} ".format(cnt), end='\r')
        telem_pipe.stdout.flush()
        #print(telem_buff)
        while len(telem_buff) >= 8:
            lb = telem_buff[:4]
            if lb in labels_bytes:
                desc_0 = int(telem_buff[4])
                if desc_0 == 0 or lb == "EMPT".encode('utf-8'):
                    telem_buff = telem_buff[8:]
                    continue
                val_size = int(telem_buff[5])
                num_values = (int(telem_buff[6]) << 8) | int(telem_buff[7])
                length = val_size * num_values
                if len(telem_buff) < length+8:
                    break
                data_buff.append([lb.decode('utf-8'), num_values, val_size, telem_buff[8:8+length]])
                telem_buff = telem_buff[8+length:]

            else:
                telem_buff = telem_buff[1:]

    gpsf_array = np.array([int.from_bytes(data[3], "big", signed=False) for data in data_buff if data[0] == 'GPSF'])
    #print(gpsf_array)

    # gyro_array = np.array([[int.from_bytes(data[3][:2], "big", signed=False),
    #                         int.from_bytes(data[3][2:4], "big", signed=False),
    #                         int.from_bytes(data[3][4:6], "big", signed=False)] for data in data_buff if data[0] == 'GYRO'])
    #print()
    #print(gyro_array[0])
    #exit(0)

    #print(gpsf_array)
    print("Max GPSF value: {}".format(np.max(gpsf_array)))
    #[print(data) for data in data_buff if data[0] == 'SCAL' and data[2] == 4]
    latlon = [[int.from_bytes(data[3][:4], "big", signed=True), int.from_bytes(data[3][4:8], "big", signed=False)] for data in data_buff if data[0] == 'GPS5']
    latlon = np.array(latlon) / np.array([1e7, 1e7])
    valid_latlon = latlon[gpsf_array > 0]
    latlon_avg = np.mean(latlon, axis=0)
    print(latlon_avg)
    plt.scatter(latlon[:, 0], latlon[:, 1])
    plt.scatter(valid_latlon[:, 0], valid_latlon[:, 1], color='r')
    plt.show()

    exit(0)

def printa():
    print("hello")
    return 0

frame_cnt = 3819
if IMSHOW:
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
vid_paths = list(P.Path(GOPRO_VIDEO_FOLDER).iterdir())
vid_paths.sort()
for vid_path, strt, end in list(zip(vid_paths, START_TIMES, END_TIMES))[START_IDX:]:
    command = ["ffmpeg",
               '-ss', strt,
               '-i', vid_path,
               '-t', end,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-r', str(EXTRACT_FPS),
               '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

    while pipe.stdout.readable():
        # try:
        #     timeout_timer = threading.Timer(5.0, printa)
        #     timeout_timer.start()
        #     raw_image = pipe.stdout.read(FRAME_SHAPE[0] * FRAME_SHAPE[1] * FRAME_SHAPE[2])
        # except KeyboardInterrupt:
        #     break
        raw_image = pipe.stdout.read(FRAME_SHAPE[0] * FRAME_SHAPE[1] * FRAME_SHAPE[2])
        image = np.fromstring(raw_image, dtype='uint8')

        try:
            frame = image.reshape((FRAME_SHAPE[0], FRAME_SHAPE[1], FRAME_SHAPE[2]))[:, :, ::-1]
            frame_cnt += 1
            if IMSHOW:
                cv2.imshow('Frames', frame)

            if SAVE_IMGS:
                center = np.array(frame.shape) / 2
                h, w, _ = (np.array(frame.shape) * CENTER_CROP_MUL).astype(np.int)
                x = int(center[1] - w/2)
                y = int(center[0] - h/2)
                frame_cropped = frame[y:y+h, x:x+w]
                if IMSHOW:
                    cv2.imshow("Cropped", frame_cropped)

                cv2.imwrite(IMAGE_WRITE_DIR + SUB_DIR + str(frame_cnt) + ".png", frame_cropped)

                # blue = cv2.cvtColor(frame_cropped[:, :, 0], cv2.COLOR_GRAY2BGR)
                # green = cv2.cvtColor(frame_cropped[:, :, 1], cv2.COLOR_GRAY2BGR)
                # red = cv2.cvtColor(frame_cropped[:, :, 2], cv2.COLOR_GRAY2BGR)

                # STD_PRES = 10
                #
                # blue_mean = np.mean(blue.astype(np.float32))
                # blue_std = np.std(blue.astype(np.float32))
                # blue_normalised = (255 * np.clip(0.5 * (blue.astype(np.float32) - blue_mean) / (STD_PRES * blue_std) + 0.5, 0.0, 1.0)).astype(np.uint8)
                #
                # green_mean = np.mean(green.astype(np.float32))
                # green_std = np.std(green.astype(np.float32))
                # green_normalised = (255 * np.clip(0.5 * (green.astype(np.float32) - green_mean) / (STD_PRES * green_std) + 0.5, 0.0, 1.0)).astype(np.uint8)
                #
                # red_mean = np.mean(red.astype(np.float32))
                # red_std = np.std(red.astype(np.float32))
                # red_normalised = (255 * np.clip(0.5 * (red.astype(np.float32) - red_mean) / (STD_PRES * red_std) + 0.5, 0.0, 1.0)).astype(np.uint8)
                #
                # cv2.imwrite(IMAGE_WRITE_DIR+'in_imgs_b/'+VID_IDENTIFIER+str(frame_cnt)+".png", blue_normalised)
                # cv2.imwrite(IMAGE_WRITE_DIR+'in_imgs_g/'+VID_IDENTIFIER+str(frame_cnt)+".png", green_normalised)
                # cv2.imwrite(IMAGE_WRITE_DIR+'in_imgs_r/'+VID_IDENTIFIER+str(frame_cnt)+".png", red_normalised)
        except:
            "Frame failed!"

        pipe.stdout.flush()

        if IMSHOW:
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(0)
            elif key == ord(' '):
                break

if IMSHOW:
    cv2.destroyAllWindows()

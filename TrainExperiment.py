import numpy as np
import os
import gc
import pathlib
from detectron2.data import transforms
from detectron2.engine import launch
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
import cv2
from datetime import datetime
from utils import maskrcnn_setup, train_net, file_utils, augmentations as A
from CreateDatasets import CNN_INPUT_SHAPE

WRITE = True
RESUME = True
SHOW_TRAINING_IMGS = False

BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'  # '/local/'#'/scratch/data/tkr25/'  #
NUM_GPUS = 3
BATCH_SIZE = 20  # Images per GPU

augs = [transforms.RandomBrightness(0.8, 1.2),
        transforms.RandomContrast(0.8, 1.2),
        transforms.RandomSaturation(0.8, 1.2),
        transforms.RandomLighting(2),
        # transforms.RandomRotation([-90, 0, 90, 180], sample_style="choice"),
        transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        #transforms.ResizeScale(min_scale=1, max_scale=4.0, target_height=CNN_INPUT_SHAPE[0], target_width=CNN_INPUT_SHAPE[1]),
        # transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
        #A.RandomErasing(),
        #A.RandomColourNoise(),
        #A.GeometricTransform()
        ]
no_augs = []  # transforms.ScaleTransform(h=IMG_SHAPE[0], w=IMG_SHAPE[1],
#                                      new_h=CNN_INPUT_SHAPE[0], new_w=CNN_INPUT_SHAPE[1],
#                                      interp="nearest")]
           #  transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),]

EXP_START_IDX = 0
experiment_titles = ["fifth",]
augmentation_sets = [augs]

# valid_dataset = [BASE_DIR+'ScallopMaskDataset/'+dir for dir in ['lowres_scan_210113_064700_prop', 'gopro_116_0_ortho', 'gopro_116_0_prop']]
# train_dataset_1 = [BASE_DIR+'ScallopReconstructions/'+dir for dir in ['gopro_119/left']]
# train_valid_dataset_sets = [[train_dataset_1, valid_dataset],
#                             ]
# TODO: Dont use validation set in training or testing outputs
valid_dataset_1 = ['240713-134608', '240620-135134', '240714-113449', '240615-144558', '240617-080551', '240617-132136',
                   '240714-084519', '240713-104835', '240621-082509', '240616-082046', '240618-090121', '240714-140552',
                   '240617-112604', '240616-132704', '240616-105315', '240713-082638', '240618-115508',]

valid_dataset_1 = [BASE_DIR + d + '/dataset-' + d + '/' for d in valid_dataset_1]
train_dataset_1 = ['240628-073947', '240630-065651', '240629-152334', '240626-082107', '240628-084724', '240629-134014',
                   '240629-105107', '240627-113019', '240626-154500', '240629-121748', '240626-162116', '240626-165556',
                   '240626-101911', '240607-093240', '240605-092733', '240620-123219', '240629-100813', '240605-092733',
                   '240625-110542', '240625-125205', '240629-093514', '240629-092634', '240619-104223', '240710-132415',
                   '240604-133813', '240712-111940']

train_dataset_1 = [BASE_DIR + d + '/dataset-' + d + '/' for d in train_dataset_1]
datasets = [[train_dataset_1, valid_dataset_1]]

def main(args):
    cfg = maskrcnn_setup.setup(args)

    if SHOW_TRAINING_IMGS:
        mapper = A.CustomMapper(cfg, is_train=True, augmentations=augmentation_sets[args["IDX"]])
        data_loader = build_detection_train_loader(cfg, mapper=mapper, prefetch_factor=2)
        for data in data_loader:
            image = data[0]['image'].to('cpu').numpy().transpose([1, 2, 0])
            v = Visualizer(image[:, :, ::-1])
            if data[0]["instances"].__len__() > 0:
                v = v.overlay_instances(masks=data[0]["instances"].gt_masks, boxes=data[0]["instances"].gt_boxes)
                image_ann = v.get_image()[:, :, ::-1]
                cv2.imshow("Training Image Annotated", image_ann)
            #print(data[0]['image'].shape)
            cv2.imshow("Training Image", image)
            if cv2.waitKey() == ord('q'):
                exit(0)
            #print(data)

    trainer = train_net.Trainer(cfg, args["augmentations"])
    trainer.resume_or_load(resume=RESUME)
    return trainer.train()


if __name__ == '__main__':
    for exp_idx in np.arange(EXP_START_IDX, len(experiment_titles)):
        exp_title = experiment_titles[exp_idx]
        exp_augs = augmentation_sets[exp_idx]
        exp_datasets = datasets[exp_idx]
        output_dir = BASE_DIR + 'training_outputs/' + exp_title
        file_utils.ensure_dir_exists(output_dir, clear=WRITE and not RESUME)

        if WRITE and not RESUME:
            [path.unlink() for path in pathlib.Path(output_dir).iterdir()]
            with open(output_dir + "/Info.txt", 'w') as info_f:
                info_f.write("Experiment Title: {}\n".format(exp_title))
                info_f.write("Experiment Date: {:%B %d, %Y}\n\n".format(datetime.now()))
                info_f.write("Training Directories:\n")
                info_f.writelines(dtst + '\n' for dtst in exp_datasets[0])
                info_f.write("\nAugmentations:\n")
                info_f.writelines([str(aug) + '\n' for aug in exp_augs])

        launch(
            main,
            NUM_GPUS,
            num_machines=1,
            machine_rank=0,
            dist_url='tcp://127.0.0.1:5000'+str(np.random.randint(0, 9)),
            args=({"output_dir": output_dir, "dataset_dirs": exp_datasets, "num_gpus": NUM_GPUS,
                   "gpu_batch_size": BATCH_SIZE, "augmentations": exp_augs, "IDX": exp_idx},),
        )
        gc.collect()

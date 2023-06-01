import os
import numpy as np
import nibabel as nib
import glob
import random
from PIL import Image
import sys

random.seed(42)
NUMBER_OF_PATIENTS = 150

def load_nii_from_dir(source, dest):
    ROOT_PATH = source
    DEST_PATH = dest

    TRAIN_TEST_RATIO = 0.8

    path = os.walk(ROOT_PATH)
    unlabeled_imgs = []
    labeled_imgs = []
    labels = []
    for root, directories, files in path:
        for directory in directories:
            frame_num = []
            for i, filename in enumerate(glob.iglob(f"{root}/{directory}/*[0-9].nii.gz")):
                frame_num.append(int(filename[-9: -7]))
                frame = np.moveaxis(nib.load(filename).get_fdata().astype(np.uint8), -1, 0)
                for slice in frame:
                    slice = np.repeat(np.expand_dims(slice, axis=0), 3, axis=0)
                    labeled_imgs.append(slice)
            for i, filename in enumerate(glob.iglob(f"{root}/{directory}/*gt.nii.gz")):
                frame = np.moveaxis(nib.load(filename).get_fdata().astype(np.uint8), -1, 0)
                for slice in frame:
                    labels.append(slice)

            for i, filename in enumerate(glob.iglob(f"{root}/{directory}/*4d.nii.gz")):
                frames = np.moveaxis(nib.load(filename).get_fdata().astype(np.uint8), [2, 3], [1, 0])
                for i, frame in enumerate(frames):
                    if i+1 not in frame_num:
                        for slice in frame:
                            img = np.repeat(np.expand_dims(slice, axis=0), 3, axis=0)
                            unlabeled_imgs.append(img)

    temp = list(zip(labeled_imgs, labels))
    random.shuffle(temp)
    labeled_imgs, labels = zip(*temp)
    labeled_imgs, labels = list(labeled_imgs), list(labels)

    count = len(labeled_imgs)
    train = 0
    test = 0
    for i, img in enumerate(labeled_imgs):
        img = np.moveaxis(img, 0, -1)
        img = Image.fromarray(img, 'RGB')
        label = Image.fromarray(labels[i], 'L')
        
        if i < count * TRAIN_TEST_RATIO:
            train += 1
            img.save(DEST_PATH + "/labeled-imgs/imgs/train/" + str(i) + ".jpg")
            label.save(DEST_PATH + "/labeled-imgs/labels/train/" + str(i) + ".png")
        else:
            test += 1
            img.save(DEST_PATH + "/labeled-imgs/imgs/val/" + str(i) + ".jpg")
            label.save(DEST_PATH + "/labeled-imgs/labels/val/" + str(i) + ".png")
    print('Labeled:', count, train, test)
    

    random.shuffle(unlabeled_imgs)
    count = len(unlabeled_imgs)
    train = 0
    for i, img in enumerate(unlabeled_imgs):
        img = np.moveaxis(img, 0, -1)
        img = Image.fromarray(img, 'RGB')
        img.save(DEST_PATH + "/all-imgs/imgs/train/" + str(i) + ".jpg")
        train += 1

    print('Unlabeled:', count, train)

if __name__ == "__main__":
    load_nii_from_dir(sys.argv[1], sys.argv[2])
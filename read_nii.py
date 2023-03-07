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
    ONLY_LABELLED = True
    path1 = ROOT_PATH+'/training'
    path2 = ROOT_PATH+'/testing'


    IMG_PATTERN = '*[0-9].nii.gz' if ONLY_LABELLED else '*4d.nii.gz'
    LABEL_PATTERN = '*gt.nii.gz'
    TRAIN_TEST_RATIO = 0.8

    if not ONLY_LABELLED:
        imgs = load_dir_imgs(path1, IMG_PATTERN)
        imgs += load_dir_imgs(path2, IMG_PATTERN)

        random.shuffle(imgs)
        count = len(imgs)
        for i, img in enumerate(imgs):
            img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img, 'RGB')
            if i < count * TRAIN_TEST_RATIO:
                img.save(DEST_PATH + "/imgs/train/" + str(i) + ".jpg")
            else:
                img.save(DEST_PATH + "/imgs/val/" + str(i) + ".jpg")

    else:
        imgs = load_dir_labeled_imgs(path1, IMG_PATTERN, False)
        imgs += load_dir_labeled_imgs(path2, IMG_PATTERN, False)

        labels = load_dir_labeled_imgs(path1, LABEL_PATTERN, True)
        labels += load_dir_labeled_imgs(path2, LABEL_PATTERN, True)

        temp = list(zip(imgs, labels))
        random.shuffle(temp)
        imgs, labels = zip(*temp)
        imgs, labels = list(imgs), list(labels)

        count = len(imgs)

        for i, img in enumerate(imgs):
            img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img, 'RGB')
            label = Image.fromarray(labels[i], 'L')
            
            if i < count * TRAIN_TEST_RATIO:
                img.save(DEST_PATH + "/imgs/train/" + str(i) + ".jpg")
                label.save(DEST_PATH + "/labels/train/" + str(i) + ".png")
            else:
                img.save(DEST_PATH + "/imgs/val/" + str(i) + ".jpg")
                label.save(DEST_PATH + "/labels/val/" + str(i) + ".png")


def load_dir_imgs(root, pattern):
    path = os.walk(root)
    imgs = []
    for root, directories, files in path:
        for directory in directories:
            for i, filename in enumerate(glob.iglob(f"{root}/{directory}/"+pattern)):
                frames = np.moveaxis(nib.load(filename).get_fdata().astype(np.uint8), [2, 3], [1, 0])
                for frame in frames:
                    for slice in frame:
                        img = np.repeat(np.expand_dims(slice, axis=0), 3, axis=0)
                        imgs.append(img)
    return imgs

def load_dir_labeled_imgs(root, pattern, is_label):
    path = os.walk(root)
    imgs = []
    for root, directories, files in path:
        for directory in directories:
            for i, filename in enumerate(glob.iglob(f"{root}/{directory}/"+pattern)):
                frame = np.moveaxis(nib.load(filename).get_fdata().astype(np.uint8), -1, 0)
                for slice in frame:
                    if not is_label:
                        slice = np.repeat(np.expand_dims(slice, axis=0), 3, axis=0)
                    imgs.append(slice)
    return imgs    

if __name__ == "__main__":
    load_nii_from_dir(sys.argv[1], sys.argv[2])
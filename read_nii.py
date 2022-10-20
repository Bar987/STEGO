import os
import numpy as np
import nibabel as nib
import glob
import random
from PIL import Image
import sys

random.seed(42)

def load_nii_from_dir(root_path, dest_path):
    path = os.walk(root_path)

    imgs = []
    for root, directories, files in path:
        for directory in directories:
            for filename in glob.iglob(f"{root_path}/{directory}/*[0-9].nii.gz"):
                img = nib.load(filename).get_fdata().astype(np.uint8)
                img = np.moveaxis(img, -1, 0)

                for slice in img:
                    imgs.append(np.repeat(np.expand_dims(slice, axis=0), 3, axis=0))

    random.shuffle(imgs)

    size = len(imgs)
    for i, img in enumerate(imgs):
        img = np.moveaxis(img, 0, -1)
        img = Image.fromarray(img, 'RGB')
        if i < int(size*0.8):
            img.save(dest_path + "/imgs/train/" + str(i) + ".jpg")
        else:
            img.save(dest_path + "/imgs/val/" + str(i) + ".jpg")

if __name__ == "__main__":
    load_nii_from_dir(sys.argv[1], sys.argv[2])
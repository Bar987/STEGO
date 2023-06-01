import os
import random
from os.path import join

import numpy as np
import torch.multiprocessing
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


class DirectoryDataset(Dataset):
    def __init__(self, root, path, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        self.dir = join(root, path)
        self.img_dir = join(self.dir, "imgs", self.split)
        self.label_dir = join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert len(self.img_files) > 0
        if os.path.exists(join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_fn = self.img_files[index]
        img = Image.open(join(self.img_dir, image_fn))

        if self.label_files is not None:
            label_fn = self.label_files[index]
            label = Image.open(join(self.label_dir, label_fn))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64) - 1

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class Potsdam(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(Potsdam, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdam")
        self.transform = transform
        self.target_transform = target_transform
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }
        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform

        if dataset_name == "potsdam":
            self.n_classes = 3
            dataset_class = Potsdam
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "directory":
            self.n_classes = cfg.dir_dataset_n_classes
            dataset_class = DirectoryDataset
            extra_args = dict(path=cfg.dir_dataset_name)
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
        feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            model_type, nice_dataset_name, image_set, crop_type, cfg.res))
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file) or compute_knns:
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
        }

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        return ret

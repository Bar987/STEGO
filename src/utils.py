import collections
import os
from os.path import join
import io

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import inf
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchmetrics import Metric
from torchvision import models
from torchvision import transforms as T
from torch.utils.tensorboard.summary import hparams


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = [T.ToTensor()(image)]
    writer.log_image(name, image, step)
    plt.clf()
    plt.close()


@torch.jit.script
def shuffle(x):
    return x[torch.randperm(x.shape[0])]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.22294242, 0.22294242, 0.22294242], [0.18982761, 0.18982761, 0.18982761])
unnorm = UnNormalize([0.22294242, 0.22294242, 0.22294242],  [0.18982761, 0.18982761, 0.18982761])


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.RandomRotation(180, Image.BILINEAR),
                          T.RandomHorizontalFlip(),
                          T.RandomResizedCrop(size=res, scale=(0.8, 1.0)),
                          ToTargetTensor()])
    else:
        return T.Compose([
                          T.RandomRotation(180, Image.BILINEAR),
                          T.RandomHorizontalFlip(),
                          T.RandomResizedCrop(size=res, scale=(0.8, 1.0)),
                          T.ToTensor(),
                          normalize
                         ])



def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnsupervisedMetrics(Metric):
    
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        if self.extra_clusters >= 0:
            self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")
        else:
            self.add_state("stats",
                       default=torch.zeros(n_classes, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            if self.extra_clusters >= 0:
                N = (self.n_classes + self.extra_clusters)
                mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
                actual = actual[mask]
                preds = preds[mask]
                temp = torch.bincount(
                    N * actual + preds,
                    minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                    .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)
                self.stats += temp
            else:
                N = self.n_classes
                mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
                actual = actual[mask]
                preds = preds[mask]
                temp = torch.bincount(
                    N * actual + preds,
                    minlength=self.n_classes * (self.n_classes)) \
                    .reshape(self.n_classes, self.n_classes).t().to(self.stats.device)
                self.stats += temp
    
    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]

            # replacing not present labels with background
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, 0)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, 0)
            cluster_to_class = torch.tensor(cluster_to_class)
            temp = cluster_to_class[clusters]
            
            # 0,1,2,3 are the possible labels of ACDC
            temp[temp > 3] = 0
            return temp


    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            else:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)

                #only keeping first 4 rows and columns,
                #  adding the sum of the other ones to the background class
                histogram = self.stats[self.assignments_t[1], :][:4,:4]
                new_row = self.stats[self.assignments_t[1][4:], :4].sum(0, keepdim=True)
                max_sum_row = histogram[histogram.sum(axis=1).argmax()]
                max_sum_row = max_sum_row + new_row
                self.histogram = histogram
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)
        dice = 2 * tp / (2*tp+fp+fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item(),
                       self.prefix + "Dice": dice[~torch.isnan(dice)].mean().item(),
                       self.prefix + "Precision": precision[~torch.isnan(precision)].mean().item(),
                       self.prefix + "Recall": recall[~torch.isnan(recall)].mean().item()}
        return {k: 100 * v for k, v in metric_dict.items()}


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, inf.string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

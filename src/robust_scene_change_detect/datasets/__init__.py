import os

import torch
import torch.utils.data
import torchvision.transforms as tvf

# local dataset
from .pscd import PSCD, CroppedPSCD, DiffViewPSCD
from .vl_cmu_cd import VL_CMU_CD, Diff_VL_CMU_CD
from .custom_dataset import CustomChangeDetectionDataset  # Add this import

# local modules
import robust_scene_change_detect.torch_utils as torch_utils

################
# DATA FACTORY #
################

# global variable for accessing path of datasets
_data_factory = {
    "VL_CMU_CD": VL_CMU_CD,
    "VL_CMU_CD_Diff_View": Diff_VL_CMU_CD,
    "PSCD": CroppedPSCD,
    "PSCD_Diff_View": DiffViewPSCD,
    "PSCD_Full": PSCD,
    "CUSTOM": CustomChangeDetectionDataset,  # Add your custom dataset
}

# dataset name: directory path
_path_factory = {}


def _load_factory():
    """initialize module level variable _factory"""
    global _path_factory

    path, _ = os.path.split(__file__)
    path = os.path.join(path, "..", "..", "..", "data_factory")

    with open(path, "r") as fd:
        datas = fd.read()

    for data in datas.splitlines():
        key, value = data.split()
        _path_factory[key] = value


_load_factory()


def list_path():
    return list(_path_factory.keys())


def list_data():
    return list(_data_factory.keys())


def get_dataset_loader(name):

    if name not in list_data():
        msg = f"{name} do not support. please check other dataset like:\n"
        msg += ", ".join(list_data())
        raise RuntimeError(msg)

    return _data_factory[name]


def get_dataset(name, root=None, **kwargs):

    _remap = {
        "VL_CMU_CD": "VL_CMU_CD",
        "VL_CMU_CD_Diff_View": "VL_CMU_CD",
        "PSCD": "PSCD",
        "PSCD_Diff_View": "PSCD",
        "PSCD_Full": "PSCD",
        "CUSTOM": "CUSTOM",  # Add mapping for your custom dataset
    }

    loader = get_dataset_loader(name)

    if root is None:
        root = _path_factory.get(_remap[name], None)

    if root is None:
        raise ValueError("do not know how to specify correct path")

    return loader(root=root, **kwargs)


#############
# Transform #
#############


def prepare_transform(dataset, mean, std, figsize=None):

    if figsize is None:
        orig_figsize = dataset.figsize
        figsize = orig_figsize // 14 * 14  # be compatible with DINOv2
        figsize = figsize.tolist()

    transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            tvf.Normalize(mean=mean, std=std),
        ]
    )

    target_transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            torch.squeeze,
        ]
    )

    return transform, target_transform


def prepare_transform_wo_normalization(dataset, figsize=None):

    orig_figsize = dataset.figsize
    if figsize is None:
        figsize = orig_figsize // 14 * 14  # be compatible with DINOv2
        figsize = figsize.tolist()

    transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
        ]
    )

    target_transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            torch.squeeze,
        ]
    )

    return transform, target_transform


###########
# Wrapper #
###########


def wrap_eval_dataset(opts, shuffle=True, figsize=None):

    batch_size = opts["batch-size"]
    num_workers = opts["num-workers"]

    def wrapper(dataset, **kwargs):
        transform, target_transform = prepare_transform_wo_normalization(
            dataset, figsize=figsize
        )

        trans_opts = {
            "transform": transform,
            "target_transform": target_transform,
            "hflip_prob": 0.0,
        }

        trans_opts.update(kwargs)

        dataset = torch_utils.CDDataWrapper(dataset, **trans_opts)
        dataset = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return dataset

    return wrapper


# Add function for your custom dataset training
def get_CUSTOM_training_datasets(**opts):
    """
    Create training dataset loader for custom dataset
    """
    batch_size = opts["batch-size"]
    num_workers = opts["num-workers"]
    hflip_prob = opts["hflip-prob"]
    figsize = opts.get("figsize", None)

    wrapper = torch_utils.CDDataWrapper

    # Get your custom training dataset
    trainset_origin = get_dataset("CUSTOM", mode="train")

    transform, target_transform = prepare_transform_wo_normalization(
        trainset_origin, figsize=figsize
    )

    train_opts = {
        "transform": transform,
        "target_transform": target_transform,
        "return_ind": True,
        "hflip_prob": hflip_prob,
    }

    training_sets = wrapper(trainset_origin, **train_opts)

    training_sets = torch.utils.data.DataLoader(
        training_sets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return training_sets


def get_CMU_training_datasets_aug_diff(**opts):

    batch_size = opts["batch-size"]
    num_workers = opts["num-workers"]
    hflip_prob = opts["hflip-prob"]
    figsize = opts.get("figsize", None)

    trainset_origin = get_dataset("VL_CMU_CD", mode="train")
    trainset_origin_diff = get_dataset(
        "VL_CMU_CD_Diff_View",
        mode="train",
        adjacent_distance=1,
    )

    transform, target_transform = prepare_transform_wo_normalization(
        trainset_origin, figsize=figsize
    )

    train_opts = {
        "transform": transform,
        "target_transform": target_transform,
        "return_ind": True,
        "hflip_prob": hflip_prob,
    }

    wrapper = torch_utils.CDDataWrapper
    training_sets = wrapper(trainset_origin, **train_opts)
    training_diff_sets = wrapper(trainset_origin_diff, **train_opts)
    training_aug_diff_sets = wrapper(
        trainset_origin,
        augment_diff_degree=15,
        augment_diff_translate=[-50, 50],
        **train_opts,
    )

    training_sets = torch.utils.data.ConcatDataset(
        [training_sets, training_diff_sets, training_aug_diff_sets]
    )

    training_sets = torch.utils.data.DataLoader(
        training_sets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return training_sets


def get_CMU_training_datasets(**opts):

    if opts.get("diff-augment", False):
        return get_CMU_training_datasets_aug_diff(**opts)

    batch_size = opts["batch-size"]
    num_workers = opts["num-workers"]
    hflip_prob = opts["hflip-prob"]
    figsize = opts.get("figsize", None)

    wrapper = torch_utils.CDDataWrapper

    trainset_origin = get_dataset("VL_CMU_CD", mode="train")

    transform, target_transform = prepare_transform_wo_normalization(
        trainset_origin, figsize=figsize
    )

    train_opts = {
        "transform": transform,
        "target_transform": target_transform,
        "return_ind": True,
        "hflip_prob": hflip_prob,
    }

    training_sets = wrapper(trainset_origin, **train_opts)

    training_sets = torch.utils.data.DataLoader(
        training_sets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return training_sets


def get_PSCD_training_datasets(**opts):

    batch_size = opts["batch-size"]
    num_workers = opts["num-workers"]
    hflip_prob = opts["hflip-prob"]

    wrapper = torch_utils.CDDataWrapper

    trainset_origin = get_dataset(
        "PSCD",
        mode="train",
        use_mask_t0=True,
        use_mask_t1=False,
    )

    transform, target_transform = prepare_transform_wo_normalization(
        trainset_origin
    )

    train_opts = {
        "transform": transform,
        "target_transform": target_transform,
        "return_ind": True,
        "hflip_prob": hflip_prob,
    }

    training_sets = wrapper(trainset_origin, **train_opts)

    training_sets = torch.utils.data.DataLoader(
        training_sets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return training_sets
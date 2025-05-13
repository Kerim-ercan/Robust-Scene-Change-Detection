import torch
import torch.nn as nn

from .backbone_dinov2 import get_dino
from .backbone_resnet import ResNet
from .CD_model import (
    CoAttention,
    CrossAttention,
    SingleCrossAttention1,
    MergeTemporal,
    TemporalAttention,
    ResNet18CrossAttention,
)

import py_utils.utils_torch as utils_torch

_release_root = "https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/"

_checkpoint_factory = {
    # main model (Dinov2)
    "dino_2Cross_CMU": _release_root + "dinov2.2CrossAttn.CMU.pth",
    "dino_2Cross_DiffCMU": _release_root + "dinov2.2CrossAttn.Diff-CMU.pth",
    "dino_2Cross_PSCD": _release_root + "dinov2.2CrossAttn.PSCD.pth",
    # experimental model (ResNet18)
    "resnet_2Cross_CMU": _release_root + "resnet18.2CrossAttn.CMU.pth",
    "resnet_2Cross_PSCD": _release_root + "resnet18.2CrossAttn.PSCD.pth",
    # ablation model
    "dino_1Cross_CMU": _release_root + "dinov2.1CrossAttn.CMU.pth",
    "dino_CoAttn_CMU": _release_root + "dinov2.CoAttn.CMU.pth",
    "dino_TemporalAttn_CMU": _release_root + "dinov2.TemporalAttn.CMU.pth",
    "dino_MTF_CMU": _release_root + "dinov2.MTF.CMU.pth",
}

_model_factory = {
    "dino2 + cross_attention": CrossAttention,
    "dino2 + single_cross_attention1": SingleCrossAttention1,
    "dino2 + merge_temporal": MergeTemporal,
    "dino2 + co_attention": CoAttention,
    "dino2 + temporal_attention": TemporalAttention,
    "resnet18 + cross_attention": ResNet18CrossAttention,
}

_kwargs_map = {
    "dino2": {
        "layer1": "layer1",
        "facet1": "facet1",
        "facet2": "facet2",
        "num-heads": "num_heads",
        "dropout-rate": "dropout_rate",
        "target-shp-row": "target_shp_row",
        "target-shp-col": "target_shp_col",
        "num-blocks": "num_blocks",
    },
    "resnet18": {
        "num-heads": "num_heads",
        "dropout-rate": "dropout_rate",
        "target-shp-row": "target_shp_row",
        "target-shp-col": "target_shp_col",
        "target-feature": "target_feature",
    },
}


def get_kwargs_map(name):
    for key in _kwargs_map.keys():
        if not name.startswith(key):
            continue

        return _kwargs_map[key]
    raise ValueError(f"no kwargs map for {name}")


def get_kwargs(name, **opts):
    kwargs_map = get_kwargs_map(name)

    # check about kwargs map
    msg = ""
    for key in kwargs_map.keys():
        if key in opts.keys():
            continue
        msg += f"'{key}' is missing for {name}\n"

    if len(msg) > 0:
        raise ValueError(msg)

    kwargs = {}
    for key, value in kwargs_map.items():
        kwargs[value] = opts[key]

    return kwargs


def list_models():
    return list(_model_factory.keys())


def get_model_loader(name):
    if name not in list_models():
        msg = "'%s' do not support. please check other models.\n" % name
        msg += ", ".join(list_models())
        raise RuntimeError(msg)

    return _model_factory[name]


def get_dino_backbone(**opts):

    dino_model = opts.get("dino-model", "dinov2_vits14")
    freeze_dino = opts.get("freeze-dino", True)
    unfreeze_dino_last_n_layer = opts.get("unfreeze-dino-last-n-layer", 0)

    backbone = get_dino(
        dino_model,
        freeze=freeze_dino,
        unfreeze_last_n_layers=unfreeze_dino_last_n_layer,
    )

    return backbone


def get_model(**opts):

    if "name" not in opts:
        raise ValueError(f"no model for {name}")
    name = opts.pop("name")

    kwargs = get_kwargs(name, **opts)

    target_shp_row = kwargs.pop("target_shp_row")
    target_shp_col = kwargs.pop("target_shp_col")
    kwargs["target_shp"] = (target_shp_row, target_shp_col)

    loader = get_model_loader(name)

    if "dino" in name:
        backbone = get_dino_backbone(**opts)

    elif "resnet18" in name:
        backbone = ResNet("resnet18")

    else:
        raise ValueError(f"no backbone loader for {name}")

    return loader(backbone, **kwargs)


def get_model_from_pretrained(name):
    """
    get model from pretrained
    """

    if name not in _checkpoint_factory:
        msg = "'%s' do not support. please check other models.\n" % name
        msg += ", ".join(_checkpoint_factory.keys())
        raise RuntimeError(msg)

    # load checkpoint
    checkpoint = torch.hub.load_state_dict_from_url(
        _checkpoint_factory[name],
        map_location="cpu",
        progress=True,
        check_hash=True,
    )

    # get model
    model = get_model(**checkpoint["args"]["model"])
    model = nn.DataParallel(model)
    model = utils_torch.load_grad_required_state(model, checkpoint["model"])
    return model

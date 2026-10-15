"""Segmentation model with an ImageNet-pretrained encoder adapted to N input bands."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_model(mcfg: dict, in_channels: int) -> torch.nn.Module:
    arch = getattr(smp, mcfg["arch"])
    # smp re-initialises the first conv for in_channels != 3 by cycling the pretrained RGB
    # filters across the extra bands, so the rest of the encoder stays pretrained.
    return arch(
        encoder_name=mcfg["encoder"],
        encoder_weights=mcfg.get("encoder_weights"),
        in_channels=in_channels,
        classes=1,
    )


def param_groups(model: torch.nn.Module, lr: float, encoder_lr_mult: float, wd: float):
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    return [
        {"params": enc, "lr": lr * encoder_lr_mult, "weight_decay": wd},
        {"params": rest, "lr": lr, "weight_decay": wd},
    ]


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable

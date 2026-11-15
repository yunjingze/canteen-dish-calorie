"""Model factory and fine-tuning policies."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torchvision import models


def build_model(arch: str = "resnet18", num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return model


def _last_stage_modules(model: nn.Module) -> Iterable[nn.Module]:
    if hasattr(model, "layer4"):
        return [model.layer4]
    if hasattr(model, "features"):
        return [model.features[-1]]
    return []


def apply_finetune_policy(model: nn.Module, mode: str) -> Tuple[int, int]:
    """Freeze everything, then optionally unfreeze the last convolutional stage.

    Returns (trainable_params, total_params).
    """
    if mode not in {"frozen", "last_stage", "full"}:
        raise ValueError(mode)
    for p in model.parameters():
        p.requires_grad = False
    if mode in {"last_stage", "full"}:
        for block in _last_stage_modules(model):
            for p in block.parameters():
                p.requires_grad = True
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    for p in head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep the frozen backbone's BN statistics fixed (no running-stat drift)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def greyscale_params(model: nn.Module):
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return backbone, list(head.parameters())

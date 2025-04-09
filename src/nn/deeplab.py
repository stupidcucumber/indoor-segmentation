import pathlib

import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead,
    DeepLabV3_ResNet101_Weights,
)


def create_deeplabv3_model(
    output_channels: int = 1, weights: pathlib.Path | None = None
) -> torch.nn.Module:
    """Create DeepLabV3 model.

    Parameters
    ----------
    output_channels : int
        Number of output features of the model (classes).
    weights : pathlib.Path | None
        Weights of the model.

    Returns
    -------
    torch.nn.Module
        DeepLabV3 model.
    """
    if not weights:
        model = models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
            progress=True,
            aux_loss=True,
        )
        model.classifier = DeepLabHead(2048, output_channels)
        model.aux_classifier = DeepLabHead(1024, output_channels)
    else:
        model = torch.load(weights, map_location=lambda loc, state: loc)
    return models.segmentation.deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        progress=True,
        aux_loss=True,
    )

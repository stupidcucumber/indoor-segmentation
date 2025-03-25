import argparse
import logging
import sys
from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.nn.unet import Unet
from src.utils.data import SegmentationDataset
from src.utils.train import train

logging.basicConfig(
    level="INFO",
    format="[%(levelname)s] - [%(name)s] - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments.

    Returns
    -------
    argparse.Namespace
        Extracted arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size for training."
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to train."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        help="Backbone for the model. Choose from ['resnet', 'alexnet'].",
    )

    return parser.parse_args()


def main(
    batch: int, device: Literal["cpu", "cuda"], backbone: Literal["resnet", "alexnet"]
) -> None:
    """Start training loop.

    Parameters
    ----------
    batch : int
        Batch size.
    device : Literal["cpu", "cuda"]
        Device on which to inference model.
    backbone : Literal["resnet", "alexnet"]
        Backbone for the segmentation model.
    """
    logger.info("Loading model...")
    model = Unet(in_channels=3, nclasses=150, backbone=backbone)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    IMAGE_SIZE = (576, 576)

    logger.info("Loading datasets...")
    train_dataset = SegmentationDataset(split="train", image_size=IMAGE_SIZE)
    val_dataset = SegmentationDataset(split="validation", image_size=IMAGE_SIZE)
    test_dataset = SegmentationDataset(split="test", image_size=IMAGE_SIZE)

    logger.info(f"Start training on image sizes {IMAGE_SIZE}")
    train(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=20,
        device=device,
        train_dataloader=DataLoader(train_dataset, batch, shuffle=True),
        val_dataloader=DataLoader(val_dataset, batch),
        test_dataloader=DataLoader(test_dataset, batch),
    )


if __name__ == "__main__":
    args = parse_arguments()

    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        logger.info("User interrupted training process.")
    else:
        logger.info("Training successfully finished.")

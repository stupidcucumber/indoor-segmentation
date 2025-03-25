import argparse
import logging
import sys
from datetime import datetime
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
        "--epochs", type=int, default=20, help="Number of epochs to train for."
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

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Size of the input. It must be some degree of 2 to correctly propagate.",
    )

    return parser.parse_args()


def main(
    batch: int,
    epochs: int,
    device: Literal["cpu", "cuda"],
    backbone: Literal["resnet", "alexnet"],
    image_size: list[int],
) -> None:
    """Start training loop.

    Parameters
    ----------
    batch : int
        Batch size.
    epochs : int
        Number of epochs to train for.
    device : Literal["cpu", "cuda"]
        Device on which to inference model.
    backbone : Literal["resnet", "alexnet"]
        Backbone for the segmentation model.
    image_size : list[int]
        Size of the input in format [HEIGHT, WIDTH].
    """
    logger.info("Loading model...")
    model = Unet(in_channels=3, nclasses=150, backbone=backbone)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    logger.info("Loading datasets...")
    train_dataset = SegmentationDataset(split="train", image_size=image_size)
    val_dataset = SegmentationDataset(split="validation", image_size=image_size)

    logger.info(f"Start training on image sizes {image_size}")
    train(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        train_dataloader=DataLoader(train_dataset, batch, shuffle=True),
        val_dataloader=DataLoader(val_dataset, batch),
    )

    torch.save(model.state_dict(), f"{backbone}_last_{datetime.now().isoformat()}.pt")


if __name__ == "__main__":
    args = parse_arguments()

    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        logger.info("User interrupted training process.")
    else:
        logger.info("Training successfully finished.")

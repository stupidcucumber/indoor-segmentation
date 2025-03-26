import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.nn.deeplab import create_deeplabv3_model
from src.nn.unet import Unet
from src.utils.data import VOCSegmentationDataset
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
        "--seg-model",
        type=str,
        default="unet",
        help="Model to train: unet, deeplabv3. By default it is unet.",
    )

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
    seg_model: Literal["unet", "deeplabv3"],
    batch: int,
    epochs: int,
    device: Literal["cpu", "cuda"],
    backbone: Literal["resnet", "alexnet"],
    image_size: list[int],
) -> None:
    """Start training loop.

    Parameters
    ----------
    seg_model : Literal["unet", "deeplabv3"]
        Segmentation model to train.
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

    NCLASSES = 21

    if seg_model == "unet":
        model = Unet(in_channels=3, nclasses=NCLASSES, backbone=backbone)
    elif seg_model == "deeplabv3":
        model = create_deeplabv3_model(output_channels=NCLASSES)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    logger.info("Loading datasets...")
    train_dataset = VOCSegmentationDataset(
        split="train", root=Path("data"), nclasses=NCLASSES, image_size=image_size
    )
    val_dataset = VOCSegmentationDataset(
        split="validation", root=Path("data"), nclasses=NCLASSES, image_size=image_size
    )

    logger.info(f"Start training on image sizes {image_size}")
    train(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        train_dataloader=DataLoader(train_dataset, batch, shuffle=True),
        val_dataloader=DataLoader(val_dataset, batch),
        aux=seg_model == "deeplabv3",
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

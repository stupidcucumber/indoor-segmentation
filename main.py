import logging
import sys

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


if __name__ == "__main__":

    logger.info("Loading model...")
    model = Unet(in_channles=3, nclassses=150)
    optimizer = torch.optim.Adam(model.parameters())
    IMAGE_SIZE = (572, 572)

    logger.info("Loading datasets...")
    train_dataset = SegmentationDataset(split="train", image_size=IMAGE_SIZE)
    val_dataset = SegmentationDataset(split="validation", image_size=IMAGE_SIZE)
    test_dataset = SegmentationDataset(split="test", image_size=IMAGE_SIZE)

    logger.info(f"Start training on image sizes {IMAGE_SIZE}")
    train(
        model=model,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        epochs=20,
        device="cpu",
        train_dataloader=DataLoader(train_dataset, 2, shuffle=True),
        val_dataloader=DataLoader(val_dataset, 8),
        test_dataloader=DataLoader(test_dataset, 8),
    )

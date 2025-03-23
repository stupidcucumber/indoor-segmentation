import logging
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _val_step(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.BCEWithLogitsLoss,
) -> float:
    logits = model(images)
    return loss_fn(logits, labels).item()


def _train_step(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.BCEWithLogitsLoss,
) -> float:
    optimizer.zero_grad()
    logits = model(images)
    loss: torch.Tensor = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Literal["cpu", "cuda"],
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader | None = None,
) -> None:
    """Train model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    loss_fn : torch.nn.BCEWithLogitsLoss
        Loss function for training.
    optimizer : torch.optim.Optimizer
        Optimizer to use in training.
    epochs : int
        Number of epochs to train for.
    device : Literal["cpu", "cuda"]
        Device on which to train.
    train_dataloader : torch.utils.data.DataLoader
        Training examples.
    val_dataloader : torch.utils.data.DataLoader
        Examples for validation.
    test_dataloader : torch.utils.data.DataLoader | None
        Examples for testing. By default is None.
    """
    logger.info(f"Start training for {epochs} epochs.")

    for epoch in range(epochs):

        logger.info(f"Epoch {epoch}:")

        train_tqdm = tqdm(train_dataloader, total=len(train_dataloader))

        for images, labels in train_tqdm:

            images = images.to(device)
            labels = labels.to(device)

            loss = _train_step(model, images, labels, optimizer, loss_fn)

            train_tqdm.set_description(f"Loss - {loss:0.3f}")

        with torch.no_grad():

            val_tqdm = tqdm(val_dataloader, total=len(val_dataloader))

            for images, labels in val_tqdm:

                images = images.to(device)
                labels = labels.to(device)

                loss = _val_step(model, images, labels, loss_fn)

                val_tqdm.set_description(f"Loss - {loss:0.3f}")

    if test_dataloader is not None:

        with torch.no_grad():

            logger.info("Testing trained model...")

            loss_track = []

            for images, labels in tqdm(
                test_dataloader, desc="Testing", total=len(test_dataloader)
            ):

                images = images.to(device)
                labels = labels.to(device)

                loss = _val_step(model, images, labels, loss_fn)

                loss_track.append(loss)

            logger.info(f"Test result is: loss - {np.mean(loss)}")

import logging
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import calculate_accuracy

logger = logging.getLogger(__name__)


def _val_step(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.CrossEntropyLoss,
    aux: bool = False,
) -> tuple[torch.Tensor, float]:
    logits = model(images)

    if aux:

        loss = (loss_fn(logits["out"], labels) + loss_fn(logits["aux"], labels)).item()

    else:

        loss = loss_fn(logits, labels).item()

    return logits, loss


def _train_step(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.CrossEntropyLoss,
    aux: bool = False,
) -> tuple[torch.Tensor, float]:

    optimizer.zero_grad()
    logits = model(images)

    if aux:

        loss = loss_fn(logits["out"], labels) + loss_fn(logits["aux"], labels)

    else:

        loss: torch.Tensor = loss_fn(logits, labels)

    loss.backward()
    optimizer.step()
    return logits, loss.item()


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Literal["cpu", "cuda"],
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader | None = None,
    aux: bool = False,
) -> None:
    """Train model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    loss_fn : torch.nn.CrossEntropyLoss
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
    aux : bool
        Whether to use auxiliraly features.
    """
    logger.info(f"Start training for {epochs} epochs.")

    for epoch in range(epochs):

        logger.info(f"Epoch {epoch}:")

        train_tqdm = tqdm(train_dataloader, total=len(train_dataloader))

        loss_accumulator = []
        accuracy_accumulator = []

        model.train()

        for images, labels in train_tqdm:

            images = images.to(device)
            labels = labels.to(device)

            logits, loss = _train_step(
                model, images, labels, optimizer, loss_fn, aux=aux
            )
            accuracy = calculate_accuracy(
                logits if not aux else logits["out"].float(), labels
            )

            loss_accumulator.append(loss)
            accuracy_accumulator.append(accuracy)

            train_tqdm.set_description(
                f"Loss - {np.mean(loss_accumulator):0.3f}, "
                f"Accuracy - {np.mean(accuracy_accumulator):0.3f}"
            )

        with torch.no_grad():

            model.eval()

            val_tqdm = tqdm(val_dataloader, total=len(val_dataloader))

            loss_accumulator = []
            accuracy_accumulator = []

            for images, labels in val_tqdm:

                images = images.to(device)
                labels = labels.to(device)

                logits, loss = _val_step(model, images, labels, loss_fn, aux=aux)
                accuracy = calculate_accuracy(
                    logits if not aux else logits["out"].float(), labels
                )

                loss_accumulator.append(loss)
                accuracy_accumulator.append(accuracy)

                val_tqdm.set_description(
                    f"Loss - {np.mean(loss_accumulator):0.3f}, "
                    f"Accuracy - {np.mean(accuracy_accumulator):0.3f}"
                )

    if test_dataloader is not None:

        with torch.no_grad():

            model.eval()

            logger.info("Testing trained model...")

            loss_accumulator = []
            accuracy_accumulator = []

            for images, labels in tqdm(
                test_dataloader, desc="Testing", total=len(test_dataloader)
            ):

                images = images.to(device)
                labels = labels.to(device)

                logits, loss = _val_step(model, images, labels, loss_fn, aux)
                accuracy = calculate_accuracy(
                    logits if not aux else logits["out"].float(), labels
                )

                loss_accumulator.append(loss)
                accuracy_accumulator.append(accuracy)

            logger.info(
                f"Test result is: loss - {np.mean(loss):0.3f}, "
                f"accuracy - {np.mean(accuracy_accumulator):0.3f}"
            )

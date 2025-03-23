import torch

from src.utils.decode import decode


def calculate_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate accuracy of the model prediction.

    Parameters
    ----------
    logits : torch.Tensor
        Raw outputs from the model.
    target : torch.Tensor
        Ground truth unraveled tensor.

    Returns
    -------
    float
        Model accuracy.
    """
    predictions = decode(logits)
    ground_truth = decode(target)
    return (
        torch.sum(predictions == ground_truth) / (target.shape[-2] * target.shape[-1])
    ).item()

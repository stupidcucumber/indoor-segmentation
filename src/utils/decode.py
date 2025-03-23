import torch


def decode(logits: torch.Tensor) -> torch.Tensor:
    """Ravel output of the model.

    Parameters
    ----------
    logits : torch.Tensor
        Raw outputs of the model.

    Returns
    -------
    torch.Tensor
        Decoded outputs of the model.
    """
    return logits.argmax(dim=1, keepdim=True)

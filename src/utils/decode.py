import cv2
import numpy as np
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


def generate_color_map(nclasses: int = 151) -> dict[int, np.ndarray]:
    """Generate color map for segmentation.

    Parameters
    ----------
    nclasses : int
        Number of classes to generate colord for.

    Returns
    -------
    dict[int, np.ndarray]
        Generated map of colors for classes.
    """
    result = {}

    for class_id in range(nclasses):

        result[class_id] = np.random.randint(0, 255, 3, dtype=np.uint8)

    return result


def decode_cv2(logits: torch.Tensor, cmap: dict[int, np.ndarray]) -> cv2.Mat:
    """Decode model predictions to the image.

    Parameters
    ----------
    logits : torch.Tensor
        Outputs of the segmentation model.
    cmap : dict[int, np.ndarray]
        Map of colors for classes.

    Returns
    -------
    cv2.Mat
        Image of the segmentation.
    """
    decoded_tensor = decode(logits)[0][0]

    result = np.zeros(
        (decoded_tensor.shape[0], decoded_tensor.shape[1], 3), dtype=np.uint8
    )

    for y_index, y in enumerate(decoded_tensor):

        for x_index, x in enumerate(y):

            result[y_index, x_index] = cmap[x.cpu().item()]

    return result

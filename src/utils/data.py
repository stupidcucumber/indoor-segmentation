from typing import Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    PILToTensor,
    Resize,
    ToTensor,
)


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation.

    Parameters
    ----------
    split : Literal["test", "train", "validation"]
        Which partition of the dataset to use.
    image_size : tuple[int, int]
        Shape of the image in the output.
    """

    def __init__(
        self, split: Literal["test", "train", "validation"], image_size: tuple[int, int]
    ) -> None:
        super(SegmentationDataset, self).__init__()

        self.nclasses = 150
        self.image_size = image_size

        self.data = load_dataset("scene_parse_150", split=split)

        self.image_transform = Compose([ToTensor(), Resize(image_size)])

        self.mask_transform = Compose(
            [PILToTensor(), Resize((388, 388), interpolation=InterpolationMode.NEAREST)]
        )

    def _unravel_mask(self, mask: torch.Tensor) -> torch.Tensor:
        result = torch.zeros([self.nclasses, *mask.shape[1:]], dtype=torch.float32)

        for class_id in mask.unique():

            result[int(class_id), mask[0, ...] == class_id] = 1

        return result

    def __len__(self) -> int:
        """Calculate length of the dataset.

        Returns
        -------
        int
            Number of examples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> list[torch.Tensor, torch.Tensor]:
        """Get access to the example by the index.

        Parameters
        ----------
        index : int
            Index of the example.

        Returns
        -------
        list[torch.Tensor, torch.Tensor]
            Image and its corresponding label.
        """
        _dict = self.data[index]
        return self.image_transform(_dict["image"]), self._unravel_mask(
            self.mask_transform(_dict["annotation"])
        )

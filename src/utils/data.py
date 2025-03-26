from pathlib import Path
from typing import Literal

import cv2
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    PILToTensor,
    Resize,
    ToTensor,
    v2,
)


def _unravel_mask(mask: torch.Tensor, nclasses: int) -> torch.Tensor:

    result = torch.zeros([nclasses, *mask.shape[1:]], dtype=torch.float32)

    for class_id in mask.unique():

        result[int(class_id), mask[0, ...] == class_id] = 1

    return result


def _unravel_mask_voc(mask: torch.Tensor, cmap: dict[int, torch.Tensor]) -> None:

    result = torch.zeros([len(cmap.keys()), *mask.shape[1:]], dtype=torch.float32)

    for class_id, color in cmap.items():

        result[class_id, (mask == color).prod(dim=0).bool()] = 1

    return result


def _color_map(nclasses: int = 256) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}

    def _bitget(byteval: int, idx: int) -> bool:
        return (byteval & (1 << idx)) != 0

    for i in range(nclasses):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (_bitget(c, 0) << 7 - j)
            g = g | (_bitget(c, 1) << 7 - j)
            b = b | (_bitget(c, 2) << 7 - j)
            c = c >> 3

        result[i] = torch.tensor([[[r]], [[g]], [[b]]])

    return result


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation.

    Parameters
    ----------
    split : Literal["test", "train", "validation"]
        Which partition of the dataset to use.
    nclasses : int
        Number of classes in the dataset.
    image_size : tuple[int, int]
        Shape of the image in the output.
    """

    def __init__(
        self,
        split: Literal["test", "train", "validation"],
        nclasses: int,
        image_size: tuple[int, int],
    ) -> None:
        super(SegmentationDataset, self).__init__()

        self.nclasses = nclasses
        self.image_size = image_size

        self.data = load_dataset("scene_parse_150", split=split)

        self.image_transform = Compose([ToTensor(), Resize(image_size)])

        self.mask_transform = Compose(
            [PILToTensor(), Resize(image_size, interpolation=InterpolationMode.NEAREST)]
        )

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

        image = self.image_transform(_dict["image"])
        mask = _unravel_mask(self.mask_transform(_dict["annotation"]), self.nclasses)

        if image.shape[0] == 1:
            image = image.repeat([3, 1, 1])

        return image, mask


class VOCSegmentationDataset(Dataset):
    """Dataset for semantic segmentation.

    Parameters
    ----------
    split : Literal["train", "validation"]
        Which partition of the dataset to use.
    root : Path
        Path to the root of the VOC dataset.
    nclasses : int
        Number of classes, it must be 21 (20 classes + 1 for background).
    image_size : tuple[int, int]
        Shape of the image in the output.
    """

    def __init__(
        self,
        split: Literal["train", "validation"],
        root: Path,
        nclasses: int,
        image_size: tuple[int, int],
    ) -> None:
        super(VOCSegmentationDataset, self).__init__()
        self.nclasses = nclasses
        self.root = root

        self.cmap = _color_map(self.nclasses)
        self.example_stems = self._extract_split_images_stem(split)

        self.image_transform = Compose(
            [
                ToTensor(),
                Resize(image_size),
            ]
        )

        self.mask_transform = Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32),
                Resize(image_size, interpolation=InterpolationMode.NEAREST),
            ]
        )

    def _extract_split_images_stem(
        self, split: Literal["train", "validation"]
    ) -> list[str]:
        split_txt_directory = self.root / "ImageSets" / "Segmentation"
        split_txt = split_txt_directory / (
            "train.txt" if split == "train" else "val.txt"
        )
        return [line.strip() for line in split_txt.read_text().splitlines()]

    def __len__(self) -> int:
        """Calculate length of the dataset.

        Returns
        -------
        int
            Number of examples.
        """
        return len(self.example_stems)

    def _get_example_image(self, stem: str) -> torch.Tensor:
        image_path = self.root / "JPEGImages" / f"{stem}.jpg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.image_transform(image)

    def _get_example_label(self, stem: str) -> torch.Tensor:
        label_path = self.root / "SegmentationClass" / f"{stem}.png"
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        transformed_label = self.mask_transform(label)
        return _unravel_mask_voc(transformed_label, self.cmap)

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
        stem = self.example_stems[index]
        return (self._get_example_image(stem), self._get_example_label(stem))

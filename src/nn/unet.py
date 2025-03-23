import torch
from torch.nn import Conv2d, Module

from src.nn.part.downsampling import Downsampling
from src.nn.part.upsampling import Upsampling


class Unet(Module):
    """Unet model for semantic image segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels, e.g. 1 for monochrome images, 3 for RGB.
    nclasses : int
        Number of classes. If it is binary segmentation
        must be 1 (one for background already allocated).

    Notes
    -----
        For more, please refer to the paper https://arxiv.org/pdf/1505.04597.
    """

    def __init__(self, in_channels: int, nclasses: int) -> None:
        super(Unet, self).__init__()

        self.downsampling_1 = Downsampling(in_channels, 64, max_pool=False)
        self.downsampling_2 = Downsampling(64, 128)
        self.downsampling_3 = Downsampling(128, 256)
        self.downsampling_4 = Downsampling(256, 512)
        self.downsampling_5 = Downsampling(512, 1024)

        self.upsampling_4 = Upsampling(1024, 512)
        self.upsampling_3 = Upsampling(512, 256)
        self.upsampling_2 = Upsampling(256, 128)
        self.upsampling_1 = Upsampling(128, 64)

        self.conv_1x1 = Conv2d(64, 1 + nclasses, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment input image.

        Parameters
        ----------
        x : torch.Tensor
            Normalised image in the form of tensor.

        Returns
        -------
        torch.Tensor
            Semantic segmentation logits.
        """
        downsampled_1 = self.downsampling_1(x)
        downsampled_2 = self.downsampling_2(downsampled_1)
        downsampled_3 = self.downsampling_3(downsampled_2)
        downsampled_4 = self.downsampling_4(downsampled_3)
        downsampled_5 = self.downsampling_5(downsampled_4)

        upsampled_4 = self.upsampling_4(downsampled_5, downsampled_4)
        upsampled_3 = self.upsampling_3(upsampled_4, downsampled_3)
        upsampled_2 = self.upsampling_2(upsampled_3, downsampled_2)
        upsampled = self.upsampling_1(upsampled_2, downsampled_1)

        return self.conv_1x1(upsampled)

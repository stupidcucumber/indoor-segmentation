import torch
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU


class Downsampling(Module):
    """Downsampling part of the Unet architecture.

    Parameters
    ----------
    in_channels : int
        Number of channels expected from the previous layer.
    out_channels : int
        Number of channels expected to output.
    max_pool : bool
        Whether to use max pooling layer for the input. Default is True.
    """

    def __init__(
        self, in_channels: int, out_channels: int, max_pool: bool = True
    ) -> None:
        super(Downsampling, self).__init__()
        self.max_pool = max_pool
        if self.max_pool:
            self.max_pool_2d = MaxPool2d((2, 2), stride=(2, 2))
        self.bn = BatchNorm2d(in_channels)
        self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=(0, 0))
        self.relu_1 = ReLU()
        self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=(0, 0))
        self.relu_2 = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer. Must be with even
            dimentions (e.g. 320x320 or 128x128).

        Returns
        -------
        torch.Tensor
            Downsampled input of double the channels and half the size.
        """
        if self.max_pool:
            x = self.max_pool_2d(x)
        x = self.bn(x)
        x = self.relu_1(self.conv_1(x))
        return self.relu_2(self.conv_2(x))

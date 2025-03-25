import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


class ResidualBlock(Module):
    """Residual block from the ResNet architecture.

    Parameters
    ----------
    in_channels : int
        Number of channels expected from the previous layer.
    out_channels : int
        Number of channels expected to output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualBlock, self).__init__()

        self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding="same")
        self.bn_1 = BatchNorm2d(out_channels)
        self.relu_1 = ReLU()
        self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding="same")
        self.bn_2 = BatchNorm2d(out_channels)
        self.conv_1x1 = Conv2d(in_channels, out_channels, (1, 1))
        self.relu_2 = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer. Must be with even
            dimentions (e.g. 320x320 or 128x128).

        Returns
        -------
        torch.Tensor
            Extracted features.
        """
        residual_x = self.conv_1x1(x)
        x = self.relu_1(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x))
        return self.relu_2(residual_x + x)

import torch
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Module, ReLU

from src.nn.part.resnet import ResidualBlock


class AlexNetUpsampling(Module):
    """Upsampling part of the architecture.

    Parameters
    ----------
    in_channels : int
        Number of channels expected from the previous layer.
    out_channels : int
        Number of channels expected to output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(AlexNetUpsampling, self).__init__()
        self.conv_trans_2d = ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=(2, 2), stride=2
        )
        self.bn = BatchNorm2d(in_channels)
        self.conv_1 = Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding="same"
        )
        self.relu_1 = ReLU()
        self.conv_2 = Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding="same"
        )
        self.relu_2 = ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample input.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer.
        y : torch.Tensor
            Skipped connection from the contracting downsampling layer.

        Returns
        -------
        torch.Tensor
            Upsampled input with half channels and double the size.
        """
        x = self.conv_trans_2d(x)

        x = torch.cat([y, x], dim=1)

        x = self.bn(x)
        x = self.relu_1(self.conv_1(x))
        return self.relu_2(self.conv_2(x))


class ResNetUpsampling(Module):
    """Upsampling part of the architecture based on ResNet.

    Parameters
    ----------
    in_channels : int
        Number of channels expected from the previous layer.
    out_channels : int
        Number of channels expected to output.

    Notes
    -----
        This part is based on ResNet architecture and employs Residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResNetUpsampling, self).__init__()
        self.conv_trans_2d = ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=(2, 2), stride=2
        )
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample input.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer.
        y : torch.Tensor
            Skipped connection from the contracting downsampling layer.

        Returns
        -------
        torch.Tensor
            Upsampled input with half channels and double the size.
        """
        x = self.conv_trans_2d(x)

        x = torch.cat([y, x], dim=1)

        return self.res_block(x)

import torch
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Module, ReLU


class Upsampling(Module):
    """Upsampling part of the architecture.

    Parameters
    ----------
    in_channels : int
        Number of channels expected from the previous layer.
    out_channels : int
        Number of channels expected to output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Upsampling, self).__init__()
        self.conv_trans_2d = ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=(2, 2), stride=2
        )
        self.bn = BatchNorm2d(in_channels)
        self.conv_1 = Conv2d(in_channels, out_channels, kernel_size=(3, 3))
        self.relu_1 = ReLU()
        self.conv_2 = Conv2d(out_channels, out_channels, kernel_size=(3, 3))
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

        diffY = (y.shape[-2] - x.shape[-2]) // 2
        diffX = (y.shape[-1] - x.shape[-1]) // 2

        y = y[..., diffY:-diffY, diffX:-diffX]

        x = torch.cat([y, x], dim=1)

        x = self.bn(x)
        x = self.relu_1(self.conv_1(x))
        return self.relu_2(self.conv_2(x))

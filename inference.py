import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import cv2
import torch
import torchvision

from src.nn.deeplab import create_deeplabv3_model
from src.nn.unet import Unet
from src.utils.decode import decode_cv2, generate_color_map

logging.basicConfig(
    level="INFO",
    format="[%(levelname)s] - [%(name)s] - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


def _parse_video_capture_address(arg: Any) -> int | Path:
    _result = str(arg)
    if _result.isnumeric():
        return int(_result)
    return Path(_result)


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments.

    Returns
    -------
    argparse.Namespace
        Extracted arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seg-model",
        type=str,
        default="unet",
        help="Model to train: unet, deeplabv3. By default it is unet.",
    )

    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the model weights."
    )

    parser.add_argument(
        "--backbone", type=str, required=True, help="Type of the backbone in the model."
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Size of the input. It must be some degree of 2 to correctly propagate.",
    )

    parser.add_argument(
        "--input-stream",
        type=_parse_video_capture_address,
        required=True,
        help="Path to the input stream. 0 for fron camera.",
    )

    return parser.parse_args()


def main(
    seg_model: Literal["unet", "deeplabv3"],
    weights: Path,
    backbone: Literal["alexnet", "resnet"],
    image_size: list[int],
    input_stream: int | Path,
) -> None:
    """Run online inference of the scene.

    Parameters
    ----------
    seg_model : Literal["unet", "deeplabv3"]
        Segmentation model to use.
    weights : Path
        Path to the weights of the model.
    backbone : Literal["alexnet", "resnet"]
        Backbone to use with the model. Only applicable for unet.
    image_size : list[int]
        Image size of the input.
    input_stream : int | Path
        Input address to run inference from.
    """
    NCLASSES = 150

    cmap = generate_color_map(NCLASSES + 1)

    input_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(image_size)]
    )

    if seg_model == "unet":
        model = Unet(in_channels=3, nclasses=NCLASSES, backbone=backbone)
        model.load_state_dict(torch.load(weights, lambda storage, loc: storage))
    else:
        model = create_deeplabv3_model(output_channels=NCLASSES + 1, weights=weights)

    model.eval()

    cap = cv2.VideoCapture(input_stream)

    while cap.isOpened():

        captured, frame = cap.read()

        tensor = torch.unsqueeze(input_transform(frame), 0)

        if captured:

            logits = model(tensor)

            segmentation = decode_cv2(
                logits if seg_model == "unet" else logits["out"], cmap
            )

            result = cv2.vconcat(
                [frame, cv2.resize(segmentation, frame.shape[::-1][1:])]
            )

            cv2.imshow("Segmented image", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


if __name__ == "__main__":

    args = parse_arguments()

    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        logger.info("User interrupted inferencing.")
    else:
        logger.info("Done inferencing.")

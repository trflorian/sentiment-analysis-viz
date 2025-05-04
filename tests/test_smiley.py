from pathlib import Path

import cv2
import numpy as np
import pytest

from sentiment_analysis.utils import create_sentiment_image

IMAGE_SIZE = (512, 512)


@pytest.mark.parametrize(
    "positivity",
    np.linspace(-1, 1, 5),
)
def test_sentiments(visual_output_path: Path, positivity: float) -> None:
    """
    Test the smiley face generation.
    """
    image = create_sentiment_image(positivity, IMAGE_SIZE)

    assert image.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 4)

    # assert center pixel is opaque
    assert image[IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2, 3] == 255

    # save the image for visual inspection
    positivity_num_0_100 = int((positivity + 1) * 50)
    image_fn = f"smiley_{positivity_num_0_100}.png"
    cv2.imwrite(str(visual_output_path / image_fn), image)

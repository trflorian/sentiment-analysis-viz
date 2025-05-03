import cv2
import numpy as np


def color_hsv_to_bgr(
    hue: float, saturation: float, value: float
) -> tuple[int, int, int]:
    """
    Converts HSV color values to BGR format.

    Args:
        hue: Hue value in the range [0, 1].
        saturation: Saturation value in the range [0, 1].
        value: Value (brightness) in the range [0, 1].

    Returns:
        A tuple representing the BGR color.
    """
    px_img_hsv = np.array(
        [[[hue * 180, saturation * 255, value * 255]]], dtype=np.uint8
    )
    px_img_bgr = cv2.cvtColor(px_img_hsv, cv2.COLOR_HSV2BGR)  # Convert from HSV to BGR
    b, g, r = px_img_bgr[0][0]
    return int(b), int(g), int(r)


def create_sentiment_image(
    positivity: float, image_size: tuple[int, int]
) -> np.ndarray:
    """
    Generates a sentiment image based on the positivity score.
    This draws a smiley with its expression based on the positivity score.

    Args:
        positivity: A float representing the positivity score in the range [-1, 1].

    Returns:
        A string representing the path to the generated sentiment image.
    """
    frame = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    color_outline = (80,) * 3 + (255,)  # black
    thickness_outline = 40

    # draw a cricle in the center of the image which will be the head of the smiley
    center = (image_size[0] // 2, image_size[1] // 2)
    radius = min(image_size) // 2 - thickness_outline

    color = color_hsv_to_bgr(
        hue=(positivity + 1) / 6,
        saturation=0.5,
        value=1,
    )

    cv2.circle(frame, center, radius, color + (255,), -1)
    cv2.circle(frame, center, radius, color_outline, thickness_outline)

    # calculate the position of the eyes
    eye_radius = radius // 10
    eye_offset_x = radius // 3
    eye_offset_y = radius // 4
    eye_left = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    eye_right = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    cv2.circle(frame, eye_left, eye_radius, color_outline, -1)
    cv2.circle(frame, eye_right, eye_radius, color_outline, -1)

    # mouth parameters
    mouth_radius = radius // 2
    mouth_offset_y = radius // 4
    mouth_center_y = center[1] + mouth_offset_y + positivity * mouth_radius // 2
    mouth_left = (center[0] - mouth_radius, center[1] + mouth_offset_y)
    mouth_right = (center[0] + mouth_radius, center[1] + mouth_offset_y)

    # calculate points of polynomial for the mouth
    ply_points_t = np.linspace(-1, 1, 100)
    ply_points_y = np.polyval([positivity, 0, 0], ply_points_t)

    ply_points = np.array(
        [
            (
                mouth_left[0] + i * (mouth_right[0] - mouth_left[0]) / 100,
                mouth_center_y - ply_points_y[i] * mouth_radius,
            )
            for i in range(len(ply_points_y))
        ],
        dtype=np.int32,
    )

    # draw the mouth
    cv2.polylines(
        frame,
        [ply_points],
        isClosed=False,
        color=color_outline,
        thickness=int(thickness_outline * 1.5),
    )

    return frame

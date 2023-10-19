# TODO
# 旋转、缩放（用户锚定）
# 透视（脑锚定）
# 多图取平均（多次刺激叠加）

from typing import *

import cv2
import numpy as np


def rotate(img: np.ndarray, angle: float, center: tuple[int, int]) -> np.ndarray:
    """Rotate an image.

    Args:
        img: The image to be rotated.
        angle: The angle of rotation.
        center: The center of rotation.

    Returns:
        The rotated image.
    """
    h, w = img.shape[-2:]
    mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    return cv2.warpAffine(src=img, M=mat, dsize=(w, h), flags=cv2.INTER_LINEAR)


def zoom(
    upper: Union[int, float],
    lower: Union[int, float],
    left: Union[int, float],
    right: Union[int, float],
    state: Literal["in", "out"],
    position: Optional[tuple[int, int]] = None,
) -> tuple[float, float, float, float]:
    """Zoom an image.

    Args:
        upper: The upper bound of the zoomed image.
        lower: The lower bound of the zoomed image.
        left: The left bound of the zoomed image.
        right: The right bound of the zoomed image.
        position: The position of the zoomed image.
        state: The state of the zooming.

    Returns:
        The zoomed upper, lower, left, right bounds.
    """
    FACTOR = 0.1
    if state == "in":
        factor = FACTOR
    elif state == "out":
        factor = -FACTOR
    if position is None:
        x = int((left + right) / 2)
        y = int((upper + lower) / 2)
    else:
        x, y = position

    new_bound = lambda bound, pos: bound - (bound - pos) * factor
    return (
        new_bound(upper, y),
        new_bound(lower, y),
        new_bound(left, x),
        new_bound(right, x),
    )


def perspective(
    img: np.ndarray,
    from_points: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ],
    to_points: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ],
) -> np.ndarray:
    """Perspective an image.

    Args:
        img: The image to be perspective.
        from_points: The original points.
        to_points: The target points.

    Returns:
        The perspective image.
    """
    h, w = img.shape[-2:]
    mat = cv2.getPerspectiveTransform(np.float32(from_points), np.float32(to_points))
    return cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_LINEAR)


class ImagePerspective:
    """Perspective an image.

    Attributes:
        _fig_size: The size of the figure.
        _from_points: The original points.
        _to_points: The target points.
    """

    def __init__(
        self,
        fig_size: tuple[int, int],
        from_points: tuple[
            tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
        ],
        to_points: tuple[
            tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
        ],
    ) -> None:
        """Initialize the perspective.

        Args:
            fig_size: The size of the figure.
            from_points: The original points.
            to_points: The target points.
        """
        self._fig_size = fig_size
        self._from_points = from_points
        self._to_points = to_points
        self._mat = cv2.getPerspectiveTransform(
            np.float32(from_points), np.float32(to_points)
        )
        self._mat_inv = cv2.getPerspectiveTransform(
            np.float32(to_points), np.float32(from_points)
        )

    def __call__(self, img: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Perspective an image.

        Args:
            img: The image to be perspective.
            inverse: Whether to inverse the perspective.

        Returns:
            The perspective image.
        """
        if inverse:
            mat = self._mat_inv
            flags = cv2.INTER_NEAREST
        else:
            mat = self._mat
            flags = cv2.INTER_LINEAR
        return cv2.warpPerspective(img, mat, self._fig_size, flags=flags)


def get_average_image(images: list[np.ndarray]) -> np.ndarray:
    """Get the average image of a list of images.

    Args:
        images: The list of images.

    Returns:
        The average image.
    """
    return np.mean(np.stack(images), axis=0)

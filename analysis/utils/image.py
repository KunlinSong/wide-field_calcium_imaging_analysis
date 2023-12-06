from typing import *

import cv2
import numpy as np

__all__ = [
    "PaddingTransformer",
    "PerspectiveTransformer",
    "RotationTransformer",
    "ZoomBox",
]


class RotationTransformer:
    def __init__(self, angel: float = 0, center: tuple[int, int] = (0, 0)) -> None:
        self.angle = angel
        self.center = center

    @property
    def _rotation_matrix(self) -> np.ndarray:
        return cv2.getRotationMatrix2D(center=self.center, angle=self.angle, scale=1)

    @property
    def _inverse_rotation_matrix(self) -> np.ndarray:
        return cv2.getRotationMatrix2D(center=self.center, angle=-self.angle, scale=1)

    def rotate(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[-2:]
        return cv2.warpAffine(
            src=img, M=self._rotation_matrix, dsize=(w, h), flags=cv2.INTER_LINEAR
        )

    def inverse_rotate(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[-2:]
        return cv2.warpAffine(
            src=img,
            M=self._inverse_rotation_matrix,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
        )

    def get_original_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return tuple(np.dot(self._inverse_rotation_matrix, np.array([*point, 1]))[:2])

    def get_rotated_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return tuple(np.dot(self._rotation_matrix, np.array([*point, 1]))[:2])


class ZoomBox:
    _FACTOR = 0.1

    def __init__(
        self,
        position: tuple[int, int] = (0, 0),
        upper_left: tuple[int, int] = (0, 0),
        lower_right: tuple[int, int] = (0, 0),
    ) -> None:
        self.position = position
        self.upper_left = upper_left
        self.lower_right = lower_right

    @property
    def xlim(self) -> tuple[int, int]:
        return (
            self.position[0] + self.upper_left[0],
            self.position[0] + self.lower_right[0],
        )

    @property
    def ylim(self) -> tuple[int, int]:
        return (
            self.position[1] + self.upper_left[1],
            self.position[1] + self.lower_right[1],
        )

    def zoom(self, state: Literal["in", "out"], position: Optional[tuple[int, int]]):
        if position is None:
            position = self.position
        factor = -self._FACTOR if state == "in" else self._FACTOR
        transform = lambda point: (
            int((point[0] + self.position[0] - position[0]) * (1 + factor)),
            int((point[1] + self.position[1] - position[1]) * (1 + factor)),
        )
        self.upper_left = transform(self.upper_left)
        self.lower_right = transform(self.lower_right)
        self.position = position


class PerspectiveTransformer:
    def __init__(
        self, from_points: tuple[tuple[int, int]], to_points: tuple[tuple[int, int]]
    ) -> None:
        self._from_points = from_points
        self._to_points = to_points
        self._transform_matrix = self._get_transform_matrix(
            self._from_points, self._to_points
        )
        self._inverse_transform_matrix = self._get_transform_matrix(
            self._to_points, self._from_points
        )

    @staticmethod
    def _get_transform_matrix(
        from_points: tuple[tuple[int, int]], to_points: tuple[tuple[int, int]]
    ) -> np.ndarray:
        return cv2.getPerspectiveTransform(
            np.float32(from_points), np.float32(to_points)
        )

    @property
    def from_points(self) -> tuple[tuple[int, int]]:
        return self._from_points

    @from_points.setter
    def from_points(self, from_points: tuple[tuple[int, int]]) -> None:
        self._from_points = from_points
        self._transform_matrix = self._get_transform_matrix(
            self._from_points, self._to_points
        )
        self._inverse_transform_matrix = self._get_transform_matrix(
            self._to_points, self._from_points
        )

    @property
    def to_points(self) -> tuple[tuple[int, int]]:
        return self._to_points

    @to_points.setter
    def to_points(self, to_points: tuple[tuple[int, int]]) -> None:
        self._to_points = to_points
        self._transform_matrix = self._get_transform_matrix(
            self._from_points, self._to_points
        )
        self._inverse_transform_matrix = self._get_transform_matrix(
            self._to_points, self._from_points
        )

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            src=img,
            M=self._transform_matrix,
            dsize=(img.shape[-1], img.shape[-2]),
            flags=cv2.INTER_LINEAR,
        )

    def inverse_perspective_transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            src=img,
            M=self._inverse_transform_matrix,
            dsize=(img.shape[-1], img.shape[-2]),
            flags=cv2.INTER_NEAREST,
        )

    def get_original_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return tuple(np.dot(self._inverse_transform_matrix, np.array([*point, 1]))[:2])

    def get_transformed_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return tuple(np.dot(self._transform_matrix, np.array([*point, 1]))[:2])


class PaddingTransformer:
    def __init__(
        self, upper: int = 0, lower: int = 0, left: int = 0, right: int = 0
    ) -> None:
        self._upper = upper
        self._lower = lower
        self._left = left
        self._right = right

    def transform(self, img: np.ndarray) -> np.ndarray:
        return np.pad(img, ((self._upper, self._lower), (self._left, self._right)))

    def inverse_transform(self, img: np.ndarray) -> np.ndarray:
        return img[self._upper : -self._lower, self._left : -self._right]

    def get_original_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] - self._upper, point[1] - self._left)

    def get_transformed_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] + self._upper, point[1] + self._left)

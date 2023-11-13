# ROI选点
# ROI区域信号平均值的变化曲线

import itertools
import os
from typing import *

import numpy as np
import pandas as pd


class BoundPoints:
    """Bound points.

    Attributes:
        _points: The points.
    """

    def __init__(self) -> None:
        """Initialize the bound points."""
        self._points = []

    def append(self, point: tuple[int, int]) -> None:
        """Append a point.

        Args:
            point: The point to be appended.
        """
        self._points.append(point)

    def pop(self) -> Optional[tuple[int, int]]:
        """Pop a point.

        Returns:
            The popped point.
        """
        if len(self._points) > 0:
            return self._points.pop(-1)

    @property
    def points(self) -> list[tuple[int, int]]:
        """Get the points."""
        return self._points

    @property
    def x(self) -> list[int]:
        """Get the x coordinates."""
        return [x for x, _ in self._points]

    @property
    def y(self) -> list[int]:
        """Get the y coordinates."""
        return [y for _, y in self._points]

    @property
    def bound(self) -> list[list[int], list[int]]:
        """Get the bound coordinates."""
        return [self.bound_x, self.bound_y]

    @property
    def bound_x(self) -> list[int]:
        """Get the bound x coordinates."""
        x = self.x
        return x + [x[0]]

    @property
    def bound_y(self) -> list[int]:
        """Get the bound y coordinates."""
        y = self.y
        return y + [y[0]]

    def clear(self) -> None:
        """Clear the points."""
        self._points.clear()

    def __len__(self) -> int:
        """Get the number of points."""
        return len(self._points)

    def save(self, path: str) -> None:
        """Save the bound points.

        Args:
            path: The path to save the bound points.
        """
        df = pd.DataFrame({"x": self.x, "y": self.y})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


class Point:
    """Point.

    Attributes:
        _x: The x coordinate.
        _y: The y coordinate.
    """

    def __init__(self, x: int, y: int) -> None:
        """Initialize the point.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        self._x = x
        self._y = y

    @property
    def coordinate(self) -> tuple[int, int]:
        """Get the coordinate."""
        return self._x, self._y

    def is_on_segment(self, point1: tuple[int, int], point2: tuple[int, int]) -> bool:
        """Check if the point is on the segment.

        Args:
            point1: The first point of the segment.
            point2: The second point of the segment.

        Returns:
            Whether the point is on the segment.
        """
        x1, y1 = point1
        x2, y2 = point2
        x, y = self._x, self._y
        if (x < min(x1, x2)) or (x > max(x1, x2)):
            return False
        if (y < min(y1, y2)) or (y > max(y1, y2)):
            return False
        if (x, y) in (point1, point2):
            return True
        if x1 == x2:
            return True
        elif (x == x1) or (x == x2):
            return False
        if y1 == y2:
            return True
        elif (y == y1) or (y == y2):
            return False
        return ((y - y1) / (x - x1)) == ((y2 - y1) / (x2 - x1))

    def _right_ray_is_intersecting(
        self, point1: tuple[int, int], point2: tuple[int, int]
    ) -> bool:
        """Check if the right ray of the point is intersecting with the segment.
        If the point is on the segment, it is not intersecting.

        Args:
            point1: The first point of the segment.
            point2: The second point of the segment.

        Returns:
            Whether the right ray of the point is intersecting with the segment.
        """
        x1, y1 = point1
        x2, y2 = point2
        x, y = self._x, self._y
        if self.is_on_segment(point1, point2):
            return False
        if y1 == y2:
            return False
        if (y < min(y1, y2)) or (y > max(y1, y2)):
            return False
        line_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        return x < line_x

    def is_inside(self, bound_points: BoundPoints) -> bool:
        """Check if the point is inside the bound. If the point is on the bound,
        it is inside.

        Args:
            bound_points: The bound points.

        Returns:
            Whether the point is inside the bound.
        """
        if self.coordinate in bound_points.points:
            return True
        points = bound_points.points + [bound_points.points[0]]
        count = 0
        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]
            if self.is_on_segment(point1, point2):
                return True
            if self._right_ray_is_intersecting(point1, point2):
                count += 1
        return count % 2 == 1


def get_inside_points(
    width: int, height: int, bound_points: BoundPoints
) -> list[tuple[int, int]]:
    """Get the inside points of the bound.

    Args:
        width: The width of the image.
        height: The height of the image.
        bound_points: The bound points.

    Returns:
        The inside points of the bound.
    """
    points = set(itertools.product(range(width), range(height)))
    inside_points = []
    for point in points:
        if Point(*point).is_inside(bound_points):
            inside_points.append(point)
    return inside_points


def get_roi_average(frames: np.ndarray, bound_points: BoundPoints) -> np.ndarray:
    """Get the average of the ROI.

    Args:
        frames: The frames.
        bound_points: The bound points.

    Returns:
        The average of the ROI.
    """
    width, height = frames.shape[-2:]
    inside_points = get_inside_points(width, height, bound_points)
    inside_x = [x for x, _ in inside_points]
    inside_y = [y for _, y in inside_points]
    return frames[:, inside_x, inside_y].mean(axis=-1)

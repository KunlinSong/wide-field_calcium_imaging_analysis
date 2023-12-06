import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["BoundPoints", "get_roi_average"]


class BoundPoints:
    X = "x"
    Y = "y"

    def __init__(self) -> None:
        self._points = []

    def append(self, point: tuple[int, int]) -> None:
        self._points.append(point)

    def pop(self) -> Optional[tuple[int, int]]:
        if len(self._points) > 0:
            return self._points.pop(-1)

    def __len__(self) -> int:
        return len(self._points)

    @property
    def points(self) -> list[tuple[int, int]]:
        return self._points

    @property
    def x(self) -> list[int]:
        return [x for x, _ in self._points]

    @property
    def y(self) -> list[int]:
        return [y for _, y in self._points]

    @property
    def bound(self) -> list[tuple[int, int]]:
        return self._points + [self._points[0]]

    @property
    def bound_x(self) -> list[int]:
        return self.x + [self.x[0]]

    @property
    def bound_y(self) -> list[int]:
        return self.y + [self.y[0]]

    def clear(self) -> None:
        self._points.clear()

    def save(self, path: str) -> None:
        df = pd.DataFrame({self.X: self.x, self.Y: self.y})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def load(self, path: str) -> None:
        df = pd.read_csv(path)
        self._points.clear()
        for _, row in df.iterrows():
            self._points.append((row[self.X], row[self.Y]))


class _Point:
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def is_inside(self, bound_points: BoundPoints) -> bool:
        x, y = self._x, self._y

        def _is_on_segment(point1: tuple[int, int], point2: tuple[int, int]) -> bool:
            if x1 == x2:
                judge1 = x == x1
            else:
                judge1 = (x1 < x < x2) or (x2 < x < x1)
            if y1 == y2:
                judge2 = y == y1
            else:
                judge2 = (y1 < y < y2) or (y2 < y < y1)
            judge3 = ((y - y1) * (x2 - x1)) == ((y2 - y1) * (x - x1))
            return judge1 and judge2 and judge3

        def _right_ray_crossing(
            point1: tuple[int, int], point2: tuple[int, int]
        ) -> bool:
            judge1 = min(y1, y2) <= y <= max(y1, y2)
            judge2 = ((x2 - x1) * (y - y1)) >= ((y2 - y1) * (x - x1))
            return judge1 and judge2

        if (n := len(bound_points)) > 0:
            if (x, y) in bound_points.points:
                return True
            points = bound_points.bound
            cross = 0
            for i in range(n - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if _is_on_segment((x1, y1), (x2, y2)):
                    return True
                if _right_ray_crossing((x1, y1), (x2, y2)):
                    cross += 1
            return cross % 2 == 1
        else:
            return False


def _get_inside_points(
    width: int, height: int, bound_points: BoundPoints
) -> list[tuple[int, int]]:
    points = set(itertools.product(range(width), range(height)))
    inside_points = []
    for point in points:
        if _Point(*point).is_inside(bound_points):
            inside_points.append(point)
    return inside_points


def get_roi_average(frames: np.ndarray, bound_points: BoundPoints) -> np.ndarray:
    h, w = frames.shape[-2:]
    inside_points = _get_inside_points(w, h, bound_points)
    inside_x, inside_y = zip(*inside_points)
    return np.mean(frames[:, inside_x, inside_y], axis=-1)

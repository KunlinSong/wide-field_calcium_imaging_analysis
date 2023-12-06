import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class DorsalMap:
    _FILENAME = "DorsalMap.npz"
    _MAP = "Map"
    _EDGE = "Edge"
    _MASK = "Mask"
    _OB_LEFT = "OBLeft"
    _OB_CENTER = "OBCenter"
    _OB_RIGHT = "OBRight"
    _RSP_BASE = "RSPBase"
    _WIDTH = "Width"
    _HEIGHT = "Height"
    _PATH = os.path.join(os.path.dirname(__file__), _FILENAME)
    _MASK_COLOR_MAP = ListedColormap([(0, 0, 0, 0), "white"])

    def __init__(self) -> None:
        _data = np.load(self._PATH)
        self._data = {k: _data[k] for k in _data.files}
        self._resized_data = self._data.copy()

    @property
    def _map(self) -> np.ndarray:
        return self._data[self._MAP]

    @property
    def _edge(self) -> np.ndarray:
        return self._data[self._EDGE]

    @property
    def _mask(self) -> np.ndarray:
        return self._data[self._MASK]

    @property
    def _ob_left(self) -> tuple[int, int]:
        return tuple(self._data[self._OB_LEFT].tolist())

    @property
    def _ob_center(self) -> tuple[int, int]:
        return tuple(self._data[self._OB_CENTER].tolist())

    @property
    def _ob_right(self) -> tuple[int, int]:
        return tuple(self._data[self._OB_RIGHT].tolist())

    @property
    def _rsp_base(self) -> tuple[int, int]:
        return tuple(self._data[self._RSP_BASE].tolist())

    @property
    def _width(self) -> int:
        return self._data[self._WIDTH].item()

    @property
    def _height(self) -> int:
        return self._data[self._HEIGHT].item()

    @property
    def _aspect_ratio(self) -> float:
        return self._width / self._height

    # --
    @property
    def map_(self) -> np.ndarray:
        return self._resized_data[self._MAP]

    @property
    def edge(self) -> np.ndarray:
        return self._resized_data[self._EDGE]

    @property
    def mask(self) -> np.ndarray:
        return self._resized_data[self._MASK]

    @property
    def ob_left(self) -> tuple[int, int]:
        return tuple(self._resized_data[self._OB_LEFT].tolist())

    @property
    def ob_center(self) -> tuple[int, int]:
        return tuple(self._resized_data[self._OB_CENTER].tolist())

    @property
    def ob_right(self) -> tuple[int, int]:
        return tuple(self._resized_data[self._OB_RIGHT].tolist())

    @property
    def rsp_base(self) -> tuple[int, int]:
        return tuple(self._resized_data[self._RSP_BASE].tolist())

    @property
    def width(self) -> int:
        return self._resized_data[self._WIDTH].item()

    @property
    def height(self) -> int:
        return self._resized_data[self._HEIGHT].item()

    @property
    def map_classes(self) -> list[str]:
        return np.unique(self.map_).tolist()

    def resize(self, width: int, height: int) -> None:
        aspect_ratio = self._aspect_ratio
        target_aspect_ratio = width / height

        if target_aspect_ratio > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)

        pad_width_left = (width - new_width) // 2
        pad_width_right = width - new_width - pad_width_left
        pad_height_upper = (height - new_height) // 2
        pad_height_lower = height - new_height - pad_height_upper

        get_resized_img = lambda img: cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        get_padded_img = lambda img: np.pad(
            img,
            ((pad_height_upper, pad_height_lower), (pad_width_left, pad_width_right)),
            mode="constant",
            constant_values=0,
        )

        resized_map = get_padded_img(get_resized_img(self._map))
        resized_edge = get_padded_img(get_resized_img(self._edge))
        resized_mask = get_padded_img(get_resized_img(self._mask))

        get_resized_point = lambda point: [
            point[0] + pad_width_left,
            point[1] + pad_height_upper,
        ]

        resized_ob_left = get_resized_point(self._ob_left)
        resized_ob_center = get_resized_point(self._ob_center)
        resized_ob_right = get_resized_point(self._ob_right)
        resized_rsp_base = get_resized_point(self._rsp_base)

        self._resized_data = {
            self._MAP: resized_map,
            self._EDGE: resized_edge,
            self._MASK: resized_mask,
            self._OB_LEFT: resized_ob_left,
            self._OB_CENTER: resized_ob_center,
            self._OB_RIGHT: resized_ob_right,
            self._RSP_BASE: resized_rsp_base,
            self._WIDTH: width,
            self._HEIGHT: height,
        }

    def mask_axes(self, ax: plt.Axes) -> None:
        ax.imshow((self.mask != 0).astype(int), cmap=self._MASK_COLOR_MAP)

    def map_frame(
        self,
        frame: np.ndarray,
        ob_left: tuple[int, int],
        ob_center: tuple[int, int],
        ob_right: tuple[int, int],
        rsp_base: tuple[int, int],
    ) -> pd.DataFrame:
        """Return a DataFrame with the mean intensity of each class in the
        dorsal map for each frame as the variation of brain structure"""
        perspective_transform = cv2.getPerspectiveTransform(
            np.float32([self.ob_left, self.ob_center, self.ob_right, self.rsp_base]),
            np.float32([ob_left, ob_center, ob_right, rsp_base]),
        )
        map_ = cv2.warpPerspective(
            self.map_,
            perspective_transform,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
        )
        return pd.DataFrame(
            {
                f"{class_}": frame[..., map_ == class_].mean(axis=-1)
                for class_ in self.map_classes
            }
        )

    def map_and_save(
        self,
        frame: np.ndarray,
        ob_left: tuple[int, int],
        ob_center: tuple[int, int],
        ob_right: tuple[int, int],
        rsp_base: tuple[int, int],
        save_path: str,
    ) -> None:
        df = self.map_frame(frame, ob_left, ob_center, ob_right, rsp_base)
        df.to_csv(save_path, index=False)

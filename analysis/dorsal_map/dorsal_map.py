import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class DorsalMap:
    _FILENAME = "DorsalMap.npz"
    _MAP_NAME = "Map"
    _EDGE_NAME = "Edge"
    _MASK_NAME = "Mask"
    _OB_LEFT_NAME = "OBLeft"
    _OB_CENTER_NAME = "OBCenter"
    _OB_RIGHT_NAME = "OBRight"
    _RSP_BASE_NAME = "RSPBase"
    _WIDTH_NAME = "Width"
    _HEIGHT_NAME = "Height"
    _PATH = os.path.join(os.path.dirname(__file__), _FILENAME)
    _MASK_COLOR_MAP = ListedColormap([(0, 0, 0, 0), "white"])

    def __init__(self) -> None:
        _data = np.load(self._PATH)
        self._data = {k: _data[k] for k in _data.files}
        self._resized_data = self._data.copy()

    @property
    def _map_array(self) -> np.ndarray:
        return self._data[self._MAP_NAME]

    @property
    def _edge_array(self) -> np.ndarray:
        return self._data[self._EDGE_NAME]

    @property
    def _mask_array(self) -> np.ndarray:
        return self._data[self._MASK_NAME]

    def _get_raw_pos(self, name: str) -> tuple[int, int]:
        return tuple(self._data[name].tolist())

    @property
    def _ob_left_pos(self) -> tuple[int, int]:
        return self._get_raw_pos(self._OB_LEFT_NAME)

    @property
    def _ob_center_pos(self) -> tuple[int, int]:
        return self._get_raw_pos(self._OB_CENTER_NAME)

    @property
    def _ob_right_pos(self) -> tuple[int, int]:
        return self._get_raw_pos(self._OB_RIGHT_NAME)

    @property
    def _rsp_base_pos(self) -> tuple[int, int]:
        return self._get_raw_pos(self._RSP_BASE_NAME)

    @property
    def _raw_width(self) -> int:
        return self._data[self._WIDTH_NAME].item()

    @property
    def _raw_height(self) -> int:
        return self._data[self._HEIGHT_NAME].item()

    @property
    def _raw_aspect_ratio(self) -> float:
        return self._raw_width / self._raw_height

    # --
    @property
    def map_array(self) -> np.ndarray:
        return self._resized_data[self._MAP_NAME]

    @property
    def edge_array(self) -> np.ndarray:
        return self._resized_data[self._EDGE_NAME]

    @property
    def mask_array(self) -> np.ndarray:
        return self._resized_data[self._MASK_NAME]

    def _get_resized_pos(self, name: str) -> tuple[int, int]:
        return tuple(self._resized_data[name])

    @property
    def ob_left(self) -> tuple[int, int]:
        return self._get_resized_pos(self._OB_LEFT_NAME)

    @property
    def ob_center(self) -> tuple[int, int]:
        return self._get_resized_pos(self._OB_CENTER_NAME)

    @property
    def ob_right(self) -> tuple[int, int]:
        return self._get_resized_pos(self._OB_RIGHT_NAME)

    @property
    def rsp_base(self) -> tuple[int, int]:
        return self._get_resized_pos(self._RSP_BASE_NAME)

    @property
    def width(self) -> int:
        return self._resized_data[self._WIDTH_NAME].item()

    @property
    def height(self) -> int:
        return self._resized_data[self._HEIGHT_NAME].item()

    @property
    def map_classes(self) -> list[str]:
        return np.unique(self.map_array).tolist()

    def resize(self, width: int, height: int) -> None:
        aspect_ratio = self._raw_aspect_ratio
        target_aspect_ratio = width / height

        if target_aspect_ratio > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
        resize_factor = new_width / self._raw_width

        pad_width_left = (width - new_width) // 2
        pad_width_right = width - new_width - pad_width_left
        pad_height_upper = (height - new_height) // 2
        pad_height_lower = height - new_height - pad_height_upper

        get_resized_img = lambda img: np.pad(
            cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST),
            ((pad_height_upper, pad_height_lower), (pad_width_left, pad_width_right)),
            mode="constant",
            constant_values=0,
        )

        resized_map = get_resized_img(self._map_array)
        resized_edge = get_resized_img(self._edge_array)
        resized_mask = get_resized_img(self._mask_array)

        get_resized_point = lambda point: [
            point[0] * resize_factor + pad_width_left,
            point[1] * resize_factor + pad_height_upper,
        ]

        resized_ob_left = get_resized_point(self._ob_left_pos)
        resized_ob_center = get_resized_point(self._ob_center_pos)
        resized_ob_right = get_resized_point(self._ob_right_pos)
        resized_rsp_base = get_resized_point(self._rsp_base_pos)

        self._resized_data = {
            self._MAP_NAME: resized_map,
            self._EDGE_NAME: resized_edge,
            self._MASK_NAME: resized_mask,
            self._OB_LEFT_NAME: resized_ob_left,
            self._OB_CENTER_NAME: resized_ob_center,
            self._OB_RIGHT_NAME: resized_ob_right,
            self._RSP_BASE_NAME: resized_rsp_base,
            self._WIDTH_NAME: width,
            self._HEIGHT_NAME: height,
        }

    def mask_axes(self, ax: plt.Axes) -> None:
        ax.imshow((self.edge_array != 0).astype(int), cmap=self._MASK_COLOR_MAP)

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
            self.map_array,
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

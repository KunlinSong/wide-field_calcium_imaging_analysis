import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import ListedColormap

from . import dorsal_map_src, utils

__all__ = ["DorsalMap"]


class DorsalMap:
    MASK_NAME = "Mask"
    EDGE_MAP_NAME = "EdgeMap"
    DORSAL_MAP_NAME = "DorsalMap"
    DORSAL_ANCHOR_FILENAME = "dorsal_anchor.yaml"
    DORSAL_MAP_FILENAME = "dorsal_map.mat"
    DEFAULT_MASK_CMAP = ListedColormap([(0, 0, 0, 0), "white"])

    def __init__(self) -> None:
        dorsal_map_dirname = os.path.dirname(dorsal_map_src.__file__)
        self.dorsal_anchor_path = os.path.join(
            dorsal_map_dirname, self.DORSAL_ANCHOR_FILENAME
        )
        self.dorsal_map_path = os.path.join(
            dorsal_map_dirname, self.DORSAL_MAP_FILENAME
        )
        self._mat_file = sio.loadmat(self.dorsal_map_path)
        self.dorsal_map_anchor = utils.anchor.Anchor.load(self.dorsal_anchor_path)
        self._dorsal_map_mask = self._mat_file[self.MASK_NAME]
        self._dorsal_map_edge = self._mat_file[self.EDGE_MAP_NAME]
        self._dorsal_map_map = self._mat_file[self.DORSAL_MAP_NAME]
        self.dorsal_map_mask = self._dorsal_map_mask.copy()
        self.dorsal_map_edge = self._dorsal_map_edge.copy()
        self.dorsal_map_map = self._dorsal_map_map.copy()

    def _map_resize(
        self, dorsal_map: np.ndarray, figsize: tuple[int, int]
    ) -> np.ndarray:
        map_h, map_w = dorsal_map.shape
        if figsize == (map_w, map_h):
            return dorsal_map
        map_aspec_ratio = map_w / map_h
        fig_w, fig_h = figsize
        fig_aspec_ratio = fig_w / fig_h

        if fig_aspec_ratio == map_aspec_ratio:
            to_w = fig_w
            to_h = fig_h
            left_pad = 0
            right_pad = 0
            top_pad = 0
            bottom_pad = 0
        elif fig_aspec_ratio > map_aspec_ratio:
            to_w = int(fig_h * map_aspec_ratio)
            to_h = fig_h
            left_pad = (fig_w - to_w) // 2
            right_pad = fig_w - to_w - left_pad
            top_pad = 0
            bottom_pad = 0
        else:
            to_w = fig_w
            to_h = int(fig_w / map_aspec_ratio)
            left_pad = 0
            right_pad = 0
            top_pad = (fig_h - to_h) // 2
            bottom_pad = fig_h - to_h - top_pad

        dorsal_map = cv2.resize(
            dorsal_map, (to_w, to_h), interpolation=cv2.INTER_NEAREST
        )
        dorsal_map = np.pad(
            dorsal_map,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            mode="constant",
            constant_values=0,
        )
        return dorsal_map

    def resize(self, figsize: tuple[int, int]) -> None:
        self.dorsal_map_anchor.resize(figsize)
        self.dorsal_map_mask = self._map_resize(self._dorsal_map_mask, figsize)
        self.dorsal_map_edge = self._map_resize(self._dorsal_map_edge, figsize)
        self.dorsal_map_map = self._map_resize(self._dorsal_map_map, figsize)

    def mask_figure(self, ax: plt.Axes):
        ax.imshow((self.dorsal_map_mask != 0).astype(int), cmap=self.DEFAULT_MASK_CMAP)

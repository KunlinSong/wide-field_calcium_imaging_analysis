import os
from tkinter.filedialog import askopenfilename
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap

from .dorsal_map import DorsalMap
from .utils import *


class UserInterfaceDatabase:
    message: str
    sampling_rate: str
    rotate_degrees: str
    mask_invisible: bool
    raw_canvas: FigureCanvas
    post_perspective_canvas: FigureCanvas
    stimulus_lst: list[str]
    color_range: str
    corrected_canvas: FigureCanvas

    MESSAGE = ""
    FRAMES = None
    FRAMETIMES = None
    SAMPLING_RATE = None
    ROTATE_DEGREES = 0
    ACTIVE_ANCHOR = None
    MASK_INVISIBLE = False
    STIMULUS_LST = []
    FOUR_POINTS = ["ob_left", "ob_center", "ob_right", "rsp_base"]
    RAW_CMAP = "hot"
    CORRECTED_CMAP = ListedColormap(["dodgerblue", "navy", "black", "darkred", "gold"])
    RAW_POINT_COLOR = "blue"
    CORRECTED_POINT_COLOR = "lime"

    def __init__(self) -> None:
        self.message = self.MESSAGE
        self._sampling_rate = self.SAMPLING_RATE
        self._rotate_degrees = self.ROTATE_DEGREES
        self.mask_invisible = self.MASK_INVISIBLE
        self._active_anchor = self.ACTIVE_ANCHOR
        self._frames = self.FRAMES
        self._frametimes = self.FRAMETIMES
        self._dorsal_map = DorsalMap()
        self._anchor = Anchor()
        self._baseline_info = BaselineInfo()
        self._bound_points = BoundPoints()
        self._corrected_frames = CorrectedFrames()
        self._perspective_transformer = PerspectiveTransformer(
            from_points=[getattr(self._anchor, point) for point in self.FOUR_POINTS],
            to_points=[getattr(self._dorsal_map, point) for point in self.FOUR_POINTS],
        )
        self._padding_transformer = PaddingTransformer()
        self._rotation_transformer = RotationTransformer(angel=self._rotate_degrees)
        self._stimulus_info = StimulusInfo()
        self._zoom_box = ZoomBox()
        self._raw_fig, self._raw_ax = plt.subplots()
        self._raw_fig.canvas.mpl_connect("button_press_event", self._click_on_raw_fig)
        self._raw_fig.canvas.mpl_connect("scroll_event", self._scroll_on_raw_fig)
        self._raw_fig.canvas.mpl_connect("resize_event", self._resize_on_raw_fig)
        self.raw_canvas = FigureCanvas(self._raw_fig)
        self._post_perspective_fig, self._post_perspective_ax = plt.subplots()
        self.post_perspective_canvas = FigureCanvas(self._post_perspective_fig)
        self._corrected_fig, self._corrected_ax = plt.subplots()
        self._corrected_fig.canvas.mpl_connect(
            "button_press_event", self._click_on_corrected_fig
        )  # left and right click
        self._corrected_fig.canvas.mpl_connect(
            "scroll_event", self._scroll_on_corrected_fig
        )
        self._corrected_fig.canvas.mpl_connect(
            "resize_event", self._resize_on_corrected_fig
        )
        self.corrected_canvas = FigureCanvas(self._corrected_fig)

    @property
    def sampling_rate(self) -> str:
        return str(self._sampling_rate) if self._sampling_rate else ""

    @property
    def rotate_degrees(self) -> str:
        return str(self._rotate_degrees)

    @property
    def stimulus_lst(self) -> list[str]:
        return self._stimulus_info.content

    def load_frames(self) -> None:
        self._path = Path(
            askopenfilename(
                filetypes=[("data", "*.dat"), ("MJ2", "*.mj2"), ("TIFF", "*.tif")]
            )
        )
        self._frames = Frames(self._path.data_path)
        self._frametimes = FrameTimes(self._path.frametimes_path)
        self._dorsal_map.resize(self._frames.width, self._frames.height)
        corrected_frames_path = (
            self._path.generate_dir.preprocessing_dir.corrected_frames_path
        )
        if os.path.exists(corrected_frames_path):
            self._corrected_frames.load_from_path(corrected_frames_path)
            self.message = "Have been preprocessed."
        else:
            self.message = "Please preprocess first."
        anchor_path = self._path.generate_dir.log_dir.anchor_path
        if os.path.exists(anchor_path):
            self._anchor.load(anchor_path)
        sampling_rate_path = self._path.generate_dir.log_dir.sampling_rate_path
        if os.path.exists(sampling_rate_path):
            with open(sampling_rate_path, "r") as f:
                self._sampling_rate = float(f.read())
        center = (self._frames.width // 2, self._frames.height // 2)
        diagonal = int(np.ceil(np.sqrt((center[0] + 1) ** 2 + (center[1] + 1) ** 2)))
        left = right = diagonal - center[0]
        upper = lower = diagonal - center[1]
        center = (center[0] + left, center[1] + upper)
        self._rotation_transformer.center = center
        self._padding_transformer = PaddingTransformer(
            left=left, right=right, upper=upper, lower=lower
        )
        self._zoom_box = ZoomBox(
            position=center,
            upper_left=(left, upper),
            lower_right=(left + self._frames.width, upper + self._frames.height),
        )
        self._perspective_transformer.from_points = [
            getattr(self._anchor, point) for point in self.FOUR_POINTS
        ]
        self._perspective_transformer.to_points = [
            getattr(self._dorsal_map, point) for point in self.FOUR_POINTS
        ]
        if self._frames.n_channels == 1:
            _data = self._frames.data
        else:
            _data = self._frames.data[:, 0]
        _avg = np.mean(_data, axis=0)
        self._raw_img = (_avg - np.min(_avg)) / (np.max(_avg) - np.min(_avg))
        self._refresh_raw_fig()
        self._refresh_post_perspective_fig()

    def reset_rotate_degrees(self) -> None:
        self.assign_rotate_degrees(self.ROTATE_DEGREES)

    def assign_rotate_degrees(self, degrees: str) -> None:
        self._rotate_degrees = float(degrees)
        self._rotation_transformer.angle = self._rotate_degrees
        self._refresh_raw_fig()

    def assign_sampling_rate(self, sampling_rate: str) -> None:
        if sampling_rate:
            self._sampling_rate = float(sampling_rate)
        else:
            self._sampling_rate = None

    def _assign_anchor(self, point: tuple[int, int]) -> None:
        if self._active_anchor is None:
            return
        setattr(self._anchor, self._active_anchor, point)
        self._perspective_transformer.from_points = [
            getattr(self._anchor, point) for point in self.FOUR_POINTS
        ]

    def active_ob_left(self) -> None:
        self._active_anchor = "ob_left"

    def active_ob_center(self) -> None:
        self._active_anchor = "ob_center"

    def active_ob_right(self) -> None:
        self._active_anchor = "ob_right"

    def active_rsp_base(self) -> None:
        self._active_anchor = "rsp_base"

    def assign_mask_invisible(self, mask_invisible: bool) -> None:
        self.mask_invisible = mask_invisible
        self._refresh_post_perspective_fig()

    @property
    def _padded_img(self) -> np.ndarray:
        return self._padding_transformer.transform(self._raw_img)

    @property
    def _rotated_img(self) -> np.ndarray:
        return self._rotation_transformer.rotate(self._padded_img)

    @property
    def _post_perspective_img(self) -> np.ndarray:
        return self._perspective_transformer.perspective_transform(self._raw_img)

    def _get_orignal_point_on_raw_fig(self, point: tuple[int, int]) -> tuple[int, int]:
        return self._padding_transformer.get_original_point(
            self._rotation_transformer.get_original_point(point)
        )

    def _get_transformed_point_on_raw_fig(
        self, point: tuple[int, int]
    ) -> tuple[int, int]:
        return self._rotation_transformer.get_rotated_point(
            self._padding_transformer.get_transformed_point(point)
        )

    def _click_on_raw_fig(self, event) -> None:
        if event.button == 1:
            if (event.xdata is not None) and (event.ydata is not None):
                point = self._get_orignal_point_on_raw_fig((event.xdata, event.ydata))
                self._assign_anchor(point)
                self._refresh_raw_fig()
                self._refresh_post_perspective_fig()

    def _scroll_on_raw_fig(self, event) -> None:
        if event.button == "up":
            state = "in"
        elif event.button == "down":
            state = "out"
        if (event.xdata is None) or (event.ydata is None):
            position = None
        else:
            position = (event.xdata, event.ydata)
        self._zoom_box.zoom(state, position)
        self._refresh_raw_fig()

    def _resize_on_raw_fig(self, event) -> None:
        self._refresh_raw_fig()

    def _refresh_raw_fig(self) -> None:
        self._raw_ax: plt.Axes

        self._raw_ax.cla()
        img = self._raw_ax.imshow(self._rotated_img, cmap=self.RAW_CMAP)
        points = [getattr(self._anchor, point) for point in self.FOUR_POINTS]
        points = [self._get_transformed_point_on_raw_fig(point) for point in points]
        xs, ys = zip(*points)
        _point_size = min(*self._raw_fig.get_size_inches()) * self._raw_fig.dpi / 50
        self._raw_ax.plot(
            xs, ys, "o", color=self.RAW_POINT_COLOR, markersize=_point_size
        )
        self._raw_ax.set_xlim(self._zoom_box.xlim)
        self._raw_ax.set_ylim(self._zoom_box.ylim)
        self._raw_ax.axis("on")
        self._raw_fig.colorbar(img, ax=self._raw_ax)
        self._raw_fig.canvas.draw()

    def _refresh_post_perspective_fig(self) -> None:
        self._post_perspective_ax: plt.Axes

        self._post_perspective_ax.cla()
        img = self._post_perspective_ax.imshow(
            self._post_perspective_img, cmap=self.RAW_CMAP
        )
        if not self.mask_invisible:
            self._dorsal_map.mask_axes(self._post_perspective_ax)
        self._post_perspective_ax.axis("off")
        self._post_perspective_fig.colorbar(img, ax=self._post_perspective_ax)
        self._post_perspective_fig.canvas.draw()

    # TODO
    def preprocess(self, process_func: Optional[Callable] = None) -> None:
        filter = LowPassFilter()
        K = 200
        corrected_frames = np.zeros_like(self.frames.data)
        total = np.prod(self.frames.frame_shape_with_channel)

        for c in range(self.frames.n_channels):
            c_num = c * self.frames.height * self.frames.width
            for h in range(self.frames.height):
                h_num = h * self.frames.width
                for w in range(self.frames.width):
                    filtered_frames = filter(self.frames.data[:, c, h, w])

                    # svd
                    shape = filtered_frames.shape
                    frames_reshaped = filtered_frames.reshape(shape[0], -1)
                    u, s, vh = np.linalg.svd(frames_reshaped, full_matrices=False)
                    u = u[:, :K]
                    s = s[:K]
                    vh = vh[:K, :]
                    frames_reshaped = u @ np.diag(s) @ vh
                    svd_frames = frames_reshaped.reshape(shape)

                    t = np.arange(shape[0])
                    t_0 = t[0::2]
                    t_1 = t[1::2]
                    if c == 0:
                        corrected_frames[:, c, h, w][0::2] = svd_frames
                        corrected_frames[:, c, h, w][1::2] = interp1d(
                            t_0, svd_frames, axis=0, fill_value="extrapolate"
                        )(t_1)
                    elif c == 1:
                        corrected_frames[:, c, h, w][1::2] = svd_frames
                        corrected_frames[:, c, h, w][0::2] = interp1d(
                            t_1, svd_frames, axis=0, fill_value="extrapolate"
                        )(t_0)

                    if process_func is not None:
                        process_func(100 * (c_num + h_num + w + 1) / total)
        self.corrected_frames = utils.data.CorrectedFrames.from_array(corrected_frames)
        return self.corrected_frames

import os
from typing import Callable, Literal, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.interpolate import interp1d

from . import utils
from .dorsal_map import DorsalMap


# ------------------------ Zoom Tools --------------------------------
class ZoomDict:
    def __init__(
        self, position: tuple[int, int], upper: int, lower: int, left: int, right: int
    ) -> None:
        self.position = position
        self.upper = upper
        self.lower = lower
        self.left = left
        self.right = right

    @property
    def xlim(self) -> tuple[int, int]:
        return (self.left + self.position[0], self.right + self.position[0])

    @property
    def ylim(self) -> tuple[int, int]:
        return (self.upper + self.position[1], self.lower + self.position[1])


# ---------------------------- Mixin ----------------------------
class FramesMixin:
    path: utils.basic.Path
    frames: utils.data.Frames
    frame_times: utils.data.FrameTimes
    corrected_frames = utils.data.CorrectedFrames
    _raw_img: np.ndarray
    _raw_center: tuple[int, int]
    _raw_diagonal: int
    _padded_center: tuple[int, int]
    _zoom_dict: ZoomDict
    _padded_img: np.ndarray

    def _load_frames(self) -> utils.data.Frames:
        self.frames = utils.data.Frames(self.path.raw_path)
        self.frame_times = utils.data.FrameTimes(self.path.frametimes_path)
        return self.frames

    def _generate_corrected_frames(
        self, process_func: Optional[Callable] = None
    ) -> utils.data.CorrectedFrames:
        filter = utils.filter.LowPassFilter()
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

    @property
    def _raw_img(self) -> np.ndarray:
        if self.frames.n_channels == 1:
            return np.mean(self.frames.data, axis=0)
        else:
            return np.mean(self.frames.data[:, 0], axis=0)

    @property
    def _raw_center(self) -> tuple[int, int]:
        return (int(self.frames.width / 2), int(self.frames.height / 2))

    @property
    def _raw_diagonal(self) -> int:
        return int(np.ceil(np.sqrt(self.frames.height**2 + self.frames.width**2)))

    def _pad_transform(self, point: tuple[int, int]) -> tuple[int, int]:
        return (coordinate + self._raw_diagonal for coordinate in point)

    def _invert_pad_transofrm(self, point: tuple[int, int]) -> tuple[int, int]:
        return (coordinate - self._raw_diagonal for coordinate in point)

    @property
    def _padded_center(self) -> tuple[int, int]:
        return self._pad_transform(self._raw_center)

    def _get_init_zoom_dict(self, center: tuple[int, int]) -> ZoomDict:
        self._zoom_dict = ZoomDict(
            position=center,
            upper=-self._raw_center[1],
            lower=self.frames.height - self._raw_center[1],
            left=-self._raw_center[0],
            right=self.frames.width - self._raw_center[0],
        )
        return self._zoom_dict

    @property
    def _padded_img(self) -> np.ndarray:
        return np.pad(
            self._raw_img, self._raw_diagonal, "constant", constant_values=np.nan
        )


class RotateDegreeMixin(FramesMixin):
    _rotate_degree: float
    ROTATE_DEGREE: float
    _rotated_img: np.ndarray

    def assign_rotate_degree(self, value: str) -> float:
        self._rotate_degree = float(value.strip())
        self._rotate_img()
        return self._rotate_degree

    def reset_rotate_degree(self) -> float:
        self._rotate_degree = self.ROTATE_DEGREE
        self._rotate_img()
        return self._rotate_degree

    @property
    def _rotation_mat(self) -> np.ndarray:
        return cv2.getRotationMatrix2D(
            center=self._padded_center, angle=self._rotate_degree, scale=1
        )

    @property
    def _invert_rotation_mat(self) -> np.ndarray:
        return cv2.invertAffineTransform(self._rotation_mat)

    def _rotate_img(self) -> np.ndarray:
        h, w = self._padded_img.shape[-2:]
        self._rotated_img = cv2.warpAffine(
            src=self._padded_img,
            M=self._rotation_mat,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
        )
        return self._rotated_img

    def _invert_rotate_transform(self, point: tuple[int, int]) -> tuple[int, int]:
        return np.dot(self._invert_rotation_mat, np.array([*point, 1]))[:2]

    def _rotate_transform(self, point: tuple[int, int]) -> tuple[int, int]:
        return np.dot(self._rotation_mat, np.array([*point, 1]))[:2]


class AnchorMixin(RotateDegreeMixin):
    anchor: utils.anchor.Anchor
    active_point: Literal["ob_left", "ob_center", "ob_right", "rsp_base", None]

    def _load_anchor(self) -> utils.anchor.Anchor:
        if os.path.exists(self.path.anchor_path):
            self.anchor = utils.anchor.Anchor.load(self.path.anchor_path)
            return self.anchor
        else:
            self.anchor = utils.anchor.Anchor(
                self.path.anchor_path, figsize=self.frames.figsize
            )
            return self.anchor

    def _save_anchor(self) -> None:
        self.anchor.save(self.path.anchor_path)

    def activate_ob_left(self) -> None:
        self.active_point = "ob_left"

    def activate_ob_center(self) -> None:
        self.active_point = "ob_center"

    def activate_ob_right(self) -> None:
        self.active_point = "ob_right"

    def activate_rsp_base(self) -> None:
        self.active_point = "rsp_base"

    def _to_origin_transform(self, point: tuple[int, int]) -> tuple[int, int]:
        return self._invert_pad_transofrm(self._invert_rotate_transform(point))

    def _to_img_transform(self, point: tuple[int, int]) -> tuple[int, int]:
        return self._rotate_transform(self._pad_transform(point))

    def assign_position(self, position: tuple[int, int]) -> tuple[int, int]:
        original_position = self._to_origin_transform(position)
        match self.active_point:
            case "ob_left":
                self.anchor.ob_left = original_position
            case "ob_center":
                self.anchor.ob_center = original_position
            case "ob_right":
                self.anchor.ob_right = original_position
            case "rsp_base":
                self.anchor.rsp_base = original_position
            case _:
                return
        return original_position


class BasicFigureMixin(AnchorMixin):
    dorsal_map: DorsalMap
    mask_invisible: bool
    _raw_fig: plt.Figure
    _raw_ax: plt.Axes
    _img_perspective_transform: utils.image.ImagePerspectiveTransform
    _perspective_adjusted_img: np.ndarray
    _perspective_adjusted_fig: plt.Figure
    _perspective_adjusted_ax: plt.Axes

    def _get_img_perspective_transform(self) -> utils.image.ImagePerspectiveTransform:
        from4points = (
            self.anchor.ob_left,
            self.anchor.ob_center,
            self.anchor.ob_right,
            self.anchor.rsp_base,
        )
        to4points = (
            self.dorsal_map.dorsal_map_anchor.ob_left,
            self.dorsal_map.dorsal_map_anchor.ob_center,
            self.dorsal_map.dorsal_map_anchor.ob_right,
            self.dorsal_map.dorsal_map_anchor.rsp_base,
        )
        self._img_perspective_transform = utils.image.ImagePerspectiveTransform(
            self.frames.figsize, from4points, to4points
        )
        return self._img_perspective_transform

    def _init_raw_fig(self) -> plt.Figure:
        self._raw_fig, self._raw_ax = plt.subplots()
        self._zoom_dict = self._get_init_zoom_dict(self._padded_center)
        self.refresh_raw_fig()

    def _init_perpectived_fig(self) -> plt.Figure:
        self._perspective_adjusted_fig, self._perspective_adjusted_ax = plt.subplots()
        self.refresh_perspective_adjusted_fig()

    def _get_point_size(self, fig: plt.Figure) -> float:
        return min(*fig.get_size_inches()) * fig.dpi / 50

    @property
    def _display_ob_left(self) -> tuple[int, int]:
        return self._to_img_transform(self.anchor.ob_left)

    @property
    def _display_ob_center(self) -> tuple[int, int]:
        return self._to_img_transform(self.anchor.ob_center)

    @property
    def _display_ob_right(self) -> tuple[int, int]:
        return self._to_img_transform(self.anchor.ob_right)

    @property
    def _display_rsp_base(self) -> tuple[int, int]:
        return self._to_img_transform(self.anchor.rsp_base)

    @property
    def _display_points(self) -> tuple[tuple[int, int], ...]:
        return (
            self._display_ob_left,
            self._display_ob_center,
            self._display_ob_right,
            self._display_rsp_base,
        )

    def refresh_raw_fig(self) -> None:
        self._raw_ax.cla()
        img = self._raw_ax.imshow(self._rotated_img, cmap="hot")
        display_points = self._display_points
        x = [point[0] for point in display_points]
        y = [point[1] for point in display_points]
        self._raw_ax.plot(
            x,
            y,
            "o",
            color="blue",
            markersize=self._point_size(self._raw_fig),
        )
        self._raw_fig.colorbar(img, ax=self._raw_ax)
        self._raw_ax.set_xlim(*self._zoom_dict.xlim)
        self._raw_ax.set_ylim(*self._zoom_dict.ylim)
        self._raw_ax.axis("on")
        self._raw_fig.canvas.draw()

    def assign_position(self, position: tuple[int, int]) -> tuple[int, int]:
        position = super().assign_position(position)
        self._get_img_perspective_transform()
        return position

    @property
    def _perspective_adjusted_img(self) -> np.ndarray:
        return self._img_perspective_transform(self._raw_img)

    def refresh_perspective_adjusted_fig(self) -> None:
        self._perspective_adjusted_ax.cla()
        img = self._perspective_adjusted_ax.imshow(
            self._perspective_adjusted_img, cmap="hot"
        )
        if not self.mask_invisible:
            self.dorsal_map.mask_figure(self._perspective_adjusted_ax)
        self._perspective_adjusted_fig.colorbar(img, ax=self._perspective_adjusted_ax)
        self._perspective_adjusted_ax.axis("on")
        self._perspective_adjusted_fig.canvas.draw()

    def click_on_raw_fig(self, event) -> None:
        if event.button == 1:
            position = self.assign_position((event.xdata, event.ydata))
            if position is not None:
                self.refresh_raw_fig()
                self.refresh_perspective_adjusted_fig()
        else:
            return

    def scroll_on_raw_fig(self, event) -> None:
        if event.button == "up":
            state = "in"
        elif event.button == "down":
            state = "out"
        else:
            return

        if (event.xdata is None) or (event.ydata is None):
            position = self._zoom_dict.position
        else:
            position = self._invert_rotate_transform((event.xdata, event.ydata))

        upper, lower, left, right = utils.image.zoom(
            self._zoom_dict.upper + self._zoom_dict.position[1] - position[1],
            self._zoom_dict.lower + self._zoom_dict.position[1] - position[1],
            self._zoom_dict.left + self._zoom_dict.position[0] - position[0],
            self._zoom_dict.right + self._zoom_dict.position[0] - position[0],
            state,
        )
        self._zoom_dict = ZoomDict(position, upper, lower, left, right)
        self.refresh_raw_fig()

    def assign_rotate_degree(self, value: str) -> ...:
        degree = super().assign_rotate_degree(value)
        self.refresh_raw_fig()
        return degree

    def reset_rotate_degree(self) -> ...:
        degree = super().reset_rotate_degree()
        self.refresh_raw_fig()
        return degree

    def resize_raw_fig(self, event) -> None:
        self.refresh_raw_fig()


class CorrectedFramesMixin(FramesMixin):
    corrected_frames: Union[utils.data.CorrectedFrames, None]

    def _load_corrected_frames(self) -> utils.data.CorrectedFrames:
        if os.path.exists(self.path.corrected_frames_path):
            self.corrected_frames = utils.data.CorrectedFrames(
                self.path.corrected_frames_path
            )
            return 'No need to "Preprocess".'
        else:
            self.corrected_frames = None
            return 'Please "Preprocess" before "Analyze".'

    def preprocess(self) -> utils.data.CorrectedFrames:
        self.corrected_frames = self._generate_corrected_frames()
        self.corrected_frames.save(self.path.corrected_frames_path)
        return self.corrected_frames


class StimulusMixin(CorrectedFramesMixin):
    stimulus: utils.data.StimLog

    def _get_init_stimulus(self) -> utils.data.StimLog:
        self.stimulus = utils.data.StimLog()
        return self.stimulus

    def load_stimulus(self, path: str) -> utils.data.StimLog:
        if os.path.exists(path):
            self.stimulus.load(path, frame_times=self.frame_times)
            return self.stimulus
        else:
            return self.stimulus

    def _save_stimulus(self) -> None:
        self.stimulus.save(self.path.stimulus_path)

    def clear_stimulus(self) -> utils.data.StimLog:
        self.stimulus = utils.data.StimLog()
        return self.stimulus

    def add_stimulus(self, start: int, end: int) -> utils.data.StimLog:
        self.stimulus.append(start=start, end=end)
        return self.stimulus

    def remove_stimulus(self, idx: int) -> utils.data.StimLog:
        self.stimulus.remove(idx)
        return self.stimulus

    @property
    def _stimulus_frames_dict(self) -> dict[int, np.ndarray]:
        stimulus_frames = {}
        for start, end in self.stimulus:
            stimulus_frames[start] = self.corrected_frames.data[start:end]
        return stimulus_frames


class BaselineMixin(StimulusMixin):
    sampling_rate: float
    _baseline_info: utils.data.BaselineInfo

    def assign_baseline_info(
        self,
        mode: Literal["Frame Mode", "PreStim Mode"],
        start: int,
        end: int,
        pre_sec: float,
    ) -> utils.data.BaselineInfo:
        self._baseline_info = utils.data.BaselineInfo(
            mode=mode, start=start, end=end, pre_sec=pre_sec
        )
        return self._baseline_info

    @property
    def widefield_change(self):
        match self._baseline_info.mode:
            case "Frame Mode":
                stim_frames = self._stimulus_frames_dict.values()
                stim_data = np.mean(np.concatenate(stim_frames, axis=0), axis=0)
                baseline_data = np.mean(
                    self.corrected_frames.data[
                        self._baseline_info.start : self._baseline_info.end
                    ],
                    axis=0,
                )
                return (stim_data - baseline_data) / baseline_data
            case "PreStim Mode":
                pre_frames_num = int(self._baseline_info.pre_sec * self.sampling_rate)
                widefield_change = []
                for start, stim_frames in self._stimulus_frames_dict.items():
                    baseline_data = np.mean(
                        self.corrected_frames.data[
                            max(0, start - pre_frames_num) : start
                        ],
                        axis=0,
                    )
                    stim_data = np.mean(stim_frames, axis=0)
                    widefield_change.append((stim_data - baseline_data) / baseline_data)
                return np.mean(np.concatenate(widefield_change, axis=0), axis=0)
            case _:
                return None


class ROIMixin(CorrectedFramesMixin):
    bound_points: utils.roi.BoundPoints

    def _get_init_bound_points(self) -> utils.roi.BoundPoints:
        self.bound_points = utils.roi.BoundPoints()
        return self.bound_points

    def load_bound_points(self, path: str) -> utils.roi.BoundPoints:
        if os.path.exists(path):
            self.bound_points.load(path)
            return self.bound_points
        else:
            return self.bound_points

    def clear_bound_points(self) -> utils.roi.BoundPoints:
        self.bound_points.clear()

    def _add_bound_points(self, point: tuple[int, int]) -> utils.roi.BoundPoints:
        self.bound_points.append(point)
        return self.bound_points

    def save_bound_points(self) -> None:
        self.bound_points.save(self.path.roi_bound_path)


class AnalyzedFigureMixin(ROIMixin, BaselineMixin, BasicFigureMixin):
    change_range: float
    analyzed_fig: plt.Figure
    analyzed_ax: plt.Axes
    colormap = ListedColormap(["dodgerblue", "navy", "black", "darkred", "gold"])

    def _init_analyzed_fig(self) -> plt.Figure:
        self.analyzed_fig, self.analyzed_ax = plt.subplots()
        self._zoom_dict = self._get_init_zoom_dict(self._raw_center)
        self.refresh_analyzed_fig()

    def refresh_analyzed_fig(self, no_roi: bool = False, save: bool = False) -> None:
        self.analyzed_ax.cla()
        img = self.analyzed_ax.imshow(
            self.widefield_change,
            cmap=self.colormap,
            vmin=-self.change_range,
            vmax=self.change_range,
        )
        cbar = self.analyzed_fig.colorbar(img, ax=self.analyzed_ax)
        cbar.set_label("%")
        cbar.set_ticks([-self.change_range, 0, self.change_range])
        cbar.set_ticklabels(
            [f"-{self.change_range * 100}", "0", f"{self.change_range * 100}"]
        )
        if not save:
            self.analyzed_ax.set_xlim(*self._zoom_dict.xlim)
            self.analyzed_ax.set_ylim(*self._zoom_dict.ylim)
        if not no_roi:
            self.analyzed_ax.plot(
                self.bound_points.bound_x,
                self.bound_points.bound_y,
                "o-",
                color="white",
                markersize=self._get_point_size(self.analyzed_fig),
            )
        self.analyzed_ax.axis("on")
        self.analyzed_fig.canvas.draw()

    def assign_change_range(self, value: str) -> float:
        self.change_range = float(value.strip()) / 100
        self.refresh_analyzed_fig()
        return self.change_range

    def load_stimulus(self, path: str) -> utils.data.StimLog:
        stimulus = super().load_stimulus(path)
        self.refresh_analyzed_fig()
        return stimulus

    def clear_stimulus(self) -> utils.data.StimLog:
        stimulus = super().clear_stimulus()
        self.refresh_analyzed_fig()
        return stimulus

    def add_stimulus(self, start: int, end: int) -> utils.data.StimLog:
        stimulus = super().add_stimulus(start, end)
        self.refresh_analyzed_fig()
        return stimulus

    def remove_stimulus(self, idx: int) -> utils.data.StimLog:
        stimulus = super().remove_stimulus(idx)
        self.refresh_analyzed_fig()
        return stimulus

    def load_bound_points(self, path: str) -> utils.roi.BoundPoints:
        bound_points = super().load_bound_points(path)
        self.refresh_analyzed_fig()
        return bound_points

    def clear_bound_points(self) -> utils.roi.BoundPoints:
        bound_points = super().clear_bound_points()
        self.refresh_analyzed_fig()
        return bound_points

    def reset_bound_points(self):
        self._zoom_dict = self._get_init_zoom_dict(self._raw_center)
        self.refresh_analyzed_fig()

    def assign_baseline_info(
        self,
        mode: Literal["Frame Mode", "PreStim Mode"],
        start: int,
        end: int,
        pre_sec: float,
    ) -> utils.data.BaselineInfo:
        baselinee_info = super().assign_baseline_info(mode, start, end, pre_sec)
        self.refresh_analyzed_fig()
        return baselinee_info

    @property
    def roi_change(self) -> np.ndarray:
        if len(self.bound_points.points) < 3:
            return np.array([])
        else:
            return utils.roi.get_roi_average(
                self.corrected_frames.data,
                self.bound_points.bound,
            )

    def save_analyzed_fig(self) -> None:
        self.refresh_analyzed_fig(no_roi=True, save=True)
        self.analyzed_fig.savefig(self.path.image_path)
        self._baseline_info.save(self.path.img_baseline_path)
        self.stimulus.save(self.path.stimulus_path)
        with open(self.path.range_path, "w") as f:
            f.write(f"{self.change_range * 100} %")

    def save_roi_change(self) -> None:
        self.refresh_analyzed_fig(no_roi=False, save=True)
        self.analyzed_fig.savefig(self.path.roi_image_path)
        self.stimulus.save(self.path.roi_stimulus_path)
        self.bound_points.save(self.path.roi_bound_path)
        self._baseline_info.save(self.path.roi_baseline_path)
        with open(self.path.roi_range_path, "w") as f:
            f.write(f"{self.change_range * 100} %")
        df = pd.DataFrame({"ROI Change": self.roi_change})
        df.to_csv(self.path.roi_change_path, index=False)


# ---------------------------- UI Database ----------------------------
# TODO: Today Finish!!!!
class UIDataBase(AnalyzedFigureMixin):
    path: utils.basic.Path
    frames: utils.data.Frames
    frames_time: utils.data.FrameTimes
    # Additional Information
    sampling_rate: str
    anchor: utils.anchor.Anchor
    corrected_frames: utils.data.CorrectedFrames
    # Analysis
    stim_log: utils.data.StimLog
    bound_points: utils.roi.BoundPoints

    # Default Values
    SAMPLING_RATE = None
    CORRECTED_FRAMES = None
    STIM_LOG = utils.data.StimLog()
    BOUND_POINTS = utils.roi.BoundPoints()

    # Default Values for UI
    ROTATE_DEGREE = "0"
    ACTIVE_POINT = None
    MASK_INVISIBLE = False
    VALUE_RANGE = "5"
    ACTIVE_MODE = None
    FRAME_MODE_START = None
    FRAME_MODE_END = None
    PRESTIM_MODE_DURATION = None

    def __init__(self, path: str) -> None:
        # Basic Information
        self.path = utils.basic.Path(path)
        self.frames = utils.data.Frames(self.path.path)
        self.frames_time = utils.data.FrameTimes(self.path.frametimes_path)
        # Additional Information
        self.sampling_rate = self._get_sampling_rate()
        self.anchor = self._get_anchor()
        self.corrected_frames = self._get_corrected_frames()
        self.stim_log = self._get_stim_log()
        self.bound_points = self._get_bound_points()
        # Generate Contents

import os
import sys
from tkinter import filedialog
from typing import Callable, Literal, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from dorsal_map import DorsalMap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.interpolate import interp1d


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
        self._zoom_dict = self._get_init_zoom_dict(self._padded_center)
        self._raw_fig.canvas.mpl_connect("button_press_event", self.click_on_raw_fig)
        self._raw_fig.canvas.mpl_connect("scroll_event", self.scroll_on_raw_fig)
        self._raw_fig.canvas.mpl_connect("resize_event", self.resize_raw_fig)
        self.refresh_raw_fig()

    def _init_perpectived_fig(self) -> plt.Figure:
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

    def _load_corrected_frames(self) -> str:
        if os.path.exists(self.path.corrected_frames_path):
            self.corrected_frames = utils.data.CorrectedFrames(
                self.path.corrected_frames_path
            )
            return 'No need to "Preprocess".'
        else:
            self.corrected_frames = None
            return 'Please "Preprocess" before "Analyze".'

    def preprocess(self, func: Optional[Callable] = None) -> utils.data.CorrectedFrames:
        self.corrected_frames = self._generate_corrected_frames(func)
        self.corrected_frames.save(self.path.corrected_frames_path)
        self.message = "Preprocess Done."
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

    def _get_init_baseline_info(self) -> utils.data.BaselineInfo:
        self._baseline_info = utils.data.BaselineInfo(
            mode="Frame Mode", start=0, end=0, pre_sec=0
        )
        return self._baseline_info

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
        self.analyzed_fig.savefig(self.path.image_path, dpi=300)
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

    def scroll_on_analyzed_fig(self, event) -> None:
        if event.button == "up":
            state = "in"
        elif event.button == "down":
            state = "out"
        else:
            return

        if (event.xdata is None) or (event.ydata is None):
            position = self._zoom_dict.position
        else:
            position = (event.xdata, event.ydata)

        upper, lower, left, right = utils.image.zoom(
            self._zoom_dict.upper + self._zoom_dict.position[1] - position[1],
            self._zoom_dict.lower + self._zoom_dict.position[1] - position[1],
            self._zoom_dict.left + self._zoom_dict.position[0] - position[0],
            self._zoom_dict.right + self._zoom_dict.position[0] - position[0],
            state,
        )
        self._zoom_dict = ZoomDict(position, upper, lower, left, right)
        self.refresh_raw_fig()


# ---------------------------- UI Database ----------------------------
class UIDataBase(AnalyzedFigureMixin):
    message: str

    def __init__(self) -> None:
        self._raw_fig, self._raw_ax = plt.subplots()
        self._perspective_adjusted_fig, self._perspective_adjusted_ax = plt.subplots()
        self.analyzed_fig, self.analyzed_ax = plt.subplots()

    def _load_sampling_rate(self) -> Union[float, None]:
        if os.path.exists(self.path.sampling_rate_path):
            with open(self.path.sampling_rate_path, "r") as f:
                self.sampling_rate = float(f.read())
            return None

    def assign_sampling_rate(self, value: str) -> float:
        self.sampling_rate = float(value.strip())
        with open(self.path.sampling_rate_path, "w") as f:
            f.write(str(self.sampling_rate))
        return self.sampling_rate

    def analyze(self) -> None:
        self.message = "Start Analyzing."
        self._get_init_baseline_info()
        self._init_analyzed_fig()

    def load_data(self) -> "UIDataBase":
        path = filedialog.askopenfilename(
            defaultextension=".dat",
            filetypes=[("dat", "*.dat"), ("tif", "*.tif"), ("mj2", "*.mj2")],
            title="Load Data",
        )
        self.path = utils.basic.Path(path)
        self._load_frames()
        self.assign_rotate_degree("0")
        self._rotate_img()
        self._load_sampling_rate()
        self._load_anchor()
        self.message = self._load_corrected_frames()
        self._get_init_stimulus()
        self._get_init_bound_points()
        self._init_raw_fig()
        self._init_perpectived_fig()

    @property
    def total_frames(self) -> int:
        return self.corrected_frames.total_frames

    def load_stimulus(self) -> utils.data.StimLog:
        path = filedialog.askopenfilename(
            initialdir=self.path.generate_directory,
            defaultextension=".csv",
            filetypes=[("csv", "*.csv")],
            title="Load Stimulus",
        )
        return super().load_stimulus(path)

    def load_bound_points(self) -> utils.roi.BoundPoints:
        path = filedialog.askopenfilename(
            initialdir=self.path.generate_directory,
            defaultextension=".csv",
            filetypes=[("csv", "*.csv")],
            title="Load ROI Bound Points",
        )
        return super().load_bound_points(path)


# ---------------------------- UI ----------------------------


class Ui(object):
    def __init__(self) -> None:
        self.ui_database = UIDataBase()
        self.app = QtWidgets.QApplication(sys.argv)
        self.Analyze = None
        self.Basic = QtWidgets.QWidget()
        self.setup_Basic_Ui(self.Basic)
        self.Basic.show()
        sys.exit(self.app.exec_())

    def load_data(self) -> None:
        self.ui_database = self.ui_database.load_data()
        self.TextBrowser.setText(self.ui_database.message)

    def reset_rotate_degree(self) -> None:
        self.ui_database.reset_rotate_degree()

    def assign_rotate_degree(self) -> None:
        self.ui_database.assign_rotate_degree(self.RotateDegree.text())

    def assign_sampling_rate(self) -> None:
        self.ui_database.assign_sampling_rate(self.SamplingRate.text())

    def analyze(self) -> None:
        if self.ui_database.corrected_frames is None:
            self.ui_database.preprocess(self.update_progress_bar)
        self.ui_database.analyze()
        self.TextBrowser.setText(self.ui_database.message)
        self.Analyze = QtWidgets.QWidget()
        self.setup_Analyze_Ui(self.Analyze)
        self.Analyze.show()
        sys.exit(self.app.exec_())

    def preprocess(self) -> None:
        self.ui_database.preprocess(self.update_progress_bar)
        self.TextBrowser.setText(self.ui_database.message)

    def update_progress_bar(self, value: float) -> None:
        self.ProgressBar.setValue(value)

    def activate_ob_left(self) -> None:
        self.ui_database.activate_ob_left()

    def activate_ob_center(self) -> None:
        self.ui_database.activate_ob_center()

    def activate_ob_right(self) -> None:
        self.ui_database.activate_ob_right()

    def activate_rsp_base(self) -> None:
        self.ui_database.activate_rsp_base()

    def mask_invisible(self) -> None:
        self.ui_database.mask_invisible = self.MaskInvisible.isChecked()

    def setup_Basic_Ui(self, Basic):
        Basic.setObjectName("Basic")
        Basic.resize(640, 480)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Basic)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Settings = QtWidgets.QFrame(Basic)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Settings.sizePolicy().hasHeightForWidth())
        self.Settings.setSizePolicy(sizePolicy)
        self.Settings.setMinimumSize(QtCore.QSize(0, 0))
        self.Settings.setMaximumSize(QtCore.QSize(190, 16777215))
        self.Settings.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Settings.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Settings.setObjectName("Settings")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.Settings)
        self.verticalLayout.setObjectName("verticalLayout")
        self.LoadButton = QtWidgets.QPushButton(self.Settings, clicked=self.load_data)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LoadButton.sizePolicy().hasHeightForWidth())
        self.LoadButton.setSizePolicy(sizePolicy)
        self.LoadButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.LoadButton.setFont(font)
        self.LoadButton.setObjectName("LoadButton")
        self.verticalLayout.addWidget(self.LoadButton)
        self.SamplingRateSetting = QtWidgets.QWidget(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.SamplingRateSetting.sizePolicy().hasHeightForWidth()
        )
        self.SamplingRateSetting.setSizePolicy(sizePolicy)
        self.SamplingRateSetting.setMinimumSize(QtCore.QSize(0, 40))
        self.SamplingRateSetting.setObjectName("SamplingRateSetting")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.SamplingRateSetting)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.SamplingRateLabel = QtWidgets.QLabel(self.SamplingRateSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.SamplingRateLabel.sizePolicy().hasHeightForWidth()
        )
        self.SamplingRateLabel.setSizePolicy(sizePolicy)
        self.SamplingRateLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.SamplingRateLabel.setFont(font)
        self.SamplingRateLabel.setObjectName("SamplingRateLabel")
        self.horizontalLayout_3.addWidget(
            self.SamplingRateLabel, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.SamplingRate = QtWidgets.QLineEdit(self.SamplingRateSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SamplingRate.sizePolicy().hasHeightForWidth())
        self.SamplingRate.setSizePolicy(sizePolicy)
        self.SamplingRate.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.SamplingRate.setFont(font)
        self.SamplingRate.setObjectName("SamplingRate")
        self.horizontalLayout_3.addWidget(self.SamplingRate)
        self.SamplingRateUnitLabel = QtWidgets.QLabel(self.SamplingRateSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.SamplingRateUnitLabel.sizePolicy().hasHeightForWidth()
        )
        self.SamplingRateUnitLabel.setSizePolicy(sizePolicy)
        self.SamplingRateUnitLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.SamplingRateUnitLabel.setFont(font)
        self.SamplingRateUnitLabel.setObjectName("SamplingRateUnitLabel")
        self.horizontalLayout_3.addWidget(
            self.SamplingRateUnitLabel,
            0,
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        )
        self.verticalLayout.addWidget(self.SamplingRateSetting)
        self.RotateSetting = QtWidgets.QFrame(self.Settings)
        self.RotateSetting.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.RotateSetting.setFrameShadow(QtWidgets.QFrame.Raised)
        self.RotateSetting.setObjectName("RotateSetting")
        self.gridLayout = QtWidgets.QGridLayout(self.RotateSetting)
        self.gridLayout.setObjectName("gridLayout")
        self.RotateLabel = QtWidgets.QLabel(self.RotateSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RotateLabel.sizePolicy().hasHeightForWidth())
        self.RotateLabel.setSizePolicy(sizePolicy)
        self.RotateLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.RotateLabel.setFont(font)
        self.RotateLabel.setObjectName("RotateLabel")
        self.gridLayout.addWidget(
            self.RotateLabel, 0, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.RotateResetButton = QtWidgets.QPushButton(
            self.RotateSetting, clicked=self.reset_rotate_degree
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.RotateResetButton.sizePolicy().hasHeightForWidth()
        )
        self.RotateResetButton.setSizePolicy(sizePolicy)
        self.RotateResetButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.RotateResetButton.setFont(font)
        self.RotateResetButton.setObjectName("RotateResetButton")
        self.gridLayout.addWidget(self.RotateResetButton, 0, 1, 1, 1)
        self.RotateConfirmButton = QtWidgets.QPushButton(
            self.RotateSetting, clicked=self.assign_rotate_degree
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.RotateConfirmButton.sizePolicy().hasHeightForWidth()
        )
        self.RotateConfirmButton.setSizePolicy(sizePolicy)
        self.RotateConfirmButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.RotateConfirmButton.setFont(font)
        self.RotateConfirmButton.setObjectName("RotateConfirmButton")
        self.gridLayout.addWidget(self.RotateConfirmButton, 0, 3, 1, 1)
        self.RotateDegree = QtWidgets.QLineEdit(self.RotateSetting, text="0")
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RotateDegree.sizePolicy().hasHeightForWidth())
        self.RotateDegree.setSizePolicy(sizePolicy)
        self.RotateDegree.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.RotateDegree.setFont(font)
        self.RotateDegree.setObjectName("RotateDegree")
        self.gridLayout.addWidget(self.RotateDegree, 1, 0, 1, 2)
        self.DegreeUnitLabel = QtWidgets.QLabel(self.RotateSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.DegreeUnitLabel.sizePolicy().hasHeightForWidth()
        )
        self.DegreeUnitLabel.setSizePolicy(sizePolicy)
        self.DegreeUnitLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.DegreeUnitLabel.setFont(font)
        self.DegreeUnitLabel.setObjectName("DegreeUnitLabel")
        self.gridLayout.addWidget(
            self.DegreeUnitLabel, 1, 3, 1, 1, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout.addWidget(self.RotateSetting)
        self.AnchorSetting = QtWidgets.QFrame(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.AnchorSetting.sizePolicy().hasHeightForWidth()
        )
        self.AnchorSetting.setSizePolicy(sizePolicy)
        self.AnchorSetting.setMinimumSize(QtCore.QSize(0, 100))
        self.AnchorSetting.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AnchorSetting.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AnchorSetting.setObjectName("AnchorSetting")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.AnchorSetting)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.AnchorLabel = QtWidgets.QLabel(self.AnchorSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AnchorLabel.sizePolicy().hasHeightForWidth())
        self.AnchorLabel.setSizePolicy(sizePolicy)
        self.AnchorLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.AnchorLabel.setFont(font)
        self.AnchorLabel.setObjectName("AnchorLabel")
        self.gridLayout_2.addWidget(
            self.AnchorLabel, 0, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.OBLeftButton = QtWidgets.QPushButton(
            self.AnchorSetting, clicked=self.activate_ob_left
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OBLeftButton.sizePolicy().hasHeightForWidth())
        self.OBLeftButton.setSizePolicy(sizePolicy)
        self.OBLeftButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.OBLeftButton.setFont(font)
        self.OBLeftButton.setObjectName("OBLeftButton")
        self.gridLayout_2.addWidget(self.OBLeftButton, 1, 0, 1, 1)
        self.OBRightButton = QtWidgets.QPushButton(
            self.AnchorSetting, clicked=self.activate_ob_right
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.OBRightButton.sizePolicy().hasHeightForWidth()
        )
        self.OBRightButton.setSizePolicy(sizePolicy)
        self.OBRightButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.OBRightButton.setFont(font)
        self.OBRightButton.setObjectName("OBRightButton")
        self.gridLayout_2.addWidget(self.OBRightButton, 1, 1, 1, 1)
        self.OBCenterButton = QtWidgets.QPushButton(
            self.AnchorSetting, clicked=self.activate_ob_center
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.OBCenterButton.sizePolicy().hasHeightForWidth()
        )
        self.OBCenterButton.setSizePolicy(sizePolicy)
        self.OBCenterButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.OBCenterButton.setFont(font)
        self.OBCenterButton.setObjectName("OBCenterButton")
        self.gridLayout_2.addWidget(self.OBCenterButton, 2, 0, 1, 1)
        self.RSPBaseButton = QtWidgets.QPushButton(
            self.AnchorSetting, clicked=self.activate_rsp_base
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.RSPBaseButton.sizePolicy().hasHeightForWidth()
        )
        self.RSPBaseButton.setSizePolicy(sizePolicy)
        self.RSPBaseButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.RSPBaseButton.setFont(font)
        self.RSPBaseButton.setObjectName("RSPBaseButton")
        self.gridLayout_2.addWidget(self.RSPBaseButton, 2, 1, 1, 1)
        self.verticalLayout.addWidget(self.AnchorSetting)
        self.MaskInvisible = QtWidgets.QCheckBox(
            self.Settings, clicked=self.mask_invisible
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.MaskInvisible.sizePolicy().hasHeightForWidth()
        )
        self.MaskInvisible.setSizePolicy(sizePolicy)
        self.MaskInvisible.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.MaskInvisible.setFont(font)
        self.MaskInvisible.setObjectName("MaskInvisible")
        self.verticalLayout.addWidget(self.MaskInvisible)
        self.ProcessSetting = QtWidgets.QWidget(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ProcessSetting.sizePolicy().hasHeightForWidth()
        )
        self.ProcessSetting.setSizePolicy(sizePolicy)
        self.ProcessSetting.setMinimumSize(QtCore.QSize(0, 40))
        self.ProcessSetting.setObjectName("ProcessSetting")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.ProcessSetting)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PreprocessButton = QtWidgets.QPushButton(
            self.ProcessSetting, clicked=self.preprocess
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.PreprocessButton.sizePolicy().hasHeightForWidth()
        )
        self.PreprocessButton.setSizePolicy(sizePolicy)
        self.PreprocessButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.PreprocessButton.setFont(font)
        self.PreprocessButton.setObjectName("PreprocessButton")
        self.horizontalLayout_2.addWidget(self.PreprocessButton)
        self.AnalyzeButton = QtWidgets.QPushButton(
            self.ProcessSetting, clicked=self.analyze
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.AnalyzeButton.sizePolicy().hasHeightForWidth()
        )
        self.AnalyzeButton.setSizePolicy(sizePolicy)
        self.AnalyzeButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.AnalyzeButton.setFont(font)
        self.AnalyzeButton.setObjectName("AnalyzeButton")
        self.horizontalLayout_2.addWidget(self.AnalyzeButton)
        self.verticalLayout.addWidget(self.ProcessSetting)
        self.ProgressBar = QtWidgets.QProgressBar(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ProgressBar.sizePolicy().hasHeightForWidth())
        self.ProgressBar.setSizePolicy(sizePolicy)
        self.ProgressBar.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ProgressBar.setFont(font)
        self.ProgressBar.setProperty("value", 0)
        self.ProgressBar.setObjectName("ProgressBar")
        self.verticalLayout.addWidget(self.ProgressBar)
        self.TextBrowser = QtWidgets.QTextBrowser(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TextBrowser.sizePolicy().hasHeightForWidth())
        self.TextBrowser.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.TextBrowser.setFont(font)
        self.TextBrowser.setObjectName("TextBrowser")
        self.verticalLayout.addWidget(self.TextBrowser)
        self.horizontalLayout.addWidget(self.Settings)
        self.ImageShower = QtWidgets.QFrame(Basic)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageShower.sizePolicy().hasHeightForWidth())
        self.ImageShower.setSizePolicy(sizePolicy)
        self.ImageShower.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ImageShower.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ImageShower.setObjectName("ImageShower")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.ImageShower)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.RawImageShower = QtWidgets.QFrame(self.ImageShower)
        self.RawImageShower.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.RawImageShower.setFrameShadow(QtWidgets.QFrame.Raised)
        self.RawImageShower.setObjectName("RawImageShower")
        # __
        self.raw_image_canvas = FigureCanvas(self.ui_database._raw_fig)
        self.raw_image_layout = QtWidgets.QVBoxLayout(self.RawImageShower)
        self.raw_image_layout.addWidget(self.raw_image_canvas)
        # ^^
        self.verticalLayout_2.addWidget(self.RawImageShower)
        self.PerspectiveImageShower = QtWidgets.QFrame(self.ImageShower)
        self.PerspectiveImageShower.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PerspectiveImageShower.setFrameShadow(QtWidgets.QFrame.Raised)
        self.PerspectiveImageShower.setObjectName("PerspectiveImageShower")
        # __
        self.perspective_image_canvas = FigureCanvas(
            self.ui_database._perspective_adjusted_fig
        )
        self.perspective_image_layout = QtWidgets.QVBoxLayout(
            self.PerspectiveImageShower
        )
        self.perspective_image_layout.addWidget(self.perspective_image_canvas)
        # ^^
        self.verticalLayout_2.addWidget(self.PerspectiveImageShower)
        self.horizontalLayout.addWidget(self.ImageShower)

        self.retranslate_Basic_Ui(Basic)
        QtCore.QMetaObject.connectSlotsByName(Basic)

    def retranslate_Basic_Ui(self, Basic):
        _translate = QtCore.QCoreApplication.translate
        Basic.setWindowTitle(_translate("Basic", "Preprocess"))
        self.LoadButton.setText(_translate("Basic", "Load"))
        self.SamplingRateLabel.setText(_translate("Basic", "Sampling Rate"))
        self.SamplingRateUnitLabel.setText(_translate("Basic", "Hz"))
        self.RotateLabel.setText(_translate("Basic", "Rotate"))
        self.RotateResetButton.setText(_translate("Basic", "Reset"))
        self.RotateConfirmButton.setText(_translate("Basic", "Confirm"))
        self.DegreeUnitLabel.setText(_translate("Basic", "degrees"))
        self.AnchorLabel.setText(_translate("Basic", "Anchor"))
        self.OBLeftButton.setText(_translate("Basic", "OB Left"))
        self.OBRightButton.setText(_translate("Basic", "OB Right"))
        self.OBCenterButton.setText(_translate("Basic", "OB Center"))
        self.RSPBaseButton.setText(_translate("Basic", "RSP Base"))
        self.MaskInvisible.setText(_translate("Basic", "Mask Invisible"))
        self.PreprocessButton.setText(_translate("Basic", "Preprocess"))
        self.AnalyzeButton.setText(_translate("Basic", "Analyze"))

    def load_stimulus(self) -> None:
        self.ui_database.load_stimulus()
        self.StimulusList.clear()
        self.StimulusList.addItems(self.ui_database.stimulus.content)

    def clear_stimulus(self) -> None:
        self.ui_database.clear_stimulus()
        self.StimulusList.clear()
        self.StimulusList.addItems(self.ui_database.stimulus.content)

    def add_stimulus(self) -> None:
        self.ui_database.add_stimulus(
            start=int(self.StimulusStart.text().strip()),
            end=int(self.StimulusEnd.text().strip()),
        )
        self.StimulusList.clear()
        self.StimulusList.addItems(self.ui_database.stimulus.content)

    def remove_stimulus(self) -> None:
        selected = self.StimulusList.currentRow()
        if len(selected) == 0:
            return
        self.ui_database.remove_stimulus(idx=int(selected))
        self.StimulusList.clear()
        self.StimulusList.addItems(self.ui_database.stimulus.content)

    def assign_range(self) -> None:
        self.ui_database.assign_change_range(self.ImageRangePercentage.text())

    def save_range(self) -> None:
        self.ui_database.save_analyzed_fig()

    def load_roi(self) -> None:
        self.ui_database.load_bound_points()

    def clear_roi(self) -> None:
        self.ui_database.clear_bound_points()

    def reset_roi(self) -> None:
        self.ui_database.reset_bound_points()

    def save_roi(self) -> None:
        self.ui_database.save_roi_change()

    def assign_baseline_info(self) -> None:
        start = self.BaselineStart.text().strip()
        end = self.BaselineEnd.text().strip()
        pre_sec = self.PreStimModeProSecs.text().strip()
        if self.FrameModeButton.isChecked():
            mode = "Frame Mode"
        elif self.PreStimModeButton.isChecked():
            mode = "PreStim Mode"
        else:
            return
        self.ui_database.assign_baseline_info(
            mode=mode,
            start=int(start) if start else 0,
            end=int(end) if end else self.ui_database.total_frames - 1,
            pre_sec=float(pre_sec) if pre_sec else 1.0,
        )

    def setup_Analyze_Ui(self, Analyze):
        Analyze.setObjectName("Analyze")
        Analyze.resize(640, 480)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Analyze)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Settings = QtWidgets.QFrame(Analyze)
        self.Settings.setMaximumSize(QtCore.QSize(240, 16777215))
        self.Settings.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Settings.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Settings.setObjectName("Settings")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.Settings)
        self.verticalLayout.setObjectName("verticalLayout")
        self.TotalFrames = QtWidgets.QLabel(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TotalFrames.sizePolicy().hasHeightForWidth())
        self.TotalFrames.setSizePolicy(sizePolicy)
        self.TotalFrames.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.TotalFrames.setFont(font)
        self.TotalFrames.setObjectName("TotalFrames")
        self.verticalLayout.addWidget(self.TotalFrames)
        self.SamplingRateLabel = QtWidgets.QLabel(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.SamplingRateLabel.sizePolicy().hasHeightForWidth()
        )
        self.SamplingRateLabel.setSizePolicy(sizePolicy)
        self.SamplingRateLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.SamplingRateLabel.setFont(font)
        self.SamplingRateLabel.setObjectName("SamplingRateLabel")
        self.verticalLayout.addWidget(self.SamplingRateLabel)
        self.Stimulus = QtWidgets.QFrame(self.Settings)
        self.Stimulus.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Stimulus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Stimulus.setObjectName("Stimulus")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.Stimulus)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.StimulusLoader = QtWidgets.QWidget(self.Stimulus)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusLoader.sizePolicy().hasHeightForWidth()
        )
        self.StimulusLoader.setSizePolicy(sizePolicy)
        self.StimulusLoader.setMinimumSize(QtCore.QSize(0, 40))
        self.StimulusLoader.setObjectName("StimulusLoader")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.StimulusLoader)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.StimulusLabel = QtWidgets.QLabel(self.StimulusLoader)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusLabel.sizePolicy().hasHeightForWidth()
        )
        self.StimulusLabel.setSizePolicy(sizePolicy)
        self.StimulusLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.StimulusLabel.setFont(font)
        self.StimulusLabel.setObjectName("StimulusLabel")
        self.horizontalLayout_3.addWidget(self.StimulusLabel)
        self.StimulusLoadButton = QtWidgets.QPushButton(
            self.StimulusLoader, clicked=self.load_stimulus
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusLoadButton.sizePolicy().hasHeightForWidth()
        )
        self.StimulusLoadButton.setSizePolicy(sizePolicy)
        self.StimulusLoadButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusLoadButton.setFont(font)
        self.StimulusLoadButton.setObjectName("StimulusLoadButton")
        self.horizontalLayout_3.addWidget(self.StimulusLoadButton)
        self.StimulusClearButton = QtWidgets.QPushButton(
            self.StimulusLoader, clicked=self.clear_stimulus
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusClearButton.sizePolicy().hasHeightForWidth()
        )
        self.StimulusClearButton.setSizePolicy(sizePolicy)
        self.StimulusClearButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusClearButton.setFont(font)
        self.StimulusClearButton.setObjectName("StimulusClearButton")
        self.horizontalLayout_3.addWidget(self.StimulusClearButton)
        self.verticalLayout_3.addWidget(self.StimulusLoader)
        self.StimulusManager = QtWidgets.QFrame(self.Stimulus)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusManager.sizePolicy().hasHeightForWidth()
        )
        self.StimulusManager.setSizePolicy(sizePolicy)
        self.StimulusManager.setMinimumSize(QtCore.QSize(0, 70))
        self.StimulusManager.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.StimulusManager.setFrameShadow(QtWidgets.QFrame.Raised)
        self.StimulusManager.setObjectName("StimulusManager")
        self.gridLayout = QtWidgets.QGridLayout(self.StimulusManager)
        self.gridLayout.setObjectName("gridLayout")
        self.StimulusStartLabel = QtWidgets.QLabel(self.StimulusManager)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusStartLabel.sizePolicy().hasHeightForWidth()
        )
        self.StimulusStartLabel.setSizePolicy(sizePolicy)
        self.StimulusStartLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusStartLabel.setFont(font)
        self.StimulusStartLabel.setObjectName("StimulusStartLabel")
        self.gridLayout.addWidget(self.StimulusStartLabel, 0, 0, 1, 1)
        self.StimulusStart = QtWidgets.QLineEdit(self.StimulusManager)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusStart.sizePolicy().hasHeightForWidth()
        )
        self.StimulusStart.setSizePolicy(sizePolicy)
        self.StimulusStart.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusStart.setFont(font)
        self.StimulusStart.setObjectName("StimulusStart")
        self.gridLayout.addWidget(self.StimulusStart, 0, 1, 1, 1)
        self.StimulusEndLabel = QtWidgets.QLabel(self.StimulusManager)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.StimulusEndLabel.sizePolicy().hasHeightForWidth()
        )
        self.StimulusEndLabel.setSizePolicy(sizePolicy)
        self.StimulusEndLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusEndLabel.setFont(font)
        self.StimulusEndLabel.setObjectName("StimulusEndLabel")
        self.gridLayout.addWidget(self.StimulusEndLabel, 0, 2, 1, 1)
        self.StimulusEnd = QtWidgets.QLineEdit(self.StimulusManager)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StimulusEnd.sizePolicy().hasHeightForWidth())
        self.StimulusEnd.setSizePolicy(sizePolicy)
        self.StimulusEnd.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusEnd.setFont(font)
        self.StimulusEnd.setObjectName("StimulusEnd")
        self.gridLayout.addWidget(self.StimulusEnd, 0, 3, 1, 1)
        self.AddStimulusButton = QtWidgets.QPushButton(
            self.StimulusManager, clicked=self.add_stimulus
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.AddStimulusButton.sizePolicy().hasHeightForWidth()
        )
        self.AddStimulusButton.setSizePolicy(sizePolicy)
        self.AddStimulusButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.AddStimulusButton.setFont(font)
        self.AddStimulusButton.setObjectName("AddStimulusButton")
        self.gridLayout.addWidget(self.AddStimulusButton, 1, 0, 1, 2)
        self.RemoveStimulusButton = QtWidgets.QPushButton(
            self.StimulusManager, clicked=self.remove_stimulus
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.RemoveStimulusButton.sizePolicy().hasHeightForWidth()
        )
        self.RemoveStimulusButton.setSizePolicy(sizePolicy)
        self.RemoveStimulusButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.RemoveStimulusButton.setFont(font)
        self.RemoveStimulusButton.setObjectName("RemoveStimulusButton")
        self.gridLayout.addWidget(self.RemoveStimulusButton, 1, 2, 1, 2)
        self.verticalLayout_3.addWidget(self.StimulusManager)
        self.StimulusList = QtWidgets.QListWidget(self.Stimulus)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StimulusList.sizePolicy().hasHeightForWidth())
        self.StimulusList.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.StimulusList.setFont(font)
        self.StimulusList.setObjectName("StimulusList")
        self.verticalLayout_3.addWidget(self.StimulusList)
        self.verticalLayout.addWidget(self.Stimulus)
        self.ImageRangeSetting = QtWidgets.QFrame(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangeSetting.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangeSetting.setSizePolicy(sizePolicy)
        self.ImageRangeSetting.setMinimumSize(QtCore.QSize(0, 40))
        self.ImageRangeSetting.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ImageRangeSetting.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ImageRangeSetting.setObjectName("ImageRangeSetting")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.ImageRangeSetting)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.ImageRangeLabel = QtWidgets.QLabel(self.ImageRangeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangeLabel.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangeLabel.setSizePolicy(sizePolicy)
        self.ImageRangeLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.ImageRangeLabel.setFont(font)
        self.ImageRangeLabel.setObjectName("ImageRangeLabel")
        self.horizontalLayout_5.addWidget(self.ImageRangeLabel)
        self.ImageRangePercentage = QtWidgets.QLineEdit(self.ImageRangeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangePercentage.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangePercentage.setSizePolicy(sizePolicy)
        self.ImageRangePercentage.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ImageRangePercentage.setFont(font)
        self.ImageRangePercentage.setObjectName("ImageRangePercentage")
        self.horizontalLayout_5.addWidget(self.ImageRangePercentage)
        self.ImageRangeUnit = QtWidgets.QLabel(self.ImageRangeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangeUnit.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangeUnit.setSizePolicy(sizePolicy)
        self.ImageRangeUnit.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ImageRangeUnit.setFont(font)
        self.ImageRangeUnit.setObjectName("ImageRangeUnit")
        self.horizontalLayout_5.addWidget(self.ImageRangeUnit)
        self.ImageRangeConfirmButton = QtWidgets.QPushButton(
            self.ImageRangeSetting, clicked=self.assign_range
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangeConfirmButton.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangeConfirmButton.setSizePolicy(sizePolicy)
        self.ImageRangeConfirmButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ImageRangeConfirmButton.setFont(font)
        self.ImageRangeConfirmButton.setObjectName("ImageRangeConfirmButton")
        self.horizontalLayout_5.addWidget(self.ImageRangeConfirmButton)
        self.ImageRangeSaveButton = QtWidgets.QPushButton(
            self.ImageRangeSetting, clicked=self.save_range
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ImageRangeSaveButton.sizePolicy().hasHeightForWidth()
        )
        self.ImageRangeSaveButton.setSizePolicy(sizePolicy)
        self.ImageRangeSaveButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ImageRangeSaveButton.setFont(font)
        self.ImageRangeSaveButton.setObjectName("ImageRangeSaveButton")
        self.horizontalLayout_5.addWidget(self.ImageRangeSaveButton)
        self.verticalLayout.addWidget(self.ImageRangeSetting)
        self.ROISetting = QtWidgets.QFrame(self.Settings)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ROISetting.sizePolicy().hasHeightForWidth())
        self.ROISetting.setSizePolicy(sizePolicy)
        self.ROISetting.setMinimumSize(QtCore.QSize(0, 70))
        self.ROISetting.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ROISetting.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ROISetting.setObjectName("ROISetting")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.ROISetting)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.ROILabel = QtWidgets.QLabel(self.ROISetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ROILabel.sizePolicy().hasHeightForWidth())
        self.ROILabel.setSizePolicy(sizePolicy)
        self.ROILabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.ROILabel.setFont(font)
        self.ROILabel.setObjectName("ROILabel")
        self.gridLayout_2.addWidget(self.ROILabel, 0, 0, 2, 1)
        self.ROILoadButton = QtWidgets.QPushButton(
            self.ROISetting, clicked=self.load_roi
        )
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ROILoadButton.setFont(font)
        self.ROILoadButton.setObjectName("ROILoadButton")
        self.gridLayout_2.addWidget(self.ROILoadButton, 0, 1, 1, 1)
        self.ROIRemoveButton = QtWidgets.QPushButton(
            self.ROISetting, clicked=self.reset_roi
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ROIRemoveButton.sizePolicy().hasHeightForWidth()
        )
        self.ROIRemoveButton.setSizePolicy(sizePolicy)
        self.ROIRemoveButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ROIRemoveButton.setFont(font)
        self.ROIRemoveButton.setObjectName("ROIRemoveButton")
        self.gridLayout_2.addWidget(self.ROIRemoveButton, 0, 2, 1, 1)
        self.ROIClearButton = QtWidgets.QPushButton(
            self.ROISetting, clicked=self.clear_roi
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ROIClearButton.sizePolicy().hasHeightForWidth()
        )
        self.ROIClearButton.setSizePolicy(sizePolicy)
        self.ROIClearButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ROIClearButton.setFont(font)
        self.ROIClearButton.setObjectName("ROIClearButton")
        self.gridLayout_2.addWidget(self.ROIClearButton, 1, 1, 1, 1)
        self.ROISaveButton = QtWidgets.QPushButton(
            self.ROISetting, clicked=self.save_roi
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ROISaveButton.sizePolicy().hasHeightForWidth()
        )
        self.ROISaveButton.setSizePolicy(sizePolicy)
        self.ROISaveButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ROISaveButton.setFont(font)
        self.ROISaveButton.setObjectName("ROISaveButton")
        self.gridLayout_2.addWidget(self.ROISaveButton, 1, 2, 1, 1)
        self.verticalLayout.addWidget(self.ROISetting)
        self.horizontalLayout.addWidget(self.Settings)
        self.BaselingSettingandImage = QtWidgets.QFrame(Analyze)
        self.BaselingSettingandImage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.BaselingSettingandImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.BaselingSettingandImage.setObjectName("BaselingSettingandImage")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.BaselingSettingandImage)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.BaselineSetting = QtWidgets.QFrame(self.BaselingSettingandImage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.BaselineSetting.sizePolicy().hasHeightForWidth()
        )
        self.BaselineSetting.setSizePolicy(sizePolicy)
        self.BaselineSetting.setMinimumSize(QtCore.QSize(0, 200))
        self.BaselineSetting.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.BaselineSetting.setFrameShadow(QtWidgets.QFrame.Raised)
        self.BaselineSetting.setObjectName("BaselineSetting")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.BaselineSetting)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.BaselineLabel = QtWidgets.QLabel(self.BaselineSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.BaselineLabel.sizePolicy().hasHeightForWidth()
        )
        self.BaselineLabel.setSizePolicy(sizePolicy)
        self.BaselineLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.BaselineLabel.setFont(font)
        self.BaselineLabel.setObjectName("BaselineLabel")
        self.verticalLayout_2.addWidget(self.BaselineLabel)
        self.FrameModeButton = QtWidgets.QRadioButton(self.BaselineSetting)
        self.FrameModeButton.toggled.connect(self.assign_baseline_info)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.FrameModeButton.sizePolicy().hasHeightForWidth()
        )
        self.FrameModeButton.setSizePolicy(sizePolicy)
        self.FrameModeButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.FrameModeButton.setFont(font)
        self.FrameModeButton.setObjectName("FrameModeButton")
        self.verticalLayout_2.addWidget(self.FrameModeButton)
        self.FrameModeSetting = QtWidgets.QWidget(self.BaselineSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.FrameModeSetting.sizePolicy().hasHeightForWidth()
        )
        self.FrameModeSetting.setSizePolicy(sizePolicy)
        self.FrameModeSetting.setMinimumSize(QtCore.QSize(0, 40))
        self.FrameModeSetting.setObjectName("FrameModeSetting")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.FrameModeSetting)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.BaselineStartLabel = QtWidgets.QLabel(self.FrameModeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.BaselineStartLabel.sizePolicy().hasHeightForWidth()
        )
        self.BaselineStartLabel.setSizePolicy(sizePolicy)
        self.BaselineStartLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.BaselineStartLabel.setFont(font)
        self.BaselineStartLabel.setObjectName("BaselineStartLabel")
        self.horizontalLayout_4.addWidget(self.BaselineStartLabel)
        self.BaselineStart = QtWidgets.QLineEdit(self.FrameModeSetting)
        self.BaselineStart.editingFinished.connect(self.assign_baseline_info)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.BaselineStart.sizePolicy().hasHeightForWidth()
        )
        self.BaselineStart.setSizePolicy(sizePolicy)
        self.BaselineStart.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.BaselineStart.setFont(font)
        self.BaselineStart.setObjectName("BaselineStart")
        self.horizontalLayout_4.addWidget(self.BaselineStart)
        self.BaselineEndLabel = QtWidgets.QLabel(self.FrameModeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.BaselineEndLabel.sizePolicy().hasHeightForWidth()
        )
        self.BaselineEndLabel.setSizePolicy(sizePolicy)
        self.BaselineEndLabel.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.BaselineEndLabel.setFont(font)
        self.BaselineEndLabel.setObjectName("BaselineEndLabel")
        self.horizontalLayout_4.addWidget(self.BaselineEndLabel)
        self.BaselineEnd = QtWidgets.QLineEdit(self.FrameModeSetting)
        self.BaselineEnd.editingFinished.connect(self.assign_baseline_info)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BaselineEnd.sizePolicy().hasHeightForWidth())
        self.BaselineEnd.setSizePolicy(sizePolicy)
        self.BaselineEnd.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.BaselineEnd.setFont(font)
        self.BaselineEnd.setObjectName("BaselineEnd")
        self.horizontalLayout_4.addWidget(self.BaselineEnd)
        self.verticalLayout_2.addWidget(self.FrameModeSetting)
        self.PreStimModeButton = QtWidgets.QRadioButton(self.BaselineSetting)
        self.PreStimModeButton.toggled.connect(self.assign_baseline_info)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.PreStimModeButton.sizePolicy().hasHeightForWidth()
        )
        self.PreStimModeButton.setSizePolicy(sizePolicy)
        self.PreStimModeButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        self.PreStimModeButton.setFont(font)
        self.PreStimModeButton.setObjectName("PreStimModeButton")
        self.verticalLayout_2.addWidget(self.PreStimModeButton)
        self.PreStimModeSetting = QtWidgets.QWidget(self.BaselineSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.PreStimModeSetting.sizePolicy().hasHeightForWidth()
        )
        self.PreStimModeSetting.setSizePolicy(sizePolicy)
        self.PreStimModeSetting.setMinimumSize(QtCore.QSize(0, 40))
        self.PreStimModeSetting.setObjectName("PreStimModeSetting")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.PreStimModeSetting)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PreStimModeProSecs = QtWidgets.QLineEdit(self.PreStimModeSetting)
        self.PreStimModeProSecs.editingFinished.connect(self.assign_baseline_info)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.PreStimModeProSecs.sizePolicy().hasHeightForWidth()
        )
        self.PreStimModeProSecs.setSizePolicy(sizePolicy)
        self.PreStimModeProSecs.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.PreStimModeProSecs.setFont(font)
        self.PreStimModeProSecs.setObjectName("PreStimModeProSecs")
        self.horizontalLayout_2.addWidget(self.PreStimModeProSecs)
        self.PreStimModeUnit = QtWidgets.QLabel(self.PreStimModeSetting)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.PreStimModeUnit.sizePolicy().hasHeightForWidth()
        )
        self.PreStimModeUnit.setSizePolicy(sizePolicy)
        self.PreStimModeUnit.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.PreStimModeUnit.setFont(font)
        self.PreStimModeUnit.setObjectName("PreStimModeUnit")
        self.horizontalLayout_2.addWidget(self.PreStimModeUnit)
        self.verticalLayout_2.addWidget(self.PreStimModeSetting)
        self.verticalLayout_4.addWidget(self.BaselineSetting)
        self.AnalyzedImageShower = QtWidgets.QFrame(self.BaselingSettingandImage)
        # __
        self.analyzed_image_canvas = FigureCanvas(self.ui_database.analyzed_fig)
        self.analyzed_image_layout = QtWidgets.QVBoxLayout(self.AnalyzedImageShower)
        self.analyzed_image_layout.addWidget(self.analyzed_image_canvas)
        # ^^
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.AnalyzedImageShower.sizePolicy().hasHeightForWidth()
        )
        self.AnalyzedImageShower.setSizePolicy(sizePolicy)
        self.AnalyzedImageShower.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AnalyzedImageShower.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AnalyzedImageShower.setObjectName("AnalyzedImageShower")
        self.verticalLayout_4.addWidget(self.AnalyzedImageShower)
        self.horizontalLayout.addWidget(self.BaselingSettingandImage)

        self.retranslate_Analyze_Ui(Analyze)
        QtCore.QMetaObject.connectSlotsByName(Analyze)

    def retranslate_Analyze_Ui(self, Analyze):
        _translate = QtCore.QCoreApplication.translate
        Analyze.setWindowTitle(_translate("Analyze", "Analyze"))
        self.TotalFrames.setText(
            _translate("Analyze", f"Total Frames: {self.ui_database.total_frames}")
        )
        self.SamplingRateLabel.setText(
            _translate("Analyze", f"Sampling Rate: {self.ui_database.sampling_rate}")
        )
        self.StimulusLabel.setText(_translate("Analyze", "Stimulus"))
        self.StimulusLoadButton.setText(_translate("Analyze", "Load"))
        self.StimulusClearButton.setText(_translate("Analyze", "Clear"))
        self.StimulusStartLabel.setText(_translate("Analyze", "Start"))
        self.StimulusEndLabel.setText(_translate("Analyze", "End"))
        self.AddStimulusButton.setText(_translate("Analyze", "Add"))
        self.RemoveStimulusButton.setText(_translate("Analyze", "Remove"))
        self.ImageRangeLabel.setText(_translate("Analyze", "Range"))
        self.ImageRangeUnit.setText(_translate("Analyze", "%"))
        self.ImageRangeConfirmButton.setText(_translate("Analyze", "Confirm"))
        self.ImageRangeSaveButton.setText(_translate("Analyze", "Save"))
        self.ROILabel.setText(_translate("Analyze", "ROI"))
        self.ROILoadButton.setText(_translate("Analyze", "Load"))
        self.ROIRemoveButton.setText(_translate("Analyze", "Reset"))
        self.ROIClearButton.setText(_translate("Analyze", "Clear"))
        self.ROISaveButton.setText(_translate("Analyze", "Save"))
        self.BaselineLabel.setText(_translate("Analyze", "Baseline"))
        self.FrameModeButton.setText(_translate("Analyze", "Frame Mode"))
        self.BaselineStartLabel.setText(_translate("Analyze", "Start"))
        self.BaselineEndLabel.setText(_translate("Analyze", "End"))
        self.PreStimModeButton.setText(_translate("Analyze", "PreStim Mode"))
        self.PreStimModeUnit.setText(_translate("Analyze", "seconds"))


if __name__ == "__main__":
    import sys

    ui = Ui()

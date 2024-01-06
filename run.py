import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.interpolate import interp1d

from analysis import utils
from analysis.dorsal_map import DorsalMap


class AnchorLocater:
    def __init__(
        self, frames: utils.Frames, dorsal_map: DorsalMap, path: utils.Path
    ) -> None:
        self.anchor = utils.Anchor()
        self.frames = frames
        self.dorsal_map = dorsal_map
        self.transformer = None
        self.active_strucutre = None
        self.active_mask = True
        self.raw_fig, self.raw_ax = plt.subplots()
        self.raw_fig_canvas = FigureCanvas(self.raw_fig)
        self.perspective_fig, self.perspective_ax = plt.subplots()
        self.perspective_fig_canvas = FigureCanvas(self.perspective_fig)
        self.raw_fig.canvas.mpl_connect("button_press_event", self._click_raw)
        self.raw_fig.canvas.mpl_connect("resize_event", self._resize_raw)
        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QWidget()
        self.setup_ui(window)
        window.show()
        _ = app.exec_()
        self.anchor.save(path.generate_dir.log_dir.anchor_path)

    def _click_raw(self, event) -> None:
        if event.button == 1 and self.active_strucutre is not None:
            if (event.xdata is not None) and (event.ydata is not None):
                point = (float(event.xdata), float(event.ydata))
                match self.active_strucutre:
                    case "ob_left":
                        self.anchor.ob_left = point
                    case "ob_center":
                        self.anchor.ob_center = point
                    case "ob_right":
                        self.anchor.ob_right = point
                    case "rsp_base":
                        self.anchor.rsp_base = point
                self._refresh_raw_fig()
                self._refresh_perspective_fig()

    def _resize_raw(self, event) -> None:
        self._refresh_raw_fig()

    @property
    def _from_points(self) -> list[tuple[int, int]]:
        return [
            getattr(self.anchor, point)
            for point in ["ob_left", "ob_center", "ob_right", "rsp_base"]
            if getattr(self.anchor, point) != (0, 0)
        ]

    @property
    def _to_points(self) -> list[tuple[int, int]]:
        return [
            getattr(self.dorsal_map, point)
            for point in ["ob_left", "ob_center", "ob_right", "rsp_base"]
        ]

    def _refresh_raw_fig(self) -> None:
        self.raw_ax.cla()
        self.frames.average
        self.raw_ax.imshow(self.frames.average[0], cmap="hot")
        points = self._from_points
        xs, ys = zip(*points)
        point_size = min(*self.raw_fig.get_size_inches()) * self.raw_fig.dpi / 50
        self.raw_ax.plot(xs, ys, "o", color="blue", markersize=point_size)
        self.raw_ax.axis("on")
        self.raw_fig.canvas.draw()

    def _refresh_perspective_fig(self) -> None:
        if len(self._from_points) < 4:
            return
        else:
            self.transformer = utils.PerspectiveTransformer(
                self._from_points, self._to_points
            )
        self.perspective_ax.cla()
        self.perspective_ax.imshow(
            self.transformer.perspective_transform(self.frames.average[0]), cmap="hot"
        )
        if self.active_mask:
            self.dorsal_map.mask_axes(self.perspective_ax)
        self.perspective_ax.axis("on")
        self.perspective_fig.canvas.draw()

    def setup_ui(self, window):
        window.setObjectName("window")
        window.resize(640, 480)
        self.window_layout = QtWidgets.QVBoxLayout(window)
        self.window_layout.setObjectName("window_layout")
        self.widget = QtWidgets.QWidget(window)
        self.widget.setObjectName("widget")
        self.display_image_layout = QtWidgets.QHBoxLayout(self.widget)
        self.display_image_layout.setObjectName("display_image_layout")
        self.raw_image_displayer = QtWidgets.QFrame(self.widget)
        self.raw_image_displayer.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.raw_image_displayer.setFrameShadow(QtWidgets.QFrame.Raised)
        self.raw_image_displayer.setObjectName("raw_image_displayer")
        self.raw_image_layout = QtWidgets.QVBoxLayout(self.raw_image_displayer)
        self.raw_image_layout.addWidget(self.raw_fig_canvas)
        self.raw_image_layout.setObjectName("raw_image_layout")
        self.display_image_layout.addWidget(self.raw_image_displayer)
        self.perspective_image_displayer = QtWidgets.QFrame(self.widget)
        self.perspective_image_displayer.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.perspective_image_displayer.setFrameShadow(QtWidgets.QFrame.Raised)
        self.perspective_image_displayer.setObjectName("perspective_image_displayer")
        self.perspective_image_layout = QtWidgets.QVBoxLayout(
            self.perspective_image_displayer
        )
        self.perspective_image_layout.addWidget(self.perspective_fig_canvas)
        self.perspective_image_layout.setObjectName("perspective_image_layout")
        self.display_image_layout.addWidget(self.perspective_image_displayer)
        self.window_layout.addWidget(self.widget)
        self.ui_buttons = QtWidgets.QFrame(window)
        self.ui_buttons.setMaximumSize(QtCore.QSize(16777215, 50))
        self.ui_buttons.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ui_buttons.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui_buttons.setObjectName("ui_buttons")
        self.button_layout = QtWidgets.QHBoxLayout(self.ui_buttons)
        self.button_layout.setObjectName("button_layout")
        self.ob_left_button = QtWidgets.QPushButton(
            self.ui_buttons, clicked=self._active_ob_left
        )
        self.ob_left_button.setObjectName("ob_left_button")
        self.button_layout.addWidget(self.ob_left_button)
        self.ob_center_button = QtWidgets.QPushButton(
            self.ui_buttons, clicked=self._active_ob_center
        )
        self.ob_center_button.setObjectName("ob_center_button")
        self.button_layout.addWidget(self.ob_center_button)
        self.ob_right_button = QtWidgets.QPushButton(
            self.ui_buttons, clicked=self._active_ob_right
        )
        self.ob_right_button.setObjectName("ob_right_button")
        self.button_layout.addWidget(self.ob_right_button)
        self.rsp_base_button = QtWidgets.QPushButton(
            self.ui_buttons, clicked=self._active_rsp_base
        )
        self.rsp_base_button.setObjectName("rsp_base_button")
        self.button_layout.addWidget(self.rsp_base_button)
        self.mask_or_unmask_button = QtWidgets.QPushButton(
            self.ui_buttons, clicked=self._change_mask_state
        )
        self.mask_or_unmask_button.setObjectName("mask_or_unmask_button")
        self.button_layout.addWidget(self.mask_or_unmask_button)
        self.window_layout.addWidget(self.ui_buttons)

        self.retranslate_ui(window)
        QtCore.QMetaObject.connectSlotsByName(window)

    def _active_ob_left(self):
        self.active_strucutre = "ob_left"

    def _active_ob_center(self):
        self.active_strucutre = "ob_center"

    def _active_ob_right(self):
        self.active_strucutre = "ob_right"

    def _active_rsp_base(self):
        self.active_strucutre = "rsp_base"

    def _change_mask_state(self):
        self.active_mask = not self.active_mask
        self._refresh_perspective_fig()

    def retranslate_ui(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("window", "Locate Anchor"))
        self.ob_left_button.setText(_translate("window", "OB Left"))
        self.ob_center_button.setText(_translate("window", "OB Center"))
        self.ob_right_button.setText(_translate("window", "OB Right"))
        self.rsp_base_button.setText(_translate("window", "RSP Base"))
        self.mask_or_unmask_button.setText(_translate("window", "Mask/Unmask"))


class AnalysisDatabase:
    def __init__(
        self,
        path: str,
        sampling_rate: float,
        baseline_duration: float,
        stimulus_path: str,
        supplement_stimulus: list[tuple[int, int]],
    ) -> None:
        self.path = utils.Path(path)
        self.frames = utils.Frames(self.path.data_path)
        self.dorsal_map = DorsalMap()
        self.dorsal_map.resize(width=self.frames.width, height=self.frames.height)
        self.save_sampling_rate(sampling_rate)
        self.baseline_duration = baseline_duration
        self.stimulus = utils.StimulusInfo()
        if stimulus_path is not None:
            self.stimulus.load(stimulus_path, self.frametimes)
        for start, end in supplement_stimulus:
            self.stimulus.append(start * self.sampling_rate, end * self.sampling_rate)

    def save_sampling_rate(self, sampling_rate: float) -> None:
        with open(self.path.generate_dir.log_dir.sampling_rate_path, "w") as f:
            f.write(str(sampling_rate))

    @property
    def frametimes(self) -> utils.FrameTimes:
        return utils.FrameTimes(self.path.frametimes_path)

    @property
    def sampling_rate(self) -> float:
        with open(self.path.generate_dir.log_dir.sampling_rate_path, "r") as f:
            sampling_rate = float(f.read())
        return sampling_rate

    @property
    def need_preprocessing(self) -> bool:
        return not self.path.generate_dir.preprocessing_dir.have_been_preprocessed

    @property
    def corrected_frames(self) -> np.ndarray:
        corrected_frames = utils.CorrectedFrames()
        corrected_frames.load_from_path(
            self.path.generate_dir.preprocessing_dir.corrected_frames_path
        )
        return corrected_frames

    @property
    def anchor(self) -> dict[str, int]:
        anchor = utils.Anchor()
        anchor.load(self.path.generate_dir.log_dir.anchor_path)
        return anchor

    @property
    def transformer(self) -> utils.PerspectiveTransformer:
        point_names = ["ob_left", "ob_center", "ob_right", "rsp_base"]
        return utils.PerspectiveTransformer(
            from_points=[getattr(self.anchor, point) for point in point_names],
            to_points=[getattr(self.dorsal_map, point) for point in point_names],
        )


def locate_anchor(analysis_database: AnalysisDatabase) -> None:
    if os.path.exists(analysis_database.path.generate_dir.log_dir.anchor_path):
        print("Anchor already exists.")
    else:
        _ = AnchorLocater(
            frames=analysis_database.frames,
            dorsal_map=analysis_database.dorsal_map,
            path=analysis_database.path,
        )


def get_corrected_frames(analysis_database: AnalysisDatabase) -> None:
    if os.path.exists(
        analysis_database.path.generate_dir.preprocessing_dir.corrected_frames_path
    ):
        print("Corrected frames already exists.")
    else:
        low_pass_filter = utils.LowPassFilter()
        keep_factor = 0.2
        shape = analysis_database.frames.frame_shape_with_channel
        num_frames = analysis_database.frames.total_frames
        num_channel = analysis_database.frames.n_channels
        num_width = analysis_database.frames.width
        num_height = analysis_database.frames.height
        corrected_frames = np.zeros((num_frames * num_channel, *shape))

        filtered_frames = np.zeros((num_frames, *shape))
        for idx in tqdm.tqdm(
            list(range(np.prod(shape))), desc="Filtering", unit="pixels"
        ):
            c_num = idx // (num_width * num_height)
            h_num = (idx - c_num * num_width * num_height) // num_width
            w_num = idx - c_num * num_width * num_height - h_num * num_width
            filtered_pixel = low_pass_filter(
                analysis_database.frames.data[:, c_num, h_num, w_num]
            )
            filtered_frames[:, c_num, h_num, w_num] = filtered_pixel

        svd_frames = np.zeros_like(filtered_frames)
        for idx in tqdm.tqdm(
            list(range(len(filtered_frames))), desc="SVD", unit="frames"
        ):
            frames_reshaped = filtered_frames[idx].reshape(-1, shape[-1])
            u, sigma, vt = np.linalg.svd(frames_reshaped, full_matrices=False)
            sigma[int(keep_factor * len(sigma)) :] = 0
            frames_reshaped = u @ np.diag(sigma) @ vt
            svd_frames[idx] = frames_reshaped.reshape(shape)

        t = np.arange(num_frames * num_channel)
        t_0 = t[0::2]
        t_1 = t[1::2]
        for idx in tqdm.tqdm(
            list(range(np.prod(shape))), desc="Interpolation", unit="pixels"
        ):
            c_num = idx // (num_width * num_height)
            h_num = (idx - c_num * num_width * num_height) // num_width
            w_num = idx - c_num * num_width * num_height - h_num * num_width
            if c_num == 0:
                corrected_frames[:, c_num, h_num, w_num][0::2] = svd_frames[
                    :, c_num, h_num, w_num
                ]
                corrected_frames[:, c_num, h_num, w_num][1::2] = interp1d(
                    t_0,
                    svd_frames[:, c_num, h_num, w_num],
                    axis=0,
                    fill_value="extrapolate",
                )(t_1)
            elif c_num == 1:
                corrected_frames[:, c_num, h_num, w_num][1::2] = svd_frames[
                    :, c_num, h_num, w_num
                ]
                corrected_frames[:, c_num, h_num, w_num][0::2] = interp1d(
                    t_1,
                    svd_frames[:, c_num, h_num, w_num],
                    axis=0,
                    fill_value="extrapolate",
                )(t_0)

        corrected_frames = utils.CorrectedFrames(corrected_frames)
        corrected_frames.save(
            analysis_database.path.generate_dir.preprocessing_dir.corrected_frames_path
        )


def preprocessing(analysis_database: AnalysisDatabase) -> None:
    locate_anchor(analysis_database=analysis_database)
    get_corrected_frames(analysis_database=analysis_database)


class ui_corrected_analysis_displayer(object):
    DEFAULT_COLOR_RANGE = 5

    def __init__(
        self, analysis_database: AnalysisDatabase, show_frame: np.ndarray, cmap
    ) -> None:
        self.analysis_database = analysis_database
        self.show_frame = show_frame
        self.cmap = cmap
        self.show_fig = plt.figure()
        self.canvas = FigureCanvas(self.show_fig)
        self._refresh_show_fig(self.DEFAULT_COLOR_RANGE)
        app = QtWidgets.QApplication(sys.argv)
        analysis_ui = QtWidgets.QWidget()
        self.setup_ui(analysis_ui)
        analysis_ui.show()
        sys.exit(app.exec_())

    def setup_ui(self, corrected_analysis_displayer):
        corrected_analysis_displayer.setObjectName("corrected_analysis_displayer")
        corrected_analysis_displayer.resize(640, 480)
        self.ui_basic_layout = QtWidgets.QVBoxLayout(corrected_analysis_displayer)
        self.ui_basic_layout.setObjectName("ui_basic_layout")
        self.result_displayer = QtWidgets.QFrame(corrected_analysis_displayer)
        self.result_displayer.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.result_displayer.setFrameShadow(QtWidgets.QFrame.Raised)
        self.result_displayer.setObjectName("result_displayer")
        self.show_fig_layout = QtWidgets.QVBoxLayout(self.result_displayer)
        self.show_fig_layout.addWidget(self.canvas)
        self.ui_basic_layout.addWidget(self.result_displayer)
        self.setting_widget = QtWidgets.QFrame(corrected_analysis_displayer)
        self.setting_widget.setMaximumSize(QtCore.QSize(16777215, 60))
        self.setting_widget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setting_widget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setting_widget.setObjectName("setting_widget")
        self.setting_layout = QtWidgets.QHBoxLayout(self.setting_widget)
        self.setting_layout.setObjectName("setting_layout")
        self.color_range = QtWidgets.QLineEdit(self.setting_widget)
        self.color_range.setText(f"{self.DEFAULT_COLOR_RANGE}")
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        # size_policy.setHeightForWidth(
        #     self.color_range.size_policy().hasHeightForWidth()
        # )
        self.color_range.setSizePolicy(size_policy)
        self.color_range.setMaximumSize(QtCore.QSize(200, 16777215))
        self.color_range.setObjectName("color_range")
        self.setting_layout.addWidget(self.color_range)
        self.percentage_label = QtWidgets.QLabel(self.setting_widget)
        self.percentage_label.setMaximumSize(QtCore.QSize(20, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.percentage_label.setFont(font)
        self.percentage_label.setObjectName("percentage_label")
        self.setting_layout.addWidget(self.percentage_label)
        self.confirm_button = QtWidgets.QPushButton(
            self.setting_widget, clicked=self.confirm_setting_params
        )
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        # size_policy.setHeightForWidth(
        #     self.confirm_button.size_policy().hasHeightForWidth()
        # )
        self.confirm_button.setSizePolicy(size_policy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.confirm_button.setFont(font)
        self.confirm_button.setObjectName("confirm_button")
        self.setting_layout.addWidget(self.confirm_button)
        self.save_button = QtWidgets.QPushButton(self.setting_widget, clicked=self.save)
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        # size_policy.setHeightForWidth(
        #     self.save_button.size_policy().hasHeightForWidth()
        # )
        self.save_button.setSizePolicy(size_policy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.setting_layout.addWidget(self.save_button)
        self.ui_basic_layout.addWidget(self.setting_widget)

        self.retranslate_ui(corrected_analysis_displayer)
        QtCore.QMetaObject.connectSlotsByName(corrected_analysis_displayer)

    def confirm_setting_params(self):
        color_range = float(self.color_range.text())
        self._refresh_show_fig(color_range)

    def _refresh_show_fig(self, color_range: float) -> None:
        self.show_fig.clear()
        ax = self.show_fig.add_subplot(111)
        img = ax.imshow(
            self.show_frame,
            cmap=self.cmap,
            vmin=-color_range,
            vmax=color_range,
        )
        self.analysis_database.dorsal_map.mask_axes(ax)
        cbar = self.show_fig.colorbar(img)
        cbar.set_label("%")
        cbar.set_ticks([-color_range, 0, color_range])
        self.canvas.draw()

    def save(self):
        self.show_fig.savefig(
            self.analysis_database.path.generate_dir.analysis_dir.image_dir.image_path,
            dpi=300,
        )
        self.analysis_database.stimulus.save(
            self.analysis_database.path.generate_dir.analysis_dir.image_dir.stimulus_path
        )
        with open(
            self.analysis_database.path.generate_dir.analysis_dir.image_dir.baseline_path,
            "w",
        ) as f:
            f.write(f"{str(self.analysis_database.baseline_duration)} s")
        with open(
            self.analysis_database.path.generate_dir.analysis_dir.image_dir.color_range_path,
            "w",
        ) as f:
            f.write(f"{self.color_range.text()} %")

    def retranslate_ui(self, corrected_analysis_displayer):
        _translate = QtCore.QCoreApplication.translate
        corrected_analysis_displayer.setWindowTitle(
            _translate("corrected_analysis_displayer", "Analysis")
        )
        self.percentage_label.setText(_translate("corrected_analysis_displayer", "%"))
        self.confirm_button.setText(
            _translate("corrected_analysis_displayer", "Confirm")
        )
        self.save_button.setText(_translate("corrected_analysis_displayer", "Save"))


def analyze(analysis_database: AnalysisDatabase) -> None:
    corrected_frames = analysis_database.corrected_frames
    baseline_idx = analysis_database.stimulus.get_baseline_idx(
        duration=analysis_database.baseline_duration,
        sampling_rate=analysis_database.sampling_rate,
    )
    baseline_frames = []
    for start, end in baseline_idx:
        baseline_frames.append(np.nanmean(corrected_frames.data[start:end], axis=0))
    stimulus_frames = []
    for start, end in analysis_database.stimulus.frames_idx:
        stimulus_frames.append(np.nanmean(corrected_frames.data[start:end], axis=0))
    show_frames = (np.array(stimulus_frames) - np.array(baseline_frames)) / np.array(
        baseline_frames
    )
    show_frame = np.nanmean(show_frames, axis=0)
    show_frame = analysis_database.transformer.perspective_transform(show_frame)
    show_frame[analysis_database.dorsal_map.mask_array == 0] = np.nan
    show_frame *= 100
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["c", "blue", "black", "red", "orange"]
    )
    cmap.set_bad(color="black")

    _ = ui_corrected_analysis_displayer(
        analysis_database=analysis_database, show_frame=show_frame, cmap=cmap
    )


def main(
    path: str,
    sampling_rate: float,
    baseline_duration: float,
    stimulus_path: str,
    supplement_stimulus: list[tuple[int, int]],
) -> None:
    analysis_db = AnalysisDatabase(
        path=path,
        sampling_rate=sampling_rate,
        baseline_duration=baseline_duration,
        stimulus_path=stimulus_path,
        supplement_stimulus=supplement_stimulus,
    )
    if analysis_db.need_preprocessing:
        preprocessing(analysis_db)
    else:
        print("Already preprocessed.")
    analyze(analysis_db)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    with open(f"{os.path.join(dirname, 'config.yaml')}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(**config)

import os
from typing import Optional

__all__ = ["Path"]


class ImageMixin:
    _dirname: str
    _num: int
    _filename_lst: list[str]
    _COLOR_RANGE = "Color Range.txt"
    _STIMULUS = "Stimulus.csv"
    _BASELINE = "Baseline.csv"

    @property
    def _num(self) -> int:
        _num = len(os.listdir(self._dirname))
        if _num == 0:
            return 0
        else:
            _num -= 1
            get_sub_path = lambda filename: os.path.join(
                self._dirname, f"{_num}", filename
            )
            for filename in self._filename_lst:
                if not os.path.exists(get_sub_path(filename)):
                    return _num
            return _num + 1

    @property
    def _num_dirname(self) -> str:
        _num_dirname = os.path.join(self._dirname, f"{self._num}")
        os.makedirs(_num_dirname, exist_ok=True)
        return _num_dirname

    def _get_sub_path(self, filename: str) -> str:
        return os.path.join(self._num_dirname, filename)

    @property
    def color_range_path(self) -> str:
        return self._get_sub_path(self._COLOR_RANGE)

    @property
    def stimulus_path(self) -> str:
        return self._get_sub_path(self._STIMULUS)

    @property
    def baseline_path(self) -> str:
        return self._get_sub_path(self._BASELINE)


class ROIDirectory(ImageMixin):
    _ROI_VARIATION = "ROI Variation.csv"
    _ROI_BOUNDING_POINTS = "ROI Bounding Points.csv"
    _ROI_IMAGE = "ROI.png"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    @property
    def _filename_lst(self) -> list[str]:
        return [
            self._ROI_VARIATION,
            self._ROI_BOUNDING_POINTS,
            self._ROI_IMAGE,
            self._COLOR_RANGE,
            self._STIMULUS,
            self._BASELINE,
        ]

    @property
    def roi_variation_path(self) -> str:
        return self._get_sub_path(self._ROI_VARIATION)

    @property
    def roi_bounding_points_path(self) -> str:
        return self._get_sub_path(self._ROI_BOUNDING_POINTS)

    @property
    def roi_image_path(self) -> str:
        return self._get_sub_path(self._ROI_IMAGE)


class ImageDirectory(ImageMixin):
    IMAGE = "Image.png"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    @property
    def _filename_lst(self) -> list[str]:
        return [self.IMAGE, self._COLOR_RANGE, self._STIMULUS, self._BASELINE]

    @property
    def image_path(self) -> str:
        return self._get_sub_path(self.IMAGE)


class AnalysisDirectory:
    roi_dir: ROIDirectory
    image_dir: ImageDirectory
    _STRUCTURE_VARIATION = "Structure Variation.csv"
    _ROI_DIRNAME = "ROI"
    _IMAGE_DIRNAME = "Image"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)
        self.roi_dir = ROIDirectory(os.path.join(dirname, self._ROI_DIRNAME))
        self.image_dir = ImageDirectory(os.path.join(dirname, self._IMAGE_DIRNAME))

    @property
    def structure_variation_path(self) -> str:
        return os.path.join(self._dirname, self._STRUCTURE_VARIATION)


class PreprocessingDirectory:
    _CORRECTED_FRAMES = "Corrected Frames.npy"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    @property
    def corrected_frames_path(self) -> str:
        return os.path.join(self._dirname, self._CORRECTED_FRAMES)

    @property
    def have_been_preprocessed(self) -> bool:
        return os.path.exists(self.corrected_frames_path)


class LogDirectory:
    _SAMPLING_RATE = "Sampling Rate.txt"
    _ANCHOR = "Anchor.yaml"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    @property
    def sampling_rate_path(self) -> str:
        return os.path.join(self._dirname, self._SAMPLING_RATE)

    @property
    def anchor_path(self) -> str:
        return os.path.join(self._dirname, self._ANCHOR)


class GenerateDirectory:
    log_dir: LogDirectory
    preprocessing_dir: PreprocessingDirectory
    analysis_dir: AnalysisDirectory
    _LOG_DIRNAME = "Log"
    _PREPROCESSING_DIRNAME = "Preprocessing"
    _ANALYSIS_DIRNAME = "Analysis"

    def __init__(self, dirname: str) -> None:
        self._dirname = dirname
        os.makedirs(dirname, exist_ok=True)
        self.log_dir = LogDirectory(os.path.join(dirname, self._LOG_DIRNAME))
        self.preprocessing_dir = PreprocessingDirectory(
            os.path.join(dirname, self._PREPROCESSING_DIRNAME)
        )
        self.analysis_dir = AnalysisDirectory(
            os.path.join(dirname, self._ANALYSIS_DIRNAME)
        )


class Path:
    _NaN = "NaN"
    _FRAMETIMES = "frameTimes.mat"

    def _get_file_number(self, path: str) -> Optional[int]:
        _no_ext = os.path.splitext(path)[0]
        _file_number = _no_ext.rsplit("_", 1)[-1]
        return int(_file_number) if _file_number.isdigit() else None

    def __init__(self, path: str) -> None:
        self._path = path
        self._dirname = os.path.dirname(path)
        _file_number = self._get_file_number(path)
        join_dirname = lambda filename: os.path.join(self._dirname, filename)
        if _file_number is None:
            _generate_dirname = join_dirname(self._NaN)
            self.frametimes_path = join_dirname(self._FRAMETIMES)
        else:
            _generate_dirname = join_dirname(f"{_file_number:04}")
            _filename, _ext = os.path.splitext(self._FRAMETIMES)
            _frametimes_filename = f"{_filename}_{_file_number:04}{_ext}"
            self.frametimes_path = join_dirname(_frametimes_filename)
        self.generate_dir = GenerateDirectory(_generate_dirname)

    @property
    def data_path(self) -> str:
        return self._path

import datetime
import os
import re
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.io as sio
from skvideo.io import FFmpegReader
from tifffile import imread

__all__ = [
    "BaselineInfo",
    "CorrectedFrames",
    "Frames",
    "FrameTimes",
    "StimulusInfo",
]


class FramesInfo:
    def __init__(self, path: str) -> None:
        self.path = path
        _info = os.path.splitext(os.path.dirname(path))[0].split("_")
        self.file_number = int(_info.pop(-1)) if _info[-1].isdigit() else None
        assert len(_info) >= 4
        self._info = _info

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1]

    @property
    def n_channels(self) -> int:
        return int(self._info[-4])

    @property
    def height(self) -> int:
        return int(self._info[-3])

    @property
    def width(self) -> int:
        return int(self._info[-2])

    @property
    def dtype(self) -> str:
        return self._info[-1]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def frame_shape_with_channel(self) -> tuple[int, int, int]:
        return (self.n_channels, self.height, self.width)

    @property
    def figsize(self) -> tuple[int, int]:
        return (self.width, self.height)


class Frames(FramesInfo):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        match self.ext:
            case ".dat":
                self.data = self._load_dat()
            case ".tif":
                self.data = self._load_tif()
            case ".mj2":
                self.data = self._load_mj2()
            case _:
                raise ValueError(f"Unsupported file type: {self.ext}")
        self.average = np.mean(self.data, axis=0)

    def _load_dat(self) -> np.ndarray:
        file_size = os.path.getsize(self.path)
        n_frames_with_channel = file_size // (
            np.dtype(self.dtype).itemsize * np.prod(self.frame_shape_with_channel)
        )
        return np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            shape=(n_frames_with_channel, *self.frame_shape_with_channel),
        )

    def _load_tif(self) -> np.ndarray:
        data = imread(self.path)
        data = np.transpose(data, (0, 2, 1))
        if (len(self.data) % 2 == 1) and (len(self.frame_shape_with_channel) == 3):
            data = data[:-1]
        return np.reshape(data, (-1, *self.frame_shape_with_channel))

    def _load_mj2(self) -> np.ndarray:
        # have not been tested.
        sq = FFmpegReader(self.path, outputdict={"-pix_fmt": "gray16le"})
        data = np.stack(list(sq)).squeeze()
        sq.close()
        return np.reshape(data, (-1, *self.frame_shape_with_channel))

    @property
    def total_frames(self) -> int:
        return len(self.data)


class CorrectedFrames:
    def __init__(self, array: Optional[np.ndarray] = None) -> None:
        self._data = array

    def load_from_path(self, path: str) -> None:
        self._data = np.load(path)

    def load_from_array(self, array: np.ndarray) -> None:
        self._data = array

    @property
    def data(self) -> np.ndarray:
        return self._data[:, 0, ...] - self._data[:, 1, ...]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self._data)

    @property
    def total_frames(self) -> int:
        return len(self._data)


class FrameTimes:
    frame_times: list[datetime.datetime]
    _KEY = "frameTimes"

    def __init__(self, path: str) -> None:
        frame_times = sio.loadmat(path)[self._KEY].squeeze().tolist()
        self.frame_times = [self._trans_time(time) for time in frame_times]

    @staticmethod
    def _trans_time(matlab_time: float) -> datetime.datetime:
        return (
            datetime.datetime.fromordinal(int(matlab_time))
            + datetime.timedelta(days=matlab_time % 1)
            - datetime.timedelta(days=366)
        )


class UltraSoundStimLog:
    def __init__(self, path: str) -> None:
        self._stim_data = self._get_stim_data(path)

    def _get_stim_data(
        self, path: str
    ) -> list[list[datetime.datetime, str, datetime.datetime]]:
        with open(path, "r") as f:
            content = f.read()

        pattern = r"'begin:(.*?)', '(.*?)', 'end:(.*?)'"
        matches = re.findall(pattern, content)
        param_pattern = r"(\w+):(\w+)"

        stim_data = []
        for match in matches:
            start_time = datetime.datetime.strptime(match[0], "%Y-%m-%d %H:%M:%S.%f")
            params = re.findall(param_pattern, match[1])
            params = {key: value for key, value in params}
            end_time = datetime.datetime.strptime(match[2], "%Y-%m-%d %H:%M:%S.%f")
            stim_data.append([start_time, params, end_time])

        return stim_data

    def get_stim_frames_lst(self, frame_times: FrameTimes) -> list[list[int, int]]:
        stim_frames_lst = []
        binary_search_idx = lambda stim_time: self._binary_search(
            frame_times.frame_times, stim_time
        )
        for stim in self._stim_data:
            start_idx = binary_search_idx(stim[0])
            end_idx = binary_search_idx(stim[2])
            stim_frames_lst.append([start_idx, end_idx])
        return stim_frames_lst

    @staticmethod
    def _binary_search(arr: list[datetime.datetime], target: datetime.datetime) -> int:
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low


class StimulusInfo:
    START_COLNAME = "Start"
    END_COLNAME = "End"

    def __init__(self) -> None:
        self._stim_log = []

    def append(self, start: int, end: int) -> None:
        self._stim_log.append([start, end])
        self._stim_log = list(set(self._stim_log))
        self._stim_log.sort()

    def remove(self, idx: int) -> None:
        self._stim_log.pop(idx)

    def clear(self) -> None:
        self._stim_log = []

    def __len__(self) -> int:
        return len(self._stim_log)

    def load(self, path: str, frame_times: Optional[FrameTimes] = None) -> None:
        if os.path.basename(path).endswith(".txt.txt"):
            if frame_times is None:
                raise ValueError("frameTimes must be provided.")
            _stim_log = UltraSoundStimLog(path).get_stim_frames_lst(frame_times)
            self._stim_log += _stim_log
            self._stim_log = list(set(self._stim_log))
        else:
            _stim_log = pd.read_csv(path)
            for _, row in _stim_log.iterrows():
                self._stim_log.append(row[self.START_COLNAME], row[self.END_COLNAME])
        self._stim_log.sort()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        _stim_log = np.array(self._stim_log)
        _stim_log = pd.DataFrame(
            _stim_log, columns=[self.START_COLNAME, self.END_COLNAME]
        )
        _stim_log.to_csv(path, index=False)

    @property
    def frames_idx(self) -> list[int]:
        return [idx for start, end in self._stim_log for idx in range(start, end)]

    @property
    def content(self) -> list[str]:
        return [f"Start: {start}; End: {end}" for start, end in self._stim_log]


class BaselineInfo:
    mode: Union[Literal["Frame Mode", "PreStim Mode"], None]
    start: Union[int, None]
    end: Union[int, None]
    pre_sec: Union[float, None]

    def __init__(self) -> None:
        self.mode = None
        self.start = None
        self.end = None
        self.pre_sec = None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.mode + "\n")
            match self.mode:
                case "Frame Mode":
                    f.write(f"Start: {self.start}\n")
                    f.write(f"End: {self.end}\n")
                case "PreStim Mode":
                    f.write(f"Seconds: {self.pre_sec}\n")

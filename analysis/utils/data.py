import datetime
import os
from tkinter import filedialog
from typing import *

import numpy as np
import pandas as pd
import scipy.io as sio
from skvideo.io import FFmpegReader
from tifffile import imread


def get_data_file() -> str:
    """Get the path of the data file.

    Returns:
        The path of the data file.
    """
    return filedialog.askdirectory(title='Select a data file')


class FramesInfo:
    """The information of the frames.
    
    Attributes:
        path: The path of the frames.
        filename: The filename of the frames.
        file_number: The file number of the frames.
        info: The information of the frames.
    """
    def __init__(self, path: str) -> None:
        self.path = path
        self.filename = os.path.basename(path)
        info = self.no_ext.split("_")
        self.file_number = int(info.pop(-1)) if info[-1].isdigit() else None
        assert len(info) >= 4
        self.info = info

    @property
    def no_ext(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def extention(self) -> str:
        return os.path.splitext(self.filename)[1]

    @property
    def n_channels(self) -> int:
        return int(self.info[-4])

    @property
    def height(self) -> int:
        return int(self.info[-3])

    @property
    def width(self) -> int:
        return int(self.info[-2])

    @property
    def dtype(self) -> str:
        return self.info[-1]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def frame_shape_with_channel(self) -> tuple[int, int, int]:
        return (self.n_channels, self.height, self.width)

    @property
    def figsize(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def aspec_ratio(self) -> float:
        return self.width / self.height


class Frames(FramesInfo):
    def __init__(self, path: str) -> None:
        try:
            super().__init__(path)
        except AssertionError:
            print(f"Warning: Invalid filename: {path}")
        match self.extention:
            case ".dat":
                self.data = self._load_dat()
            case ".tif":
                self.data = self._load_tif()
            case ".mj2":
                self.data = self._load_mj2()
            case _:
                print(f"Warning: Invalid extention: {self.extention}")
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

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class CorrectedFrames:
    FILENAME = "corrected_frames.npy"

    def __init__(self, dirname: str) -> None:
        self._data = np.load(os.path.join(dirname, self.FILENAME))

    @property
    def data(self) -> np.ndarray:
        return self._data[:, 0, ...] - self._data[:, 1, ...]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]

    @classmethod
    def from_array(cls, data: np.ndarray) -> "CorrectedFrames":
        """Load the corrected frames.

        Args:
            data: The corrected frames.

        Returns:
            The corrected frames.
        """
        corrected_frames = cls()
        corrected_frames._data = data
        return corrected_frames

    def save(self, dirname: str) -> None:
        """Save the corrected frames.

        Args:
            path: The path to save the corrected frames.
        """
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, self.FILENAME), self._data)


class FrameTimes:
    KEY = 'frameTimes'

    def __init__(self, path: str) -> None:
        frame_times = sio.loadmat(path)[self.KEY].squeeze().tolist()
        self.frame_times = [self._trans_time(time) for time in frame_times]
    
    @staticmethod
    def _trans_time(matlab_time: float) -> datetime.datetime:
        return (datetime.datetime.fromordinal(int(matlab_time)) 
                + datetime.timedelta(days=matlab_time % 1) 
                - datetime.timedelta(days = 366))


class UltraSoundStimLog:
    COLS = ['Begin', 'Params', 'End']
    def __init__(self, path: str) -> None:
        self.path = path
        self.stim_data = self._get_stim_data()

    def _get_stim_data(self) -> list[list[datetime.datetime, str, datetime.datetime]]:
        with open(self.path, 'r') as f:
            cont = f.readline()
        cont = cont.strip('[]\n').split(',')
        first_cont = cont[0].strip("'")
        sign = first_cont[-6]
        cont = [elem.strip("'") for elem in cont]
        df_cont = []
        for i, elem in enumerate(cont):
            if i % 3 == 1:
                df_cont.append(elem)
            else:
                elem_preprocessed = elem.rsplit(sign, 1)[0]
                if i % 3 == 0:
                    elem_preprocessed = elem_preprocessed.lstrip('begin:')
                elif i % 3 == 2:
                    elem_preprocessed = elem_preprocessed.lstrip('end:')
                df_cont.append(datetime.datetime.strptime(elem_preprocessed, '%Y-%m-%d %H:%M:%S.%f'))
        stim_len = len(self.COLS)
        stims = [cont[i:i+stim_len] for i in range(0, len(cont), stim_len)]
        return stims

    def get_stim_lst(self, frame_times: FrameTimes) -> list[list[int, int]]:
        stim_frames_lst = []
        for stim in self.stim_data:
            stim_frames = []
            for i, frame_time in enumerate(frame_times.frame_times):
                if stim[0] <= frame_time <= stim[2]:
                    stim_frames.append(i)
                elif frame_time > stim[2]:
                    break
            stim_frames_lst.append([stim_frames_lst[0], stim_frames[-1]])
        return stim_frames_lst

class StimLog:
    def __init__(self, stim_frames_lst: list[list[int, int]]) -> None:
        self.init_stim_log = stim_frames_lst.copy()
        self.stim_log = stim_frames_lst.copy()
        self.stim_log.sort()
    
    def append(self, start: int, end: int) -> None:
        self.stim_log.append([start, end])
        self.stim_log.sort()
    
    def remove(self, idx: int) -> None:
        self.stim_log.pop(idx)
    
    def clear(self) -> None:
        self.stim_log = self.init_stim_log.copy()
        self.stim_log.sort()
    
    def reset(self) -> None:
        self.stim_log = self.init_stim_log.copy()
        self.stim_log.sort()
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        stim_log = np.array(self.stim_log)
        stim_log = pd.DataFrame(stim_log, columns=['start', 'end'])
        stim_log.to_csv(path, index=False)
        
    
    @classmethod
    def load_from_file(cls, path: str) -> "StimLog":
        stim_log = pd.read_csv(path).values
        return cls(stim_log.tolist())
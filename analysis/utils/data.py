import os
from typing import *

import numpy as np
from skvideo.io import FFmpegReader
from tifffile import imread


class FramesMixin:
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


class Frames(FramesMixin):
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

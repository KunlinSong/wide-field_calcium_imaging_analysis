from typing import *

import numpy as np
from scipy import signal

__all__ = ["LowPassFilter"]


class LowPassFilter:
    """Low pass filter. The filter is applied along the first axis. The
    low pass frequency is the Nyquist frequency of the sampling rate.

    Attributes:
        _b: The numerator coefficient vector of the filter.
        _a: The denominator coefficient vector of the filter.
        EPS: A small number to avoid numerical error.
    """

    EPS = 1e-6

    def __init__(self) -> None:
        """Initialize the filter."""
        self._b, self._a = signal.butter(
            N=4, Wn=(1 - self.EPS), btype="lowpass", output="ba"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter the input signal.

        Args:
            x: The input signal.
        """
        return signal.filtfilt(b=self._b, a=self._a, x=x, axis=0, method="gust")

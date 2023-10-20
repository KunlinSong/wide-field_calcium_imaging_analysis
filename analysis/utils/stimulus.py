import os

import pandas as pd


class Stimulus:
    FILENAME = "stimulus.csv"

    def __init__(self) -> None:
        self.stim = {"Start": [], "End": []}

    def add(self, start: int, end: int) -> None:
        self.stim["Start"].append(start)
        self.stim["End"].append(end)

    def remove(self, idx: int) -> None:
        self.stim["Start"].pop(idx)
        self.stim["End"].pop(idx)

    @classmethod
    def from_log(cls, dirname: str) -> "Stimulus":
        """Load the stimulus.

        Args:
            path: The path to load the stimulus.

        Returns:
            The stimulus.
        """
        stim = cls()
        df = pd.read_csv(os.path.join(dirname, cls.FILENAME))
        for start, end in zip(df["Start"], df["End"]):
            stim.add(start, end)
        return stim

    @property
    def frames_idx(self) -> list[int]:
        frames_idx = []
        for start, end in zip(self.stim["Start"], self.stim["End"]):
            frames_idx += list(range(start, end))
        return frames_idx

    def __len__(self) -> int:
        return len(self.stim["Start"])

    def save(self, dirname: str) -> None:
        """Save the stimulus.

        Args:
            path: The path to save the stimulus.
        """
        df = pd.DataFrame(self.stim)
        os.makedirs(dirname, exist_ok=True)
        df.to_csv(os.path.join(dirname, self.FILENAME), index=False)

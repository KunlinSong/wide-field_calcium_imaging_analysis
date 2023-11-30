import os

import yaml

__all__ = ["Anchor"]


class Anchor:
    """Anchor points.

    Attributes:
        ob_left: The OB left point.
        ob_center: The OB center point.
        ob_right: The OB right point.
        rsp_base: The RSP base point.
        figsize: The figure size.
    """

    OB_LEFT = "OB Left"
    OB_CENTER = "OB Center"
    OB_RIGHT = "OB Right"
    RSP_BASE = "RSP Base"
    FIG_SIZE = "figsize"

    def __init__(
        self,
        ob_left: tuple[int, int] = (0, 0),
        ob_center: tuple[int, int] = (0, 0),
        ob_right: tuple[int, int] = (0, 0),
        rsp_base: tuple[int, int] = (0, 0),
        figsize: tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize the anchor points.

        Args:
            ob_left (tuple[int, int]): The OB left point.
            ob_center (tuple[int, int]): The OB center point.
            ob_right (tuple[int, int]): The OB right point.
            rsp_base (tuple[int, int]): The RSP base point.
            figsize (tuple[int, int]): The figure size.
        """
        self.ob_left = ob_left
        self.ob_center = ob_center
        self.ob_right = ob_right
        self.rsp_base = rsp_base
        self.figsize = figsize

    @property
    def content(self) -> dict[str, dict[str, tuple[int, int]]]:
        """Get the content.

        Returns:
            The content of the anchor points.
        """
        return {
            self.OB_LEFT: {"x": self.ob_left[0], "y": self.ob_left[1]},
            self.OB_CENTER: {"x": self.ob_center[0], "y": self.ob_center[1]},
            self.OB_RIGHT: {"x": self.ob_right[0], "y": self.ob_right[1]},
            self.RSP_BASE: {"x": self.rsp_base[0], "y": self.rsp_base[1]},
            self.FIG_SIZE: {"width": self.figsize[0], "height": self.figsize[1]},
        }

    def save(self, path: str) -> None:
        """Save the anchor points.

        Args:
            path: The path to save the anchor points.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.content, f)

    @staticmethod
    def _load(path: str) -> dict[str, dict[str, int]]:
        with open(path, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        return content

    @classmethod
    def load(cls, path: str) -> "Anchor":
        """Load the anchor points.

        Args:
            path: The path to load the anchor points.

        Returns:
            The anchor points.
        """
        content = cls._load(path)
        return cls(
            ob_left=(content[cls.OB_LEFT]["x"], content[cls.OB_LEFT]["y"]),
            ob_center=(content[cls.OB_CENTER]["x"], content[cls.OB_CENTER]["y"]),
            ob_right=(content[cls.OB_RIGHT]["x"], content[cls.OB_RIGHT]["y"]),
            rsp_base=(content[cls.RSP_BASE]["x"], content[cls.RSP_BASE]["y"]),
            figsize=(content[cls.FIG_SIZE]["width"], content[cls.FIG_SIZE]["height"]),
        )

    @property
    def aspect_ratio(self) -> float:
        """Get the aspect ratio.

        Returns:
            The aspect ratio.
        """
        return self.figsize[0] / self.figsize[1]

    def resize(self, figsize: tuple[int, int]):
        if figsize == self.figsize:
            return

        aspect_ratio = figsize[0] / figsize[1]
        if aspect_ratio == self.aspect_ratio:
            to_w = figsize[0]
            to_h = figsize[1]
            x_add = 0
            y_add = 0
        elif aspect_ratio > self.aspect_ratio:
            to_w = int(self.figsize[0] * figsize[1] / self.figsize[1])
            to_h = figsize[1]
            x_add = (figsize[0] - to_w) // 2
            y_add = 0
        else:
            to_w = figsize[0]
            to_h = int(self.figsize[1] * figsize[0] / self.figsize[0])
            x_add = 0
            y_add = (figsize[1] - to_h) // 2
        self.ob_left = (
            self.ob_left[0] * to_w / self.figsize[0] + x_add,
            self.ob_left[1] * to_h / self.figsize[1] + y_add,
        )
        self.ob_center = (
            self.ob_center[0] * to_w / self.figsize[0] + x_add,
            self.ob_center[1] * to_h / self.figsize[1] + y_add,
        )
        self.ob_right = (
            self.ob_right[0] * to_w / self.figsize[0] + x_add,
            self.ob_right[1] * to_h / self.figsize[1] + y_add,
        )
        self.rsp_base = (
            self.rsp_base[0] * to_w / self.figsize[0] + x_add,
            self.rsp_base[1] * to_h / self.figsize[1] + y_add,
        )

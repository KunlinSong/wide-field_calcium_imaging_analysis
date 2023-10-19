import os

import yaml


class Anchor:
    FILENAME = "anchor.yaml"
    OB_LEFT = "OB Left"
    OB_CENTER = "OB Center"
    OB_RIGHT = "OB Right"
    RSP_BASE = "RSP Base"

    def __init__(
        self,
        ob_left: tuple[int, int],
        ob_center: tuple[int, int],
        ob_right: tuple[int, int],
        rsp_base: tuple[int, int],
        figsize: tuple[int, int],
    ) -> None:
        self.ob_left = ob_left
        self.ob_center = ob_center
        self.ob_right = ob_right
        self.rsp_base = rsp_base
        self.figsize = figsize

    @property
    def content(self) -> dict[str, dict[str, tuple[int, int]]]:
        return {
            self.OB_LEFT: {"x": self.ob_left[0], "y": self.ob_left[1]},
            self.OB_CENTER: {"x": self.ob_center[0], "y": self.ob_center[1]},
            self.OB_RIGHT: {"x": self.ob_right[0], "y": self.ob_right[1]},
            self.RSP_BASE: {"x": self.rsp_base[0], "y": self.rsp_base[1]},
            "figsize": {"width": self.figsize[0], "height": self.figsize[1]},
        }

    def save(self, dirname: str) -> None:
        """Save the anchor points.

        Args:
            path: The path to save the anchor points.
        """
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, self.FILENAME), "w") as f:
            yaml.dump(self.content, f)

    @classmethod
    def load_by_dirname(cls, dirname: str) -> "Anchor":
        """Load the anchor points.

        Args:
            path: The path to load the anchor points.

        Returns:
            The anchor points.
        """
        with open(os.path.join(dirname, cls.FILENAME), "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        return cls(
            ob_left=(content[cls.OB_LEFT]["x"], content[cls.OB_LEFT]["y"]),
            ob_center=(content[cls.OB_CENTER]["x"], content[cls.OB_CENTER]["y"]),
            ob_right=(content[cls.OB_RIGHT]["x"], content[cls.OB_RIGHT]["y"]),
            rsp_base=(content[cls.RSP_BASE]["x"], content[cls.RSP_BASE]["y"]),
            figsize=(content["figsize"]["width"], content["figsize"]["height"]),
        )

    @classmethod
    def load_from_file(cls, path: str) -> "Anchor":
        """Load the anchor points.

        Args:
            path: The path to load the anchor points.

        Returns:
            The anchor points.
        """
        with open(path, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        return cls(
            ob_left=(content[cls.OB_LEFT]["x"], content[cls.OB_LEFT]["y"]),
            ob_center=(content[cls.OB_CENTER]["x"], content[cls.OB_CENTER]["y"]),
            ob_right=(content[cls.OB_RIGHT]["x"], content[cls.OB_RIGHT]["y"]),
            rsp_base=(content[cls.RSP_BASE]["x"], content[cls.RSP_BASE]["y"]),
            figsize=(content["figsize"]["width"], content["figsize"]["height"]),
        )

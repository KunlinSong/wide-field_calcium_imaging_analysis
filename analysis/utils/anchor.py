import os

import yaml

__all__ = ["Anchor"]


class Anchor:
    _OB_LEFT = "ob_left"
    _OB_CENTER = "ob_center"
    _OB_RIGHT = "ob_right"
    _RSP_BASE = "rsp_base"

    def __init__(
        self,
        ob_left: tuple[int, int] = (0, 0),
        ob_center: tuple[int, int] = (0, 0),
        ob_right: tuple[int, int] = (0, 0),
        rsp_base: tuple[int, int] = (0, 0),
    ) -> None:
        self.ob_left = ob_left
        self.ob_center = ob_center
        self.ob_right = ob_right
        self.rsp_base = rsp_base

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.content, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        self.ob_left = content[self._OB_LEFT]
        self.ob_center = content[self._OB_CENTER]
        self.ob_right = content[self._OB_RIGHT]
        self.rsp_base = content[self._RSP_BASE]

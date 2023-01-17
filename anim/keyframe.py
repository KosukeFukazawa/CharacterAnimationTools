from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from anim.skel import Skel

@dataclass
class KeyFrame:
    frame: int
    joint: str | int
    rotation: np.ndarray = None
    position: np.ndarray = None
    interp: str = "linear" # interpolation method.


class KeyFrameAnimation:
    def __init__(
        self,
        skel: Skel,
        keys: list[KeyFrame],
        fps: int,
        anim_name: str=None,
        positions: np.ndarray=None,
    ) -> None:
        self.skel = skel
        # TBD
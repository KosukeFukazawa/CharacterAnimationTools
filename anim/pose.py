from __future__ import annotations

import numpy as np
from util import quat
from anim.skel import Skel

class Pose:
    def __init__(
        self,
        skel: Skel,
        quats: np.ndarray,
        root_pos: np.ndarray,
    ) -> None:
        self.skel = skel
        self.quats = quats #[J,]
        self.root_pos = root_pos # [3,]
        
    def forward_kinematics(self):
        if not hasattr(self, "lpos"):
            offsets = self.skel.offsets
            parents = self.skel.parents
            offsets[0] = self.root_pos
        else:
            offsets = self.lpos
        return quat.fk(self.quats, offsets, parents)
    
    def set_local_positions(self):
        offsets = self.skel.offsets
        offsets[0] = self.root_pos
        self.lpos = offsets
    
    def set_global_positions(self):
        _, self.global_positions = self.forward_kinematics()
    
    def set_global_rotations(self):
        self.global_rotations, _ = self.forward_kinematics()
    
    def set_gpos_and_grot(self):
        self.global_rotations, self.global_positions = \
            self.forward_kinematics()
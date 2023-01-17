from __future__ import annotations

import numpy as np
import taichi as ti

class Joint:
    def __init__(self, name: str, index: int, parent: int, offset: list[float], dof:int =None, root: bool=None):
        self.name = name
        self.index = index
        self.parent = parent
        self.offset = offset
        if dof != None:
            self.dof = dof
        else: self.dof = 6 if (parent==-1) else 3
        if root != None:
            self.root = root
        self.root: bool = (parent==-1)

class Skel:
    def __init__(self, joints: list[Joint], skel_name: str=None):
        self.joints = joints
        self.num_joints = len(joints)
        self.skel_name = skel_name
    
    def __len__(self):
        return self.num_joints
    
    # ==================
    # Read only property
    # ==================
    @property
    def offsets(self) -> np.ndarray: # For numpy array
        offsets = []
        for j in self.joints:
            offsets.append(j.offset)
        return np.array(offsets)

    @property
    def offsets_field(self): # For Taichi field
        offsets = ti.Vector.field(3, dtype=float, shape=(self.num_joints,))
        for i in range(len(self.joints)):
            offsets[i] = self.joints[i].offset
        return offsets
    
    @property
    def parents(self) -> list[int]:
        return [j.parent for j in self.joints]

    @property
    def parents_field(self): 
        parents_idx_field = ti.field(int, shape=len(self.joints))
        parents_idx = [joint.parent for joint in self.joints]
        for i in range(len(self.joints)):
            parents_idx_field[i] = parents_idx[i]
        return parents_idx_field
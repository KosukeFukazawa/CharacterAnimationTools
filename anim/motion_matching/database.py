from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree
from util import quat
from anim.skel import Skel
from anim.animation import Animation

class Database(Animation):
    def __init__(
        self, 
        anims: list[Animation]=None, 
        skel: Skel=None,
        quats: np.ndarray=None,
        trans: np.ndarray=None,
        fps: int=None,
        starts: list[int]=None,
        names: list[str]=None,
        db_name: str=None,
    ) -> None:
        self.name = db_name
        
        if anims is not None:
            self.skel = anims[0].skel # all `Skel` must be the same.
            self.quats = np.concatenate([anim.quats for anim in anims], axis=0)
            self.trans = np.concatenate([anim.trans for anim in anims], axis=0)
            self.fps = anims[0].fps # all fps must be the same.
            self.starts = []
            self.ends = []
            self.names = []
            cur_frame = 0
            for anim in anims:
                self.starts.append(cur_frame)
                cur_frame += len(anim)
                self.ends.append(cur_frame-1)
                self.names.append(anim.name)
        else:
            self.skel = skel
            self.quats = quats
            self.trans = trans
            self.fps = fps
            self.starts = starts
            self.names = names
        
        self.num_anim = len(self.starts)
        
    def __len__(self) -> int: return len(self.trans)
    
    def __add__(self, other: Database) -> Database:
        assert isinstance(other, Database)
        starts: list[int] = self.starts.copy()
        ends: list[int] = self.ends.copy()
        names:  list[str] = self.names.copy()
        starts.extend(list(map(lambda x: x+len(self), other.starts)))
        ends.extend(list(map(lambda x: x+len(self), other.ends)))
        names.extend(other.names)
        quats: np.ndarray = np.concatenate([self.quats, other.quats], axis=0)
        trans: np.ndarray = np.concatenate([self.trans, other.trans], axis=0)
        
        return Database(
            skel = self.skel,
            quats=quats,
            trans=trans,
            fps=self.fps,
            starts=starts,
            names=names,
            db_name=self.name,
        )
    
    def cat(self, other: Database) -> None:
        assert isinstance(other, Database)
        self.starts.extend(list(map(lambda x: x+len(self), other.starts)))
        self.ends.extend(list(map(lambda x: x+len(self), other.ends)))
        self.names.extend(other.names)
        self.quats = np.concatenate([self.quats, other.quats], axis=0)
        self.trans = np.concatenate([self.trans, other.trans], axis=0)

    # ============
    #   Velocity
    # ============
    @property
    def gposvel(self) -> np.ndarray:
        gpos : np.ndarray = self.gpos
        gpvel: np.ndarray = np.zeros_like(gpos)
        gpvel[1:] = (gpos[1:] - gpos[:-1]) * self.fps # relative position from previous frame.
        for start in self.starts:
            gpvel[start] = gpvel[start+1] - (gpvel[start+3] - gpvel[start+2])
        return gpvel
    
    @property
    def cposvel(self) -> np.ndarray:
        cpos : np.ndarray = self.cpos
        cpvel: np.ndarray = np.zeros_like(cpos)
        cpvel[1:] = (cpos[1:] - cpos[:-1]) * self.fps # relative position from previous frame.
        for start in self.starts:
            cpvel[start] = cpvel[start+1] - (cpvel[start+3] - cpvel[start+2])
        return cpvel

    @property
    def lrotvel(self) -> np.ndarray:
        """Calculate rotation velocities with rotation vector style."""    
        
        lrot : np.ndarray = self.quats.copy()
        lrvel: np.ndarray = np.zeros_like(self.lpos)
        lrvel[1:] = quat.to_axis_angle(
            quat.abs(
                quat.mul(lrot[1:], quat.inv(lrot[:-1])))
            ) * self.fps
        for start in self.starts:
            lrvel[start] = lrvel[start+1] - (lrvel[start+3] - lrvel[start+2])
        return lrvel
    
    # ===================
    #  Future trajectory
    # ===================
    def clamp_future_idxs(self, offset: int) -> np.ndarray:
        """Function to calculate the frame array for `offset` frame ahead.
        If `offset` frame ahead does not exist, 
        return the last frame in each animation.
        """
        idxs = np.arange(len(self)) + offset
        for end in self.ends:
            idxs = np.where((idxs > end) & (idxs <= end + offset), end, idxs)
        return idxs

@dataclass
class MatchingDatabase:
    features: np.ndarray # [T, num_features]
    means: np.ndarray    # [num_features]
    stds: np.ndarray     # [num_features]
    indices: list[int]

    # The elements belows are used only AABB.
    dense_bound_size: int = None
    dense_bound_mins: np.ndarray = None # [num_bounds, num_features]
    dense_bound_maxs: np.ndarray = None # [num_bounds, num_features]
    sparse_bound_size: int = None
    sparse_bound_mins: np.ndarray = None # [num_bounds, num_features]
    sparse_bound_maxs: np.ndarray = None # [num_bounds, num_features]
    
    # kd-tree
    kdtree: KDTree = None
    
    def __len__(self):
        return len(self.features)
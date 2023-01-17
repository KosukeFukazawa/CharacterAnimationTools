from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Joint:
    name: str
    index: int
    parent: int
    offset: np.ndarray
    root: bool = False
    dof: int = 3


class Skel:
    def __init__(
        self,
        joints: list[Joint],
        skel_name: str="skeleton",
        rest_forward: list[int]=[0, 1, 0]
    ) -> None:
        """Class for skeleton offset definition.

        Args:
            joints (list[Joint]): list of Joints.
            skel_name (str, optional): name of skeleton. Defaults to "skeleton".
        """
        
        self.skel_name = skel_name
        self.joints = joints
        self.rest_forward = rest_forward
    
    def __len__(self) -> int:
        return len(self.joints)
    
    def __getitem__(self, index: int | slice | str) -> Joint | list[Joint]:
        return self.get_joint(index)
    
    @property
    def parents(self) -> list[int]:
        """Get list of all joint's parent indices."""
        return [j.parent for j in self.joints]
    
    @property
    def names(self) -> list[str]:
        """Get all joints names."""
        return [j.name for j in self.joints]
    
    @property
    def offsets(self) -> np.ndarray:
        """Get offsets concatenated by all joint offset"""
        offsets: list[np.ndarray] = []
        for joint in self.joints:
            offsets.append(joint.offset)
        return np.array(offsets)
    
    def get_joint(self, index: int | slice | str) -> Joint | list[Joint]:
        """Get Joint from index or slice."""
        if isinstance(index, str):
            index: int = self.get_index_from_jname(index)
        return self.joints[index]
    
    def get_index_from_jname(self, jname: str) -> int:
        """Get joint name from joint index."""
        jname_list = self.jnames
        return jname_list.index(jname)
    
    def get_children(
        self, 
        index: int | str,
        return_idx: bool = False
    ) -> list[Joint] | list[int]:
        """Get list of children joints or children indices.
        args:
            index: index of joint or name of joint.
            return_idx: if True, return joint index.
        return:
            list of joints or list of indices (if return_idx).
        """
        if isinstance(index, str):
            index: int = self.get_index_from_jname(index)
        
        children: list[Joint] = []
        children_idx: list[int] = []
        for i, parent in enumerate(self.parents):
            if index == parent:
                children.append(self.joints[i])
                children_idx.append(i)
        return children_idx if return_idx else children
    
    def get_parent(
        self, 
        index: int | str,
        return_idx: bool = False
    ) -> Joint | int | None:
        """Get parent joint or parent index.
        args:
            index: index of joint or name of joint.
            return_idx: if True, return joint index.
        return:
            Joint or index (if return_idx).
        """
        if isinstance(index, str):
            _index: int = self.get_index_from_jname(index)
        elif isinstance(index, int):
            _index: int = index
        
        if return_idx: return self.parents[_index]
        elif self.parents[_index] == -1:
            return None
        else:
            return self.joints[self.parents[_index]]
    
    @staticmethod
    def from_names_parents_offsets(
        names: list[str],
        parents: list[int],
        offsets: np.ndarray,
        skel_name: str="skeleton"
    ) -> Skel:
        """Construct new Skel from names, parents, offsets.
        
        args:
            names : list of joints names.
            parents: list of joints parents.
            offsets: numpy array of offsets.
        return:
            Skel
        """
        
        joints = []
        indices = np.arange(len(names))
        for name, idx, parent, offset in zip(names, indices, parents, offsets):
            dof = 6 if parent == -1 else 3
            joints.append(
                Joint(name, idx, parent, offset, (parent==-1), dof)
            )
        return Skel(joints, skel_name)
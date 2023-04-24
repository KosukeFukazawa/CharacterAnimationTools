from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    root: bool = False
    dof: int = 3

def axis_to_vector(axis: str):
    match axis:
        case "x":
            return [1, 0, 0]
        case "y":
            return [0, 1, 0]
        case "z":
            return [0, 0, 1]
        case "-x":
            return [-1, 0, 0]
        case "-y":
            return [0, -1, 0]
        case "-z":
            return [0, 0, -1]
        case _:
            raise ValueError

class Skel:
    def __init__(
        self,
        joints: list[Joint],
        skel_name: str="skeleton",
        rest_forward: list[int]=[0, 0, 1],
        rest_vertical: list[int]=[0, 1, 0],
        forward_axis: str="z",
        vertical_axis: str="y",
    ) -> None:
        """Class for skeleton offset definition.

        Args:
            joints (list[Joint]): list of Joints.
            skel_name (str, optional): name of skeleton. Defaults to "skeleton".
            rest_forward  (list[int], optional): forward direction of the rest pose. Defaults to [0, 0, 1].
            rest_vertical (list[int], optional): vertical direction of the rest pose. Defaults to [0, 1, 0].
            forward_axis (str, optional): forward axis of the coodinates. Defaults to "z".
            vertical_axis (str, optional): vertical axis of the coodinates. Defaults to "y".
        """
        self.skel_name = skel_name
        self.joints = joints
        self.rest_forward = rest_forward
        self.rest_vertical = rest_vertical
        self.forward_axis = forward_axis
        self.forward = axis_to_vector(forward_axis)
        self.vertical_axis = vertical_axis
        self.vertical = axis_to_vector(vertical_axis)
    
    def __len__(self) -> int:
        return len(self.joints)
    
    def __getitem__(self, index: int | slice | str) -> Joint | list[Joint]:
        return self.get_joint(index)
    
    @property
    def indices(self) -> list[int]:
        return [idx for idx in range(len(self))]
    
    @property
    def parents(self) -> list[int]:
        """Get list of all joint's parent indices."""
        return [j.parent for j in self.joints]
    
    @property
    def children(self) -> dict[int: list[int]]:
        children_dict = {}
        for i in range(len(self)):
            children_dict[i] = []
        for i, parent in enumerate(self.parents):
            if parent == -1:
                continue
            children_dict[parent].append(i)
        return children_dict
    
    @property
    def names(self) -> list[str]:
        """Get all joints names."""
        return [j.name for j in self.joints]
    
    @property
    def offsets(self) -> np.ndarray:
        """Offset coordinates of all joints (np.ndarray)."""
        offsets: list[np.ndarray] = []
        for joint in self.joints:
            offsets.append(joint.offset)
        return np.array(offsets)
    
    @property
    def dofs(self) -> list[int]:
        return [j.dof for j in self.joints]
    
    @property
    def joint_depths(self) -> list[int]:
        """Get hierarchical distance between joints to ROOT."""
        def get_depth(joint_idx: int, cur_depth: int=0):
            parent_idx = self.parents[joint_idx]
            if  parent_idx != -1:
                depth = cur_depth + 1
                cur_depth = get_depth(parent_idx, depth)
            return cur_depth
        return [get_depth(idx) for idx in self.indices]
    
    @property
    def bone_lengths(self) -> np.ndarray:
        lengths = np.linalg.norm(self.offsets, axis=-1)
        lengths[0] = 0
        return lengths
    
    def get_joint(self, index: int | slice | str) -> Joint | list[Joint]:
        """Get Joint from index or slice or joint name."""
        if isinstance(index, str):
            index: int = self.get_index_from_jname(index)
        return self.joints[index]
    
    def get_index_from_jname(self, jname: str) -> int:
        """Get joint index from joint name."""
        jname_list = self.names
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
        
        children_idx = self.children[index]
        if return_idx:
            return children_idx
        else:
            children = []
            for child_idx in children_idx:
                children.append(self.joints[child_idx])
            return children
    
    def get_parent(
        self, 
        index: int | str,
        return_idx: bool = False
    ) -> Joint | int | None:
        """Get parent joint or parent index.
        args:
            index (int | str): index of joint or name of joint.
            return_idx (bool): if True, return joint index.
        return:
            None : if parent doesn't exists.
            index (int): if return_idx = True.
            joint (Joint): if return_idx = False.
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
        skel_name: str="skeleton",
        rest_forward: list[int]=[0, 0, 1],
        rest_vertical: list[int] = [0, 1, 0],
        forward_axis: str = "z",
        vertical_axis: str = "y"
    ) -> Skel:
        """Construct new Skel from names, parents, offsets.
        
        args:
            names(list[str]) : list of joints names.
            parents(list[int]) : list of joints parents indices.
            offsets(np.ndarray): joint offset relative coordinates from parent joints.
            skel_name (str, optional): name of skeleton. Defaults to "skeleton".
            rest_forward  (list[int], optional): forward direction of the rest pose. Defaults to [0, 0, 1].
            rest_vertical (list[int], optional): vertical direction of the rest pose. Defaults to [0, 1, 0].
            forward_axis (str, optional): forward axis of the coodinates. Defaults to "z".
            vertical_axis (str, optional): vertical axis of the coodinates. Defaults to "y".
        return:
            Skel
        """
        ln, lp, lo = len(names), len(parents), len(offsets)
        assert len(set([ln, lp, lo])) == 1
        joints = []
        for name, parent, offset in zip(names, parents, offsets):
            dof = 6 if parent == -1 else 3
            joints.append(Joint(name, parent, offset, (parent==-1), dof))
        return Skel(joints, skel_name, rest_forward, rest_vertical, forward_axis, vertical_axis)
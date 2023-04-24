"""
animation.py
"""
# Basic animation class. Can use for 
# deep-learning (pre/post)processing 
# and visualization etc.

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
from util import quat
from util import dualquat as dq
from anim.skel import Skel


class Animation:
    def __init__(
        self,
        skel: Skel,
        quats: np.ndarray,
        trans: np.ndarray,
        positions: np.ndarray = None,
        fps: int=30,
        anim_name: str="animation",
    ) -> None:
        """ Class for Motions representations.

        Args:
            skel (Skel): Skelton definition.
            quats (np.ndarray): Joints rotations. shape: (T, J, 4)
            trans (np.ndarray): Root transitions. shape: (T, 3)
            positions (np.ndarray): Joints positions. shape: (T, J, 3). Defaults to None.
            fps (int): Frame per seconds for animation. Defaults to 30.
            anim_name (str, optional): Name of animation. Defaults to "animation".
        """
        self.skel = skel
        self.quats = quats
        self.trans = trans
        self.positions = positions
        self.fps = fps
        self.name = anim_name
    
    def __len__(self) -> int: return len(self.trans)
    
    def __add__(self, other: Animation) -> Animation:
        quats: np.ndarray = np.concatenate([self.quats, other.quats], axis=0)
        trans: np.ndarray = np.concatenate([self.trans, other.trans], axis=0)
        return Animation(
            skel=self.skel,
            quats=quats,
            trans=trans,
            fps=self.fps,
            anim_name=self.name,
        )
    
    def __getitem__(self, index: int | slice) -> Animation:
        if isinstance(index, int):
            if index == -1:
                index = len(self) - 1
            return Animation(
                skel=self.skel,
                quats=self.quats[index:index+1],
                trans=self.trans[index:index+1],
                fps=self.fps,
                anim_name=self.name,
            )
        elif isinstance(index, slice):
            return Animation(
                skel=self.skel,
                quats=self.quats[index],
                trans=self.trans[index],
                fps=self.fps,
                anim_name=self.name,
            )
        else:
            raise TypeError
    
    def cat(self, other: Animation) -> None:
        assert isinstance(other, Animation)

        self.quats = np.concatenate(
            [self.quats, other.quats], axis=0)
        self.trans = np.concatenate(
            [self.trans, other.trans], axis=0
        )
    
    @property
    def parents(self) -> list[int]:
        return self.skel.parents
    
    @property
    def joint_names(self) -> list[str]:
        return self.skel.names
    
    @property
    def offsets(self) -> np.ndarray:
        return self.skel.offsets
    
    # =====================
    #  Rotation conversion
    # =====================
    @property
    def grot(self) -> np.ndarray:
        """global rotations(quaternions) for each joints. shape: (T, J, 4)"""
        return quat.fk_rot(lrot=self.quats, parents=self.parents)
    
    @property
    def crot(self) -> np.ndarray:
        """Projected root(simulation bone) centric global rotations."""
        
        # Root rotations relative to the forward vector.
        root_rot = self.proj_root_rot
        # Root positions projected on the ground (y=0).
        root_pos = self.proj_root_pos()
        # Make relative to simulation bone.
        lrot = self.quats.copy()
        lpos = self.lpos.copy()
        lrot[:, 0] = quat.inv_mul(root_rot, lrot[:, 0])
        lpos[:, 0] = quat.inv_mul_vec(root_rot, lpos[:, 0] - root_pos)
        return quat.fk_rot(lrot, self.parents)
    
    @property
    def axangs(self) -> np.ndarray:
        """axis-angle rotation representations. shape: (T, J, 3)"""
        return quat.to_axis_angle(self.quats)
    
    @property
    def xforms(self) -> np.ndarray:
        """rotation matrix. shape: (T, J, 3, 3)"""
        return quat.to_xform(self.quats)
    
    @property
    def ortho6ds(self) -> np.ndarray:
        return quat.to_xform_xy(self.quats)
    
    @property
    def sw_tws(self) -> tuple[np.ndarray, np.ndarray]:
        """Twist-Swing decomposition.
           This function is based on HybrIK.
           Remove root rotations.
           quat.mul(swings, twists) reproduce original rotations.
        return:
            twists: (T, J-1, 4), except ROOT.
            swings: (T, J-1, 4)
        """
        
        rots: np.ndarray = self.quats.copy()
        pos:  np.ndarray = self.lpos.copy()
        
        children: np.ndarray = \
            -np.ones(len(self.parents), dtype=int)
        for i in range(1, len(self.parents)):
            children[self.parents[i]] = i
        
        swings: list[np.ndarray]  = []
        twists: list[np.ndarray]  = []
        
        for i in range(1, len(self.parents)):
            # ルートに最も近いJointはtwistのみ / only twist for nearest joint to root.
            if children[i] < 0:
                swings.append(quat.eye([len(self), 1]))
                twists.append(rots[:, i:i+1])
                continue
            
            u: np.ndarray = pos[:, children[i]:children[i]+1]
            rot: np.ndarray = rots[:, i:i+1]
            u = u / np.linalg.norm(u, axis=-1, keepdims=True)
            v: np.ndarray = quat.mul_vec(rot, u)
            
            swing: np.ndarray = quat.normalize(quat.between(u, v))
            swings.append(swing)
            twist: np.ndarray = quat.inv_mul(swing, rot)
            twists.append(twist)
        
        swings_np: np.ndarray = np.concatenate(swings, axis=1)
        twists_np: np.ndarray = np.concatenate(twists, axis=1)

        return swings_np, twists_np

    # ==================
    #  Position from FK
    # ==================
    def set_positions_from_fk(self) -> None:
        self.positions = self.gpos
    
    @property
    def lpos(self) -> np.ndarray:
        lpos: np.ndarray = self.offsets[None].repeat(len(self), axis=0)
        lpos[:,0] = self.trans
        return lpos
    
    @property
    def gpos(self) -> np.ndarray:
        """Global space positions."""
        
        _, gpos = quat.fk(
            lrot=self.quats, 
            lpos=self.lpos, 
            parents=self.parents
        )
        return gpos
    
    @property
    def rtpos(self) -> np.ndarray:
        """Root-centric local positions."""
        
        lrots: np.ndarray = self.quats.copy()
        lposs: np.ndarray = self.lpos.copy()
        # ROOT to zero.
        lrots[:,0] = quat.eye([len(self)])
        lposs[:,0] = np.zeros([len(self), 3])
        
        _, rtpos = quat.fk(
            lrot=lrots, 
            lpos=lposs, 
            parents=self.parents
        )
        return rtpos
    
    @property
    def cpos(self) -> np.ndarray:
        """Projected root(simulation bone) centric positions."""
        lrot = self.quats.copy()
        lpos = self.lpos
        c_root_rot, c_root_pos = self.croot()
        lrot[:, 0] = c_root_rot
        lpos[:, 0] = c_root_pos
        
        crot, cpos = quat.fk(lrot, lpos, self.parents)
        return cpos
    
    def croot(self, idx: int=None) -> tuple[np.ndarray, np.ndarray]:
        """return character space info.
        return:
            crot: character space root rotations. [T, 4]
            cpos: character space root positions. [T, 3]
        """
        # Root rotations relative to the forward vector.
        root_rot = self.proj_root_rot
        # Root positions projected on the ground (y=0).
        root_pos = self.proj_root_pos()
        # Make relative to simulation bone.
        crot = quat.inv_mul(root_rot, self.quats[:, 0])
        cpos = quat.inv_mul_vec(root_rot, self.trans - root_pos)
        if idx is not None:
            return crot[idx], cpos[idx]
        else: return crot, cpos
    
    # ============
    #   Velocity
    # ============
    @property
    def gposvel(self) -> np.ndarray:
        gpos : np.ndarray = self.gpos
        gpvel: np.ndarray = np.zeros_like(gpos)
        gpvel[1:] = (gpos[1:] - gpos[:-1]) * self.fps # relative position from previous frame.
        gpvel[0] = gpvel[1] - (gpvel[3] - gpvel[2])
        return gpvel
    
    @property
    def cposvel(self) -> np.ndarray:
        cpos : np.ndarray = self.cpos
        cpvel: np.ndarray = np.zeros_like(cpos)
        cpvel[1:] = (cpos[1:] - cpos[:-1]) * self.fps # relative position from previous frame.
        cpvel[0] = cpvel[1] - (cpvel[3] - cpvel[2])
        return cpvel

    @property
    def lrotvel(self) -> np.ndarray:
        """Calculate rotation velocities with 
           rotation vector style."""    
        
        lrot : np.ndarray = self.quats.copy()
        lrvel: np.ndarray = np.zeros_like(self.lpos)
        lrvel[1:] = quat.to_axis_angle(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) * self.fps
        lrvel[0] = lrvel[1] - (lrvel[3] - lrvel[2])
        return lrvel
    
    # ==============
    #   4x4 matrix
    # ==============
    @property
    def local_transform(self) -> np.ndarray:
        xforms: np.ndarray = self.xforms
        offsets: np.ndarray = self.offsets
        transform: np.ndarray = np.zeros(xforms.shape[:-2] + (4, 4,))
        transform[..., :3, :3] = xforms
        transform[..., :3,  3] = offsets
        transform[..., 3, 3] = 1
        return transform
    
    @property
    def global_transform(self) -> np.ndarray:
        ltrans: np.ndarray = self.local_transform.copy()
        parents: list[int] = self.parents
        
        gtrans = [ltrans[...,:1,:,:]]
        for i in range(1, len(parents)):
            gtrans.append(np.matmul(gtrans[parents[i]],ltrans[...,i:i+1,:,:]))
        
        return np.concatenate(gtrans, axis=-3)
    
    # ==================
    #  dual quaternions
    # ==================
    @property
    def local_dualquat(self) -> np.ndarray:
        return dq.from_rot_and_trans(self.quats, self.lpos)
    
    @property
    def global_dualquat(self) -> np.ndarray:
        return dq.fk(self.local_dualquat, self.parents)
    
    # =============
    #  trajectory 
    # =============
    def proj_root_pos(self, remove_vertical: bool=False) -> np.ndarray:
        """Root position projected on the ground (world space).
        return:
            Projected bone positons as ndarray of shape [len(self), 3] or [len(self), 2](remove_vertical).
        """
        vertical = self.skel.vertical
        if remove_vertical:
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return self.trans[..., settle_ax]
        else:
            return self.trans * np.array([abs(abs(ax) - 1) for ax in vertical])
    
    @property
    def proj_root_rot(self) -> np.ndarray:
        # root rotations relative to the forward on the ground. [len(self), 4]
        forward = self.skel.forward
        return quat.normalize(
            quat.between(np.array(forward), self.root_direction())
        )
    
    def root_direction(self, remove_vertical: bool=False) -> np.ndarray:
        """Forward orientation vectors on the ground (world space).
        return:
            Forward vectors as ndarray of shape [..., 3] or [..., 2](remove_vertical).
        """
        # Calculate forward vectors except vertical axis.
        vertical = self.skel.vertical
        rt_rots = self.quats[..., 0, :]
        forwards = np.zeros(shape=rt_rots.shape[:-1] + (3,))
        forwards[...,] = self.skel.rest_forward
        rt_fwd = quat.mul_vec(rt_rots, forwards) * np.array([abs(abs(ax) - 1) for ax in vertical]) # [T, 3]
        # Normalize vectors.
        norm_rt_fwd = rt_fwd / np.linalg.norm(rt_fwd, axis=-1, keepdims=True)
        if remove_vertical:
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            norm_rt_fwd = norm_rt_fwd[..., settle_ax]
        return norm_rt_fwd
    
    # ===================
    #  Future trajectory
    # ===================
    def future_traj_poss(self, frame: int, remove_vertical: bool=True, cspace=True) -> np.ndarray:
        """Calculate future trajectory positions on simulation bone.
        Args:
            frame (int): how many ahead frame to see.
            remove_vertical (bool, optional): remove vertical axis positions. Defaults to True.
            cspace (bool, optional): use local character space. Defaults to True.
        Returns:
            np.ndarray: future trajectories positions. shape=(len(self), 3) or (len(self), 2)
        """
        idxs = self.clamp_future_idxs(frame)
        proj_root_pos = self.proj_root_pos()
        if cspace:
            traj_pos = quat.inv_mul_vec(self.proj_root_rot, proj_root_pos[idxs] - proj_root_pos)
        else:
            traj_pos = proj_root_pos[idxs]

        if remove_vertical:
            vertical = self.skel.vertical
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return traj_pos[..., settle_ax]
        else: 
            return traj_pos

    def future_traj_dirs(self, frame: int, remove_vertical: bool=True, cspace=True) -> np.ndarray:
        """Calculate future trajectory directions on simulation bone (local character space).
        Args:
            frame (int): how many ahead frame to see.
            remove_vertical (bool, optional): remove vertical axis. Defaults to True.
            cspace (bool, optional): use local character space. Defaults to True.
        Returns:
            np.ndarray: future trajectories directions. shape=(len(self), 3) or (len(self), 2)
        """
        idxs = self.clamp_future_idxs(frame)
        root_directions = self.root_direction()
        if cspace:
            traj_dir = quat.inv_mul_vec(self.proj_root_rot, root_directions[idxs])
        else:
            traj_dir = root_directions[idxs]
        
        if remove_vertical:
            vertical = self.skel.vertical
            settle_ax = []
            for i, ax in enumerate(vertical):
                if ax == 0:
                    settle_ax.append(i)
            return traj_dir[..., settle_ax]
        else: 
            return traj_dir
    
    def clamp_future_idxs(self, offset: int) -> np.ndarray:
        """Function to calculate the frame array for `offset` frame ahead.
        If `offset` frame ahead does not exist, 
        return the last frame.
        """
        idxs = np.arange(len(self)) + offset
        idxs[-(offset + 1):] = idxs[-(offset + 1)]
        return idxs
    
    # =====================
    #    Other functions
    # =====================
    def calc_foot_contact(
        self, 
        method: str="velocity",
        threshold: float=0.15,
        left_foot_name: str="LeftToe",
        right_foot_name: str="RightToe",
    ) -> np.ndarray:
        if method == "velocity":
            contact_vel = np.linalg.norm(
                self.gposvel[:, 
                    [self.joint_names.index(left_foot_name), 
                    self.joint_names.index(right_foot_name)]
                ], axis=-1
            )
            contacts = contact_vel < threshold
        elif method == "position":
            # vertical axis position for each frame.
            settle_idx = 0
            inverse = False # Is the negative direction vertical?
            for i, ax in enumerate(self.skel.vertical):
                if ax == 1:
                    settle_idx = i
                if ax == -1:
                    settle_idx = i
                    inverse = True 
            contact_pos = self.gpos[:,
                [self.joint_names.index(left_foot_name), 
                self.joint_names.index(right_foot_name)], settle_idx]
            if inverse:
                contact_pos *= -1
            contacts = contact_pos < threshold
        else:
            raise ValueError("unknown value selected on `method`.")
        for ci in range(contacts.shape[1]):
            contacts[:, ci] = ndimage.median_filter(
                contacts[:, ci],
                size=6,
                mode="nearest"
            )
        return contacts
    
    def mirror(self, dataset: str=None) -> Animation:
        vertical = self.skel.vertical
        forward = self.skel.forward
        mirror_axis = []
        for vert_ax, fwd_ax in zip(vertical, forward):
            if abs(vert_ax) == 1 or abs(fwd_ax) == 1:
                mirror_axis.append(1)
            else:
                mirror_axis.append(-1)
        if dataset == "lafan1":
            quatM, lposM = animation_mirror(
                lrot=self.quats,
                lpos=self.lpos,
                names=self.joint_names,
                parents=self.parents
            )
            transM = lposM[:, 0]
        else:
            quatM, transM = mirror_rot_trans(
                lrot=self.quats,
                trans=self.trans,
                names=self.joint_names,
                parents=self.parents,
                mirror_axis=mirror_axis,
            )
        return Animation(
            skel=self.skel,
            quats=quatM,
            trans=transM,
            fps=self.fps,
            anim_name=self.name+"_M",
        )
    
    @staticmethod
    def no_animation(
        skel: Skel, 
        fps: int=30,
        num_frame: int=1,
        anim_name: str="animation",
    ) -> Animation:
        """Create a unit animation (no rotation, no transition) from Skel"""
        quats = quat.eye([num_frame, len(skel)])
        trans = np.zeros([num_frame, 3])
        return Animation(skel, quats, trans, None, fps, anim_name)


def mirror_rot_trans(
    lrot: np.ndarray,
    trans: np.ndarray,
    names: list[str],
    parents: np.ndarray | list[int],
    mirror_axis: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    
    lower_names = [n.lower() for n in names]
    joints_mirror: np.ndarray = np.array([(
        lower_names.index("left"+n[5:]) if n.startswith("right") else
        lower_names.index("right"+n[4:]) if n.startswith("left") else 
        lower_names.index(n)) for n in lower_names])

    mirror_pos: np.ndarray = np.array(mirror_axis)
    mirror_rot: np.ndarray = np.array([1,] + [-ax for ax in mirror_axis])
    grot: np.ndarray = quat.fk_rot(lrot, parents)
    trans_mirror: np.ndarray = mirror_pos * trans
    grot_mirror: np.ndarray = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

# for LAFAN1 dataset. This function from 
# https://github.com/orangeduck/Motion-Matching/blob/5e57a1352812494cca83396785cfc4985df9170c/resources/generate_database.py#L15
def animation_mirror(lrot, lpos, names, parents):

    joints_mirror = np.array([(
        names.index('Left'+n[5:]) if n.startswith('Right') else (
        names.index('Right'+n[4:]) if n.startswith('Left') else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:,joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))
    
    return quat.ik(grot_mirror, gpos_mirror, parents)
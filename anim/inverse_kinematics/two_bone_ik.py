from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import copy
import numpy as np
import matplotlib.pyplot as plt
from util import quat
from anim.skel import Joint, Skel
from anim.animation import Animation

def normalize(tensor: np.ndarray, axis:int=-1) -> np.ndarray:
    """ Returns a tensor normalized in the `axis` dimension.
    args: 
        tensor -> tensor / norm(tensor, axis=-axis)
        axis: dimemsion.
    """
    norm = np.linalg.norm(tensor, axis=axis, keepdims=True)
    norm = np.where(norm==0, 1e-10, norm)
    return tensor / norm
    

def two_bone_ik(
    anim: Animation,
    base_joint: int,
    middle_joint: int,
    eff_joint: int,
    tgt_pos: np.ndarray,
    fwd: np.ndarray,
    max_length_buffer: float=1e-3,
) -> Animation:
    """Simple two bone IK method. This algorithm can be solved analytically.
    args:
        anim: Animation
        base_joint: index of base (fixed) joint.
        middle_joint: index of middle joint.
        eff_joint: index of effector joint.
        tgt_pos: (T, 3), target positions for effector.
        fwd: (T, 3), forward vectors for middle joint.
        max_length_buffer: エラー回避用
    """
    anim = copy.deepcopy(anim)
    tgt = tgt_pos.copy()
    
    # 連結チェック
    assert anim.parents[eff_joint] == middle_joint and \
           anim.parents[middle_joint] == base_joint
    
    # 骨の最大の伸びを取得
    lbm: float = np.linalg.norm(anim.offsets[middle_joint], axis=-1)
    lme: float = np.linalg.norm(anim.offsets[eff_joint], axis=-1)
    max_length: float = lbm + lme - max_length_buffer
    
    bt_poss = tgt - anim.gpos[:, base_joint] # (T, 3)
    bt_lengths = np.linalg.norm(bt_poss, axis=-1) # (T)
    # targetが骨の届かない位置に存在するフレームを取得
    error_frame = np.where(bt_lengths > max_length)
    # base -> effectorの長さ1のベクトル
    bt_vec = normalize(bt_poss) # (T, 3)
    # 届かないフレームのみ更新
    tgt[error_frame] = anim.gpos[error_frame, base_joint] + \
                           bt_vec[error_frame] * max_length
    
    # get normal vector for axis of rotations.
    axis_rot = np.cross(bt_vec, fwd, axis=-1)
    axis_rot = normalize(axis_rot) # (T, 3)
    
    # get the relative positions.
    be_poss = anim.gpos[:, eff_joint] - anim.gpos[:, base_joint]
    be_vec = normalize(be_poss) # (T, 3)
    bm_poss = anim.gpos[:, middle_joint] - anim.gpos[:, base_joint]
    bm_vec = normalize(bm_poss) # (T, 3)
    me_poss = anim.gpos[:, eff_joint] - anim.gpos[:, middle_joint]
    me_vec = normalize(me_poss) # (T, 3)
    
    # get the current interior angles of the base and middle joints.
    # einsum calculate dot product.
    be_bm_0 = np.arccos(np.clip(np.einsum("ij,ij->i", be_vec, bm_vec), -1, 1)) # (T,)
    mb_me_0 = np.arccos(np.clip(np.einsum("ij,ij->i", -bm_vec, me_vec), -1, 1)) # (T,)
    
    # get the new interior angles of the base and middle joints.
    lbt = np.linalg.norm(tgt - anim.gpos[:, base_joint], axis=-1)
    be_bm_1 = np.arccos(np.clip((lbm * lbm + lbt * lbt - lme * lme)/(2 * lbm * lbt) , -1, 1)) # (T,)
    mb_me_1 = np.arccos(np.clip((lbm * lbm + lme * lme - lbt * lbt)/(2 * lbm * lme) , -1, 1)) # (T,)
    
    r0 = quat.from_angle_axis(be_bm_1 - be_bm_0, axis_rot)
    r1 = quat.from_angle_axis(mb_me_1 - mb_me_0, axis_rot)
    
    # effectorをtarget方向へ向けるための回転
    r2 = quat.from_angle_axis(
        np.arccos(np.clip(np.einsum("ij,ij->i", be_vec, bt_vec), -1, 1)),
        normalize(np.cross(be_vec, bt_vec))
    )
    
    anim.quats[:, base_joint] =  quat.mul(r2, quat.mul(r0, anim.quats[:, base_joint]))
    anim.quats[:, middle_joint] = quat.mul(r1, anim.quats[:, middle_joint])
    
    return anim

if __name__=="__main__":
    """使用例"""
    
    # 簡単なJointの定義
    joints = []
    joints.append(Joint(name="ROOT", parent=-1, offset=np.array([0, 2, 0]), root=True, dof=6))
    joints.append(Joint(name="J1", parent= 0, offset=np.array([0, -1, 0.1])))
    joints.append(Joint(name="J2", parent= 1, offset=np.array([0, -1, -0.1])))
    joints.append(Joint(name="J3", parent= 2, offset=np.array([0, 0, 0.2])))

    skel = Skel(joints=joints, skel_name="sample_skel")
    anim = Animation.no_animation(skel, num_frame=2, anim_name="sample_anim")
    anim.trans = np.array([
        [0, 2, 0],
        [0, 2, 0]
    ])
    
    # target positions for first frame and second frame.
    tgts = np.array([
        [0, 1, 0], 
        [0.2, 0.8, 0.2]
    ])
    
    fwd = np.array([
        [0.2, 0, 1],
        [0.4, 0.2, 0.5]
    ])
    
    fwd_ = normalize(fwd) * 0.2
    
    new_anim = two_bone_ik(
        anim=anim, 
        base_joint=0, 
        middle_joint=1, 
        eff_joint=2,
        tgt_pos=tgts,
        fwd=fwd)
    
    fig = plt.figure(figsize=(11, 7))
    
    # first frame
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_xlim3d(-1, 1)
    ax1.set_ylim3d(-1, 1)
    ax1.set_zlim3d( 0, 2)
    ax1.set_box_aspect((1,1,1))
    
    orig = anim.gpos[0]
    poss = new_anim.gpos[0]
    tgt  = tgts[0]
    
    ax1.plot(orig[:,0], -orig[:,2], orig[:,1], alpha=0.4)
    ax1.scatter(orig[:,0], -orig[:,2], orig[:,1], label="orig", alpha=0.4)
    ax1.plot(poss[:,0], -poss[:,2], poss[:,1])
    ax1.scatter(poss[:,0], -poss[:,2], poss[:,1], label="calc")
    ax1.plot(
        [poss[1,0], poss[1,0]+fwd_[0, 0]], 
        [-poss[1,2], -poss[1,2]-fwd_[0, 2]], 
        [poss[1,1], poss[1,1]+fwd_[0, 1]], label="fwd")
    ax1.scatter(tgt[0], -tgt[2], tgt[1], s=80, color="red", marker="*", label="target")
    ax1.legend()

    # second frame
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlim3d(-1, 1)
    ax2.set_ylim3d(-1, 1)
    ax2.set_zlim3d(0, 2)
    ax2.set_box_aspect((1,1,1))
    
    orig = anim.gpos[1]
    poss = new_anim.gpos[1]
    tgt  = tgts[1]
    
    ax2.plot(orig[:,0], -orig[:,2], orig[:,1], alpha=0.4)
    ax2.scatter(orig[:,0], -orig[:,2], orig[:,1], label="orig", alpha=0.4)
    ax2.plot(poss[:,0], -poss[:,2], poss[:,1])
    ax2.scatter(poss[:,0], -poss[:,2], poss[:,1], label="calc")
    ax2.plot(
        [poss[1,0], poss[1,0]+fwd_[1, 0]], 
        [-poss[1,2], -poss[1,2]-fwd_[1, 2]], 
        [poss[1,1], poss[1,1]+fwd_[1, 1]], label="fwd")
    ax2.scatter(tgt[0], -tgt[2], tgt[1], s=80, color="red", marker="*", label="target")
    ax2.legend()
    
    plt.show()
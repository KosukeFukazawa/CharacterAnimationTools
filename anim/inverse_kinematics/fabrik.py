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

def normalize_vector(vector: np.ndarray):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    norm = np.where(norm==0, 1e-10, norm)
    return vector / norm

def backward_reaching(
    anim: Animation,
    tgt_pos: np.ndarray,
) -> Animation:
    
    joint_idxs = anim.skel.indices
    target = tgt_pos.copy() # (T, 3)
    cur_gposs = anim.gpos.copy()
    bone_lengths = anim.skel.bone_lengths
    for j, parent in zip(reversed(joint_idxs), reversed(anim.parents)):
        if parent == -1:
            continue
        cur_quat = anim.quats[:,parent] # (T, 4)
        
        # 現在のJointを基準とした更新前の子Joint位置(global座標系)
        rel_chpos = cur_gposs[:,j] - cur_gposs[:,parent]
        ch_vec = normalize_vector(rel_chpos)
        
        # 現在のJointを基準とした更新後の子Joint位置(global座標系)
        rel_tgtpos = target - cur_gposs[:, parent]
        tgt_vec = normalize_vector(rel_tgtpos)
        
        rel_quat = quat.normalize(quat.between(ch_vec, tgt_vec))
        anim.quats[:,parent] = quat.abs(quat.normalize(quat.mul(rel_quat, cur_quat)))
        
        # 親がROOTのとき、offsetを変更
        if parent == 0:
            anim.trans = target - tgt_vec * bone_lengths[j]
        else:
            # targetの更新
            target = target - tgt_vec * bone_lengths[j]
        
    return anim

def forward_reaching(
    anim: Animation,
    base: np.ndarray,
) -> Animation:
    
    joint_idxs = np.arange(len(anim.skel))
    target = base.copy() # (T, 3)
    past_gposs = anim.gpos.copy()
    
    for j, parent in zip(joint_idxs, anim.parents):
        if parent == -1:
            continue
        elif parent == 0:
            anim.trans = target
            continue
        cur_quat = anim.quats[:,parent] # (T, 4)
        
        # 過去の親 -> 過去の子のJoint位置
        past_chpos = past_gposs[:,j] - past_gposs[:,parent]
        pa_vec = normalize_vector(past_chpos)
        
        # 現在の親 -> 過去の子のJoint位置
        cur_chpos = past_gposs[:,j] - anim.gpos[:,parent]
        cur_vec = normalize_vector(cur_chpos)
        
        rel_quat = quat.normalize(quat.between(pa_vec, cur_vec))
        anim.quats[:,parent] = quat.abs(quat.normalize(quat.mul(rel_quat, cur_quat)))
        
    return anim

# まずはEnd effectorのみの実装(Jointの末端が一つのみ)
def simple_fabrik(
    anim: Animation,
    tgt_pos: np.ndarray,
    iter: int=10,
) -> Animation:
    
    anim_ = copy.deepcopy(anim)
    # define base(root) positions.
    base = anim.trans.copy() # (T, 3)
    
    for i in range(iter):
        anim_ = backward_reaching(anim_, tgt_pos)
        anim_ = forward_reaching(anim_, base)
    
    return anim_

if __name__=="__main__":
    """使用例"""
    
    # 簡単なJointの定義
    joints = []
    joints.append(Joint(name="ROOT", parent=-1, offset=np.zeros(3), root=True, dof=6))
    joints.append(Joint(name="J1", parent= 0, offset=np.array([1, 0, 0])))
    joints.append(Joint(name="J2", parent= 1, offset=np.array([1, 0, 0])))
    joints.append(Joint(name="J3", parent= 2, offset=np.array([1, 0, 0])))
    #    Root       J1         J2         J3
    #     *----------*----------*----------*
    # (0, 0, 0)  (1, 0, 0)  (2, 0, 0)  (3, 0, 0)


    skel = Skel(joints=joints, skel_name="sample_skel")
    anim = Animation.no_animation(skel, num_frame=2, anim_name="sample_anim")
    
    # target positions for first frame and second frame.
    tgts = np.array([
        [1, 1, 1], 
        [1, 2, 1]]
    )
    
    new_anim = simple_fabrik(anim, tgts)
    
    fig = plt.figure(figsize=(11, 7))
    
    # first frame
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_box_aspect((1,1,1))
    
    orig = anim.gpos[0]
    poss = new_anim.gpos[0]
    tgt  = tgts[0]
    
    ax1.plot(orig[:,0], orig[:,1], orig[:,2], alpha=0.4)
    ax1.scatter(orig[:,0], orig[:,1], orig[:,2], alpha=0.4, label="orig")
    ax1.plot(poss[:,0], poss[:,1], poss[:,2])
    ax1.scatter(poss[:,0], poss[:,1], poss[:,2], label="calc")
    ax1.scatter(tgt[0], tgt[1], tgt[2], s=50, marker="*", label="target")
    ax1.legend()

    # second frame
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlim3d(-2, 2)
    ax2.set_ylim3d(-2, 2)
    ax2.set_zlim3d(-2, 2)
    ax2.set_box_aspect((1,1,1))
    
    orig = anim.gpos[1]
    poss = new_anim.gpos[1]
    tgt  = tgts[1]
    
    ax2.plot(orig[:,0], orig[:,1], orig[:,2], alpha=0.4)
    ax2.scatter(orig[:,0], orig[:,1], orig[:,2], alpha=0.4, label="orig")
    ax2.plot(poss[:,0], poss[:,1], poss[:,2])
    ax2.scatter(poss[:,0], poss[:,1], poss[:,2], label="calc")
    ax2.scatter(tgt[0], tgt[1], tgt[2], s=50, marker="*", label="target")
    ax2.legend()
    
    plt.savefig("demo.png")
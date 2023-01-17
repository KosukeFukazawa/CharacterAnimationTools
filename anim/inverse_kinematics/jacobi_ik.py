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

def simple_jacobi_ik(
    anim: Animation,
    tgt_pos: np.ndarray,
    iter: int=10,
) -> Animation:
    return

if __name__=="__main__":
    """使用例"""
    
    # 簡単なJointの定義
    joints = []
    joints.append(Joint(name="ROOT", index=0, parent=-1, offset=np.zeros(3), root=True, dof=6))
    joints.append(Joint(name="J1", index=1, parent= 0, offset=np.array([1, 0, 0])))
    joints.append(Joint(name="J2", index=2, parent= 1, offset=np.array([1, 0, 0])))
    joints.append(Joint(name="J3", index=3, parent= 2, offset=np.array([1, 0, 0])))
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
    
    new_anim = simple_jacobi_ik(anim, tgts)
    
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
    
    ax1.plot(poss[:,0], poss[:,1], poss[:,2])
    ax1.scatter(poss[:,0], poss[:,1], poss[:,2], label="calc")
    ax1.plot(orig[:,0], orig[:,1], orig[:,2])
    ax1.scatter(orig[:,0], orig[:,1], orig[:,2], label="orig")
    ax1.scatter(tgt[0], tgt[1], tgt[2], label="target")
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
    
    ax2.plot(poss[:,0], poss[:,1], poss[:,2])
    ax2.scatter(poss[:,0], poss[:,1], poss[:,2], label="calc")
    ax2.plot(orig[:,0], orig[:,1], orig[:,2])
    ax2.scatter(orig[:,0], orig[:,1], orig[:,2], label="orig")
    ax2.scatter(tgt[0], tgt[1], tgt[2], label="target")
    ax2.legend()
    
    plt.show()
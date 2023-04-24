"""
aistpp.py
AIST++ : 
    format: ***.pkl (pickle file)
    Parameters :
        "smpl_loss" (float): I don't know how to use it.
        "smpl_poses" (np.ndarray): SMPL pose paramters (rotations). shape: (num_frame, num_J * 3 = 72).
        "smpl_scaling" (float): Scaling parameters of the skeleton.
        "smpl_trans" (np.ndarray): Root translations. shape: (num_frame, 3).
"""
# loading AIST++ data.

from __future__ import annotations

from pathlib import Path
import numpy as np
from anim.skel import Skel
from anim.animation import Animation
from anim.smpl import load_model, calc_skel_offsets, SMPL_JOINT_NAMES
from util import quat
from util.load import pickle_load

def load(
    aistpp_motion_path: Path,
    skel: Skel=None,
    smpl_path: Path=Path("data/smpl/neutral/model.pkl"),
    scale: float=100.0,
    fps: int=30,
) -> Animation:
    
    NUM_JOINTS = 24
    NUM_BETAS = 10
    
    # Load a motion.
    if isinstance(aistpp_motion_path, str):
        aistpp_motion_path = Path(aistpp_motion_path)
    name = aistpp_motion_path.stem
    aistpp_dict = pickle_load(aistpp_motion_path)
    poses = aistpp_dict["smpl_poses"]
    num_frames = len(poses)
    poses = poses.reshape(num_frames, -1, 3)
    quats = quat.from_axis_angle(poses)
    scale /= aistpp_dict["smpl_scaling"]
    trans = aistpp_dict["smpl_trans"] / aistpp_dict["smpl_scaling"]

    if skel == None:
        # Load a Skel.
        smpl_dict = load_model(smpl_path)
        parents = list(map(int, smpl_dict["kintree_table"][0][:NUM_JOINTS]))
        parents[0] = -1
        J_regressor = smpl_dict["J_regressor"]
        shapedirs = smpl_dict["shapedirs"]
        v_template = smpl_dict["v_template"]
        betas = np.zeros([NUM_BETAS])
        J_positions = calc_skel_offsets(betas, J_regressor, shapedirs, v_template)[:NUM_JOINTS] * scale
        root_offset = J_positions[0]
        offsets = J_positions - J_positions[parents]
        offsets[0] = root_offset
        names = SMPL_JOINT_NAMES[:NUM_JOINTS]
        skel = Skel.from_names_parents_offsets(names, parents, offsets)

    # We need to apply smpl root offsets to translations.
    trans += root_offset
    
    return Animation(skel=skel, quats=quats, trans=trans, fps=fps, anim_name=name,)
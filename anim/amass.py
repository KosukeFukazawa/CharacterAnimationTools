"""
amass.py
AMASS :
    Format: ***.npz (numpy binary file)
    Parameters:
        "trans" (np.ndarray): Root translations. shape: (num_frames, 3).
        "gender" (np.ndarray): Gender name. "male", "female", "neutral".
        "mocap_framerate" (np.ndarray): Fps of mocap data.
        "betas" (np.ndarray): PCA based body shape parameters. shape: (num_betas=16).
        "dmpls" (np.ndarray): Dynamic body parameters of DMPL. We do not use. Unused for skeleton. shape: (num_frames, num_dmpls=8).
        "poses" (np.ndarray): SMPLH pose parameters (rotations). shape: (num_frames, num_J * 3 = 156).
"""
# loading amass data.

from __future__ import annotations

from pathlib import Path
import numpy as np
from anim.skel import Skel
from anim.animation import Animation
from anim.smpl import load_model, calc_skel_offsets, SMPL_JOINT_NAMES, SMPLH_JOINT_NAMES
from util import quat

def load(
    amass_motion_path: Path,
    skel: Skel=None,
    smplh_path: Path=Path("data/smplh/neutral/model.npz"),
    remove_betas: bool=False,
    gender: str=None,
    num_betas: int=16,
    scale: float=100.0,
    load_hand=True,
) -> Animation:
    """
    args:
        amass_motion_file (Path): Path to the AMASS motion file.
        smplh_path (Path): Optional. Path to the SMPLH model.
        remove_betas (bool): remove beta parameters from AMASS.
        gender (str): Gender of SMPLH model.
        num_betas (int): Number of betas to use. Defaults to 16.
        num_dmpls (int): Number of dmpl parameters to use. Defaults to 8.
        scale (float): Scaling paramter of the skeleton. Defaults to 100.0.
        load_hand (bool): Whether to use hand joints. Defaults to True.
        load_dmpl (bool): Whether to use DMPL. Defaults to False.
    Return:
        anim (Animation)
    """
    if load_hand: 
        NUM_JOINTS = 52
        names = SMPLH_JOINT_NAMES[:NUM_JOINTS]
    else: 
        NUM_JOINTS = 24
        names = SMPL_JOINT_NAMES[:NUM_JOINTS]

    if not isinstance(amass_motion_path, Path):
        amass_motion_path = Path(amass_motion_path)
    if not isinstance(smplh_path, Path):
        smplh_path = Path(smplh_path)
    
    
    trans_quat = quat.mul(quat.from_angle_axis(-np.pi / 2, [0, 1, 0]), quat.from_angle_axis(-np.pi / 2, [1, 0, 0]))

    # Load AMASS info.
    amass_dict = np.load(amass_motion_path, allow_pickle=True)
    if gender == None: gender = str(amass_dict["gender"])
    fps = int(amass_dict["mocap_framerate"])
    if remove_betas: betas = np.zeros([num_betas])
    else: betas = amass_dict["betas"][:num_betas]
    axangles = amass_dict["poses"]
    num_frames = len(axangles)
    axangles = axangles.reshape([num_frames, -1, 3])[:,:NUM_JOINTS]
    quats = quat.from_axis_angle(axangles)
    root_rot = quat.mul(trans_quat[None], quats[:,0])
    quats[:,0] = root_rot

    if skel == None:
        # Load SMPL parmeters.
        smplh_dict = load_model(smplh_path, gender)
        parents = smplh_dict["kintree_table"][0][:NUM_JOINTS]
        parents[0] = -1
        J_regressor = smplh_dict["J_regressor"]
        shapedirs = smplh_dict["shapedirs"]
        v_template = smplh_dict["v_template"]
        
        J_positions = calc_skel_offsets(betas, J_regressor, shapedirs, v_template)[:NUM_JOINTS] * scale
        root_offset = J_positions[0]
        offsets = J_positions - J_positions[parents]
        offsets[0] = root_offset
        skel = Skel.from_names_parents_offsets(names, parents, offsets, skel_name="SMPLH")
    
    root_pos = skel.offsets[0][None].repeat(len(quats), axis=0)
    trans = amass_dict["trans"] * scale + root_pos
    trans = trans @ quat.to_xform(trans_quat).T
    anim = Animation(skel=skel, quats=quats, trans=trans, fps=fps, anim_name=amass_motion_path.stem)

    return anim
"""
amass.py
"""
# loading and writing amass data.

from __future__ import annotations

from pathlib import Path
import torch
import numpy as np
import pickle
from smplx.body_models import SMPLH
from smplx.joint_names import SMPLH_JOINT_NAMES
from anim.skel import Skel
from anim.animation import Animation
from util import quat

def load(
    npzfile: Path,
    smplh_path: Path="data/smplh",
    remove_betas: bool=False,
    num_betas=10
) -> Animation:
    amass = np.load(npzfile)
    model = SMPLH(smplh_path)
    parents = model.parents.detach().cpu().numpy()
    
    if remove_betas: betas = torch.zeros([1, num_betas])
    else: betas = amass["betas"]
    rest = model(betas=betas)
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:len(SMPLH_JOINT_NAMES),:]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset

    aas = amass["poses"].reshape()
    quats = quat.from_axis_angle(amass["poses"])
    amass["poses"] # (T, J*3)
    root_pos = offsets[None].repeat(len(poses), axis=0)
    trans = amass["trans"] + root_pos # (T, 3)
    fps = amass["mocap_framerate"]
    
    dmpls = amass["dmpls"]
    



def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE", 
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.
    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]
    
    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100
    
    scaling = None
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")
    
    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)

if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
             mirror = args.mirror, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, 
             fps=args.fps, output=args.output)
    
    print("finished!")
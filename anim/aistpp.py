"""
aistpp.py
"""
# loading and writing aist++ data.

from __future__ import annotations

from pathlib import Path
import numpy as np

from anim.skel import Skel
from anim.animation import Animation
from util import quat
from util.load import pickle_load

def load(
    pklpath: Path,
    skel: Skel=None,
    skel_cfg: Path=Path("configs/skel_smpl_neutral.npz"),
    load_skel: bool=True,
    fps: int = 30,
) -> Animation:
    
    if load_skel:
        # Load a Skel.
        skel_info = np.load(skel_cfg, allow_pickle=True)
        skel = Skel.from_names_parents_offsets(
            names = skel_info["names"].tolist(),
            parents=skel_info["parents"].tolist(),
            offsets=skel_info["offsets"],
            skel_name="SMPL_"+str(skel_info["gender"])
        )
    assert not skel is None, "you shoud define the Skel."
    
    # Load a motion.
    if isinstance(pklpath, str):
        pklpath = Path(pklpath)
    
    name = pklpath.name.split(".")[0]
    motion_info = pickle_load(pklpath)
    poses = motion_info["smpl_poses"] # [fnum, 72]
    poses = poses.reshape(len(poses), -1, 3) # [fnum, 24, 3]
    quats = quat.from_axis_angle(poses)
    scaling = motion_info["smpl_scaling"]
    trans = motion_info["smpl_trans"] # [fnum, 3]
    trans /= scaling

    # We need to add smpl root offsets to translations.
    trans += skel_info["offsets"][0]
    
    return Animation(skel=skel, quats=quats, trans=trans, fps=fps, anim_name=name,)
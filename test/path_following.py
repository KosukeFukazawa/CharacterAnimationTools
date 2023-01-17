from __future__ import annotations

import sys
from pathlib import Path
import pickle
import numpy as np

BASEPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASEPATH))

from util.load import pickle_load
from util import quat
from anim import bvh
from anim.animation import Animation
from anim.motion_matching.database import Database
from anim.motion_matching.mm import create_matching_database, motion_matching_search

def create_database(bvh_dir: Path, files: list, starts: list, ends: list) -> Database:
    if isinstance(bvh_dir, str):
        bvh_dir = Path(bvh_dir)
    anims = []
    for file, start, end in zip(files, starts, ends):
        anim = bvh.load(bvh_dir / file, start, end)
        anims.append(anim)
    return Database(anims)

def create_query(db: Database, idx: int, traj_feats: np.ndarray) -> np.ndarray:
    left_foot_idx = db.skel.names.index("LeftFoot")
    right_foot_idx = db.skel.names.index("RightFoot")
    hips_idx = db.skel.names.index("Hips")
    
    fpos = db.cpos[idx, [left_foot_idx, right_foot_idx]].ravel()
    vels = db.cposvel[idx,[left_foot_idx, right_foot_idx, hips_idx]].ravel()
    return np.concatenate([fpos, vels, traj_feats])

def create_circle_traj():
    # Create path (circle)
    t = np.linspace(0, np.pi * 2, 100)
    x = 500 * np.cos(t)
    z = 500 * np.sin(t)
    dir_x = -z
    dir_z = x
    norm = np.sqrt(dir_x ** 2 + dir_z ** 2)
    dir_x = dir_x / norm
    dir_z = dir_z / norm
    x -= 500
    
    return np.array([
        x[20], z[20], x[40], z[40], x[60], z[60],
        dir_x[20], dir_z[20], dir_x[40], dir_z[40], dir_x[60], dir_z[60]
    ])
    

def create_animation_from_idxs(db: Database, anim_frames: list[int]) -> Animation:
    quats = []
    trans = []
    t = np.linspace(0, np.pi * 2, 100)
    x = 500 * np.cos(t) - 500
    z = 500 * np.sin(t)
    
    for i, frame in enumerate(anim_frames):
        c_root_rot, root_pos = db.croot(frame)
        rot = db.quats[frame]
        i = i % 100
        rot[0] = quat.mul(quat.from_angle_axis(t[i], [0, 1, 0]), c_root_rot)
        root_pos[0] = x[i]
        root_pos[2] = z[i]
        
        quats.append(rot)
        trans.append(root_pos)
    
    quats = np.array(quats)
    trans = np.array(trans)
    return Animation(db.skel, quats, trans, db.fps)

def main():
    matching_method = "kd-tree"
    
    bvh_dir = BASEPATH / "data/lafan1"
    files = ["pushAndStumble1_subject5.bvh", "run1_subject5.bvh", "walk1_subject5.bvh"]
    starts = [194, 90, 80]
    ends = [351, 7086, 7791]
    
    load_database = False
    save_database = True
    
    search_time = 5
    search_timer = 0
    
    # Create `Database` and `MatchingDatabase`.
    if load_database:
        db = pickle_load(BASEPATH / "data/db.pkl")
        mdb = pickle_load(BASEPATH / "data/mdb.pkl")
    else:
        db = create_database(bvh_dir, files, starts, ends)
        mdb = create_matching_database(
            db=db, method=matching_method,
            w_pos_foot=0.75, w_vel_foot=1, w_vel_hips=1, w_traj_pos=1, w_traj_dir=1.5,
            ignore_end=20, dense_bound_size=16, sparse_bound_size=64,
        )
    
    if save_database:
        with open(BASEPATH / "data/db.pkl", "wb") as f:
            pickle.dump(db, f)
        with open(BASEPATH / "data/mdb.pkl", "wb") as f:
            pickle.dump(mdb, f)

    # Initialize parameters
    frame_idx = 0
    frame_sum = 0
    animation_length = 120 # frame
    anim_frames = [] # animated frame will apend here.
    
    # We define the path that moves forward 3m every 20 frames.
    #               20,        40,       60    frames ahead.
    # path_poss = [ 200, 0,    400, 0,   600, 0 ]
    # path_dirs = [  0, 1,      0, 1,     0, 1  ]
    # traj_feats = np.concatenate([path_poss, path_dirs])
    
    # Create path (circle)
    traj_feats = create_circle_traj()
    
    # initial query
    query = create_query(db, frame_idx, traj_feats)
    
        # Animation loop
    while frame_sum < animation_length:
        # Check if we reached the end of the current anim.
        if frame_idx in db.ends: 
            end_of_anim = True
            cur_idx = -1
        else: 
            end_of_anim = False
            cur_idx = frame_idx
        
        if search_timer <= 0 or end_of_anim:
            # Motion Matching Search!
            cur_idx = motion_matching_search(cur_idx, matching_method, mdb, query)
            
            if cur_idx != frame_idx:
                frame_idx = cur_idx

            search_timer = search_time
        
        # update frame
        search_timer -= 1
        frame_idx += 1
        frame_sum += 1
        anim_frames.append(frame_idx)
        
        # Create new query
        query = create_query(db, frame_idx, traj_feats)
        
        sys.stdout.write("Processed frame: {}\n".format(frame_sum))
    
    # Save animation.
    print("Creating animation..")
    sim_anim = create_animation_from_idxs(db, anim_frames)
    bvh.save(BASEPATH / "data/sim_motion.bvh", sim_anim)
    print("saved at {}".format(str(BASEPATH / "data/sim_motion.bvh")))

if __name__ == "__main__":
    main()

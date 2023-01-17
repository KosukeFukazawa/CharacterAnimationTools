# TBD

import sys
from pathlib import Path
import pickle

BASEPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASEPATH))

from anim import bvh
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

def main():
    # init configs (this will replace to a yaml file).
    matching_method = "aabb"
    ignore_end = 20
    
    bvh_dir = BASEPATH / "data/lafan1"
    files = ["pushAndStumble1_subject5.bvh", "run1_subject5.bvh", "walk1_subject5.bvh"]
    starts = [194, 90, 80]
    ends = [351, 7086, 7791]
    
    load_database = False
    save_database = True
    
    search_time = 0.1
    search_timer = search_time
    
    desired_velocity_change_threshold = 50.0
    desired_rotation_change_threshold = 50.0
    desired_gait = 0.0
    desired_gait_velocity = 0.0
    
    simulation_velocity_halflife = 0.27
    simulation_rotation_halflife = 0.27
    
    simulation_run_fwrd_speed = 4.0
    simulation_run_side_speed = 3.0
    simulation_run_back_speed = 2.5
    simulation_walk_fwrd_speed = 1.75
    simulation_walk_side_speed = 1.5
    simulation_walk_back_speed = 1.25
    
    dt = 1 / 60
    
    
    # Create `Database` and `MatchingDatabase`.
    db = create_database(bvh_dir, files, starts, ends)
    mdb = create_matching_database(
        db=db, method=matching_method,
        w_pos_foot=0.75, w_vel_foot=1, w_vel_hips=1, w_traj_pos=1, w_traj_dir=1.5,
        dense_bound_size=16, sparse_bound_size=64,
    )
    if save_database:
        with open(BASEPATH / "data/motion_matching/db.pkl", "wb") as f:
            pickle.dump(db, f)
        with open(BASEPATH / "data/motion_matching/db.pkl", "wb") as f:
            pickle.dump(mdb, f)

    # Initialize parameters
    frame_idx = 0
    
    # Init the pose
    
    # Init the simulation (future direction etc.)
    
    # Init the camera
    
    
    
    
    
    
    """ここ以下は繰り返し"""
    
    # Generate query for motion matching.
    query = None
    
    # Check if we reached the end of the current anim.
    if frame_idx in db.ends: 
        end_of_anim = True
        cur_idx = -1
    else: 
        end_of_anim = False
        cur_idx = frame_idx
    
    if search_timer <= 0 or end_of_anim:
        # Motion Matching Search!
        cur_idx = motion_matching_search(cur_idx, matching_method, db, mdb, query, ignore_end)
        
        if cur_idx != frame_idx:
            frame_idx = cur_idx

        search_time = search_timer
    
    search_timer -= dt
    frame_idx += 1
    
    # Update the next pose
    
    # Update the simulation
    
    # Update the camera
    
    # Drawing
    

if __name__ == "__main__":
    main()

# Calculate time for motion matching.
import sys
from pathlib import Path
import timeit

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

bvh_dir = BASEPATH / "data/lafan1"

files = ["pushAndStumble1_subject5.bvh", "run1_subject5.bvh", "walk1_subject5.bvh"]
starts = [194, 90, 80]
ends = [351, 7086, 7791]

matching_method = "kd-tree"

db = create_database(bvh_dir, files, starts, ends)
mdb = create_matching_database(
    db=db, method=matching_method,
    w_pos_foot=0.75, w_vel_foot=1, w_vel_hips=1, w_traj_pos=1, w_traj_dir=1.5,
    ignore_end=20, dense_bound_size=16, sparse_bound_size=64,
)

cur_idx = 100
query = mdb.features[mdb.indices.index(1320)]

iter = 100
score = timeit.timeit(
    "motion_matching_search(cur_idx, matching_method, mdb, query, False)", 
    globals=globals(), 
    number=iter,
)
print(score / iter)
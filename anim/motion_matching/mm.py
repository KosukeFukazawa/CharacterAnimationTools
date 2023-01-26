from __future__ import annotations

import sys
import numpy as np
from scipy.spatial import KDTree
# import taichi as ti

from anim.motion_matching.database import Database, MatchingDatabase

def normalize_features(features: np.ndarray, weights: np.ndarray, axis=0) -> tuple:
    """Normalize function for matching features. (features: [T, num_features])"""
    means = np.mean(features, axis=axis) # [num_feat]
    stds = np.std(features, axis=axis)   # [num_feat]
    scales = np.where(stds==0, 1e-10, stds / weights)
    norms = (features - means[None]) / scales[None]
    return norms, means, scales

def normalize_query(query: np.ndarray, means:np.ndarray, stds:np.ndarray) -> np.ndarray:
    """Normalize function for `query`. 
       `means` and `stds` are calcurated by normalize_features().
    """
    return (query - means) / stds

def calc_box_distance(
    best_cost: float,
    query: np.ndarray,    # [n_features]
    box_mins: np.ndarray, # [n_features]
    box_maxs: np.ndarray, # [n_features]
) -> tuple[float, bool]:
    """Calculate distance between `query` to an AABB."""
    cost = 0.0
    smaller = True
    for i, feat in enumerate(query):
        cost += np.square(feat - np.clip(feat, box_mins[i], box_maxs[i]))
        if cost >= best_cost:
            smaller = False
            break
    return cost, smaller
    
def create_position_features(db: Database) -> np.ndarray:
    left_foot_idx = db.skel.names.index("LeftFoot")
    right_foot_idx = db.skel.names.index("RightFoot")
    features = db.cpos[:, [left_foot_idx, right_foot_idx]].reshape(len(db), -1) # (T, 2 * 3)
    return features

def create_velocity_features(db: Database) -> np.ndarray:
    left_foot_idx = db.skel.names.index("LeftFoot")
    right_foot_idx = db.skel.names.index("RightFoot")
    hips_idx = db.skel.names.index("Hips")
    features = db.cposvel[:, [left_foot_idx, right_foot_idx, hips_idx]].reshape(len(db), -1) # (T, 3 * 3)
    return features

def create_traj_features(db: Database) -> np.ndarray:
    traj_pos0 = db.future_traj_poss(20, remove_vertical=True)
    traj_pos1 = db.future_traj_poss(40, remove_vertical=True)
    traj_pos2 = db.future_traj_poss(60, remove_vertical=True)
    pos_features = np.concatenate([traj_pos0, traj_pos1, traj_pos2], axis=-1) # [T, 2*3]
    
    traj_dir0 = db.future_traj_dirs(20, remove_vertical=True)
    traj_dir1 = db.future_traj_dirs(40, remove_vertical=True)
    traj_dir2 = db.future_traj_dirs(60, remove_vertical=True)
    dir_features = np.concatenate([traj_dir0, traj_dir1, traj_dir2], axis=-1) # [T, 2*3]
    
    features = np.concatenate([pos_features, dir_features], axis=-1) # [T, 2*3 + 2*3]
    return features

def create_matching_features(
    db: Database,
    w_pos_foot: float,
    w_vel_foot: float,
    w_vel_hips: float,
    w_traj_pos: float,
    w_traj_dir: float,
    ignore_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a `MatchingDatabase` to find the best frame with motion matching.
       Each feature is calculated with `create_***_features()` above, 
       and this function integrates them.
    """
    pos_feats = create_position_features(db)
    vel_feats = create_velocity_features(db)
    traj_feats = create_traj_features(db)
    features = np.concatenate([pos_feats, vel_feats, traj_feats], axis=-1)
    
    # remove ignore index from mdb and create a list that ties mdb indices to db indices.
    indices = []
    rmv_starts = []
    for i, (start, end) in enumerate(zip(db.starts, db.ends)):
        rmv_starts.append(end - ignore_end * (i + 1) + 1)
        
        for j in range(start, end - ignore_end + 1):
            indices.append(j)
    for st in rmv_starts:
        np.delete(features, slice(st, st+ignore_end), axis=0)
    
    # normalize
    weights = np.array(
        [w_pos_foot] * 6 +
        [w_vel_foot] * 6 +
        [w_vel_hips] * 3 +
        [w_traj_pos] * 6 +
        [w_traj_dir] * 6
    )
    fnorms, means, stds = normalize_features(features, weights)
    
    return fnorms, means, stds, indices

def create_matching_aabb(features: np.ndarray, size: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Create Axis-Aligned Bounding Boxes (AABB) on frame dim.
       This will speed up the Nearest Neighbour search.
       based on https://dl.acm.org/doi/abs/10.1145/3386569.3392440.
    """
    num_frames = len(features)
    bound_mins = []
    bound_maxs = []
    
    num_bounds = (num_frames - 1) // size + 1 # advance
    frame_bound_ends = [(i + 1) * size for i in range(num_bounds)]
    frame_bound_ends[-1] = num_frames
    cur_idx = 0
    for end in frame_bound_ends:
        bound_mins.append(np.min(features[cur_idx:end], axis=0, keepdims=True)) # [1, num_features]
        bound_maxs.append(np.max(features[cur_idx:end], axis=0, keepdims=True))
    
    np_bound_mins = np.concatenate(bound_mins, axis=0) # [num_bounds, num_features]
    np_bound_maxs = np.concatenate(bound_maxs, axis=0)
    return np_bound_mins, np_bound_maxs

def create_matching_database(
    db: Database,
    method: str,
    w_pos_foot: float,
    w_vel_foot: float,
    w_vel_hips: float,
    w_traj_pos: float,
    w_traj_dir: float,
    ignore_end: int,
    dense_bound_size: int = None, 
    sparse_bound_size: int = None,
) -> MatchingDatabase:
    """Create a simple `MatchingDatabase`."""
    fnorms, means, stds, indices = create_matching_features(
        db, w_pos_foot, w_vel_foot, w_vel_hips, w_traj_pos, w_traj_dir, ignore_end
    )
    if method == "aabb":
        dense_bound_mins, dense_bound_maxs = create_matching_aabb(fnorms, dense_bound_size)
        sparse_bound_mins, sparse_bound_maxs = create_matching_aabb(fnorms, sparse_bound_size)
        return MatchingDatabase(
            fnorms, means, stds, indices,
            dense_bound_size, dense_bound_mins, dense_bound_maxs,
            sparse_bound_size, sparse_bound_mins, sparse_bound_maxs,
        )
    
    elif method == "kd-tree":
        kdtree = KDTree(fnorms)
        return MatchingDatabase(fnorms, means, stds, indices, kdtree=kdtree)
    
    elif method in ["brute-force", "faiss"]: 
        return MatchingDatabase(fnorms, means, stds, indices)
    
    else: 
        raise ValueError("{} is not in methods.".format(method))

# def create_cost_matrix(
#     features: np.ndarray, # [num_frames, num_features]
#     method: str="taichi",
# ):
#     num_frames, num_features = features.shape
#     if method == "taichi":
#         ti_features = ti.Vector.field(n=num_features, dtype=float, shape=(num_frames))
#         ti_features.from_numpy(features)
#         cost_mat = ti.field(float, shape=(num_frames, num_frames))
#         cost_mat.fill(0)
#         _create_cost_matrix(ti_features, cost_mat, num_features)
#         return cost_mat.to_numpy()

# @ti.kernel
# def _create_cost_matrix(
#     features: ti.template(),
#     cost_mat: ti.template(),
#     num_features: int
# ):
#     for i, j in cost_mat:
#         for k in ti.static(range(num_features)):
#             cost_mat[i, j] += (features[i][k] - features[j][k]) ** 2

def motion_matching_search(
    cur_idx: int,
    method: str, # ["brute-force", "aabb", "kd-tree", "faiss"]
    mdb: MatchingDatabase,
    query: np.ndarray, # [num_features]
    norm_query: bool=True # If we need normalize query.
):
    """Search best index in `MatchingDatabase`.
       This function is called when 
       * motion can be updated (every N frames),
       * motion should be updated (when the animation ends, `cur_idx`==-1),
       etc.
    """
    if norm_query:
        qnorm = normalize_query(query, mdb.means, mdb.stds) # [num_features]
    else: 
        qnorm = query

    if not cur_idx == -1: # can be updated
        cur_idx = mdb.indices.index(cur_idx) # db index -> mdb index
        # mdb.features[cur_idx] will be replaced to controlled character's features. 
        best_cost: float = np.sum(np.square(mdb.features[cur_idx] - qnorm))
    else: best_cost: float = sys.float_info.max # should be updated
    
    if method == "brute-force":
        return brute_force_search(mdb, qnorm, cur_idx, best_cost)
    elif method == "aabb":
        return aabb_search(mdb, qnorm, cur_idx, best_cost)
    elif method == "kd-tree":
        return kd_search(mdb, qnorm, cur_idx, best_cost)
    elif method == "faiss":
        return faiss_search(mdb, qnorm, cur_idx, best_cost)

def brute_force_search(
    mdb: MatchingDatabase,
    qnorm: np.ndarray, # [num_features]
    cur_idx: int,
    best_cost: float,
) -> int:
    costs = np.sum(np.square(mdb.features - qnorm[None].repeat(len(mdb), axis=0)), axis=-1)
    cur_cost = np.min(costs)
    if cur_cost < best_cost:
        return mdb.indices[np.argmin(costs)] # mdb index -> db index
    else:
        return mdb.indices[cur_idx]

def aabb_search(
    mdb: MatchingDatabase,
    qnorm: np.ndarray, # [num_features]
    cur_idx: int,
    best_cost: float,
) -> int:
    i = 0
    best_idx = cur_idx
    # for all sparse boxes
    while i < len(mdb):
        sparse_box_idx = i // mdb.sparse_bound_size # box index
        i_sparse_next = (sparse_box_idx + 1) * mdb.sparse_bound_size # frame index on next sparce box
        # Calculate distance between query to sparse bounding box.
        cur_cost, smaller = calc_box_distance(
            best_cost, qnorm, 
            mdb.dense_bound_mins[sparse_box_idx], 
            mdb.sparse_bound_maxs[sparse_box_idx],
        )
        
        if not smaller:
            i = i_sparse_next
            continue
        
        # for all dense boxes
        while i < len(mdb) and i < i_sparse_next:
            dense_box_idx = i // mdb.dense_bound_size # box index
            i_dense_next = (dense_box_idx + 1) * mdb.dense_bound_size # frame index

            # Calculate distance to a dense bounding box.
            cur_cost, smaller = calc_box_distance(
                best_cost, qnorm, 
                mdb.sparse_bound_mins[sparse_box_idx], 
                mdb.sparse_bound_maxs[sparse_box_idx]
            )

            if not smaller:
                i = i_dense_next
                continue

            # Calculate distance to a feature inside box.
            while i < len(mdb) and i < i_dense_next:
                cur_cost = np.sum(np.square(qnorm - mdb.features[i]))
                if cur_cost < best_cost:
                    best_idx = i
                    best_cost = cur_cost
                
                i += 1
    return mdb.indices[best_idx] # mdb index -> db index

def kd_search(
    mdb: MatchingDatabase,
    qnorm: np.ndarray,
    cur_idx: int,
    best_cost: float,
) -> int:
    dist, index = mdb.kdtree.query(qnorm, k=1)
    if dist < best_cost:
        return mdb.indices[index] # mdb index -> db index
    else:
        return mdb.indices[cur_idx]

def faiss_search(
    mdb: MatchingDatabase,
    qnorm: np.ndarray,
    cur_idx: int,
    best_cost: float,
) -> int:
    return
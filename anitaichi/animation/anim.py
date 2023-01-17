import numpy as np
import taichi as ti
from anitaichi.animation.skel import Skel
from anitaichi.transform import ti_quat, quat

def forward_kinematics_rotations(
    parents: list[int],  # (J)
    rots: np.ndarray,    # (T, J, 4) or (J, 4)
):
    grots = []
    for i, pa in enumerate(parents):
        if pa == -1:
            grots.append(rots[...,i:i+1,:])
        else:
            grots.append(quat.mul(grots[pa], rots[...,i:i+1,:]))
    return np.concatenate(grots, axis=-2)

# Forward Kinematics using numpy.
# (Since it is difficult to parallelize each bone, use this for Pose FK.)
def forward_kinematics(
    offsets: np.ndarray, # (J, 3)
    parents: list[int],  # (J)
    grots: np.ndarray,   # (T, J, 4) or (J, 4)
    trans: np.ndarray    # (T, 3) or (3)
) -> np.ndarray:
    gposs = []
    offsets = offsets[None].repeat(len(grots), axis=0)
    for i, pa in enumerate(parents):
        if pa == -1:
            gposs.append(trans[..., None, :])
        else:
            gposs.append(quat.mul_vec(grots[..., pa:pa+1, :], offsets[..., i:i+1, :]) + gposs[pa])
    return np.concatenate(gposs, axis=-2)

# Forward kinematics using taichi.
# (Parallelization in the time direction.)
@ti.kernel
def ti_forward_kinematics(
    parents: ti.template(), # (J)
    offsets: ti.template(), # (J) * 3
    lrots: ti.template(),   # (T, J) * 4
    grots: ti.template(),   # (T, J) * 4
    gposs: ti.template(),   # (T, J) * 3
):
    # parallelize time dimention.
    for i in ti.ndrange(grots.shape[0]):
        for j in ti.ndrange(grots.shape[1]):
            if parents[j] == -1:
                grots[i, j] = lrots[i, j]
                gposs[i, j] = offsets[j]
            else:
                grots[i, j] = ti_quat.mul(grots[i, parents[j]], lrots[i, j])
                gposs[i, j] = ti_quat.mul_vec3(grots[i, parents[j]], offsets[j]) + gposs[i, parents[j]]

# @ti.kernel
# def calc_local_positions(
#     parents: ti.template(), # (J)
#     offsets: ti.template(), # (J) * 3
#     grots: ti.template(),   # (T, J) * 4
#     lposs: ti.template()    # (T, J) * 3
# ):
#     for i, j in grots:
#         if parents[j] == -1:
#             lposs[i, j] = offsets[j]
#         else:
#             lposs[i, j] = ti_quat.mul_vec3(grots[i, j], offsets[j])

# Convert numpy array to taichi vector field.
def convert_to_vector_field(array: np.ndarray):
    shape, dim_vector = array.shape[:-1], array.shape[-1]
    field = ti.Vector.field(dim_vector, dtype=float, shape=shape)
    field.from_numpy(array)
    return field


class Pose:
    def __init__(
        self,
        skel: Skel,
        rots: np.ndarray,
        root_pos: np.ndarray,
    ) -> None:
        self.skel = skel
        self.rots = rots
        self.root_pos = root_pos
    
    @property
    def local_rotations(self):
        return self.rots
    
    @property
    def global_rotations(self):
        parents = self.skel.parents
        lrots = self.rots
        grots = forward_kinematics_rotations(parents, lrots)
        return grots
    
    @property
    def global_positions(self):
        offsets = self.skel.offsets
        parents = self.skel.parents
        grots = self.global_rotations
        trans = self.root_pos
        gposs = forward_kinematics(offsets, parents, grots, trans)
        return gposs


class Animation:
    def __init__(
        self,
        skel: Skel,
        rots: np.ndarray,
        trans: np.ndarray,
        fps: int,
        anim_name: str,
    ) -> None:
        self.skel = skel
        self.rots = rots
        self.trans = trans

        self.fps = fps
        self.anim_name = anim_name

    def __len__(self):
        return len(self.rots)

    @property
    def local_rotations(self):
        return self.rots

    @property
    def global_rotations(self):
        parents = self.skel.parents
        lrots = self.rots
        grots = forward_kinematics_rotations(parents, lrots)
        return grots
    
    # @property
    # def local_positions_field(self):
    #     parents = self.skel.parents_field
    #     offsets = self.skel.offsets_field
    #     grots = self.global_rotations
    #     lposs = ti.Vector.field(3, dtype=float, shape=grots.shape)
    #     calc_local_positions(parents, offsets, grots, lposs)
    #     return lposs

    @property
    def global_positions(self):
        offsets = self.skel.offsets
        parents = self.skel.parents
        grots = self.global_rotations
        trans = self.trans
        gposs = forward_kinematics(offsets, parents, grots, trans)
        return gposs
    
    def ti_fk(self):
        parents = self.skel.parents_field # (J)
        offsets = self.skel.offsets_field # (J) * 3
        lrots = convert_to_vector_field(self.local_rotations)  # (T, J) * 4
        grots = ti.Vector.field(4, dtype=float, shape=lrots.shape) # (T, J) * 4
        gposs = ti.Vector.field(3, dtype=float, shape=lrots.shape) # (T, J) * 3
        ti_forward_kinematics(parents, offsets, lrots, grots, gposs)
        self.global_rotations_field = grots
        self.global_positions_field = gposs
    
    def to_pose(self, frame_num: int) -> Pose:
        skel = self.skel
        rots = self.rots[frame_num]
        root_pos = self.trans[frame_num]
        return Pose(skel, rots, root_pos)

    def to_poses(self, frames: slice) -> list[Pose]:
        if isinstance(frames, int):
            return self.to_pose(frames)
        skel = self.skel
        anim_rots = self.rots[frames]
        anim_trans = self.trans[frames]
        poses = []
        for rots, root_pos in zip(anim_rots, anim_trans):
            poses.append(Pose(skel, rots, root_pos))
        return poses

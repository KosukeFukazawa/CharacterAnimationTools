"""
bvh.py
"""
# loading and writing a biovision hierarchy data (BVH).

from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
import logging
import numpy as np

from anitaichi.animation.anim import Animation
from anitaichi.animation.skel import Skel, Joint
from anitaichi.transform import quat

def load(
    filepath: Path | str, 
    start: int=None, 
    end: int=None, 
    order: str=None,
    load_skel: bool=True,
    load_pose: bool=True,
    skel: Skel=None,
    skel_name: str=None,
) -> Skel | Animation:
    
    if not load_skel and not load_pose:
        logging.info("Either load_skel or load_pose must be specified.")
        raise ValueError
    
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    # List bvh file by each row (line) and each word (column).
    with open(filepath, "r") as f:
        lines: list[str] = [line.strip() for line in f if line != ""]
        motion_idx: int = lines.index("MOTION")
        lines: list[str | list[str]] = \
            list(map(lambda x: x.split(), lines))
        f.close()
    
    # Load HIERARCHY term.
    if load_skel:
        skel, order = load_hierarchy(
            lines = lines[:motion_idx], 
            skel_name=skel_name,
        )

    # Load MOTION term.
    if load_pose:
        assert not skel is None, "You need to load skeleton or define skeleton." 
        name = filepath.name.split(".")[0]
        
        fps, trans, quats = load_motion(
            lines = lines[motion_idx:],
            start = start,
            end = end,
            order = order,
            skel = skel
        )
        return Animation(
            skel=skel,
            rots=quats,
            trans=trans,
            fps=fps,
            anim_name=name,
        )
    
    return skel

def load_hierarchy(
    lines: list[str | list[str]],
    skel_name: str,
) -> Skel:
    
    channelmap: dict[str, str] = {
        "Xrotation" : "x",
        "Yrotation" : "y",
        "Zrotation" : "z",   
    }
    
    stacks: list[int] = [-1]
    parents: list[int] = []
    name_list: list[str] = []
    idx = 0
    joints: list[Joint] = []
    depth:int = 0
    end_site: bool = False
    
    for line in lines:
        if "ROOT" in line or "JOINT" in line:
            parents.append(stacks[-1])
            stacks.append(len(parents) - 1)
            name_list.append(line[1])
            
        elif "OFFSET" in line:
            if not end_site:
                offset = list(map(float, line[1:]))
                joints.append(Joint(
                    name = name_list[-1],
                    index = idx,
                    parent = parents[-1],
                    offset = offset,
                    ))
                idx += 1
        
        elif "CHANNELS" in line:
            dof = int(line[1])
            if dof == 3:
                order = "".join([channelmap[p] for p in line[2:2+3]])
            elif dof == 6:
                order = "".join([channelmap[p] for p in line[5:5+3]])
            joints[-1].dof = dof
        
        elif "End" in line:
            end_site = True
            stacks.append(len(parents) - 1)
        
        elif "{" in line:
            depth += 1
        
        elif "}" in line:
            depth -= 1
            end_site = False
            stacks.pop()
    
    assert depth == 0, "Brackets are not closed."
    
    skel = Skel(joints=joints, skel_name=skel_name)
    return skel, order

def load_motion(
    lines: list[list[str]],
    order: str,
    skel: Skel,
    start: int=None,
    end: int=None,
) -> tuple[int, np.ndarray, np.ndarray]:
    
    fps: int = round(1 / float(lines[2][2]))
    lines: list[list[str]] = lines[3:]
    
    np_lines = np.array(
        list(map(lambda line: list(map(float, line)), lines))
    ) # T × dim_J(J × 3 + 3 or J × 6) matrix.
    np_lines = np_lines[start:end]
    
    dofs = []
    eangles = []
    poss = []
    ckpt = 0
    for joint in skel.joints:
        dof = joint.dof
        dofs.append(dof)
        if dof == 3:
            eangle = np_lines[:, ckpt:ckpt+3]
            eangles.append(eangle[:, None])
            ckpt += 3
        elif dof == 6:
            pos = np_lines[:, ckpt:ckpt+3] # [T, 3]
            eangle = np_lines[:, ckpt+3:ckpt+6]
            poss.append(pos[:, None]) # [T, 1, 3]
            eangles.append(eangle[:, None]) # [T, 1, 3]
            ckpt += 6
    
    assert sum(dofs) == np_lines.shape[1], \
        "Skel and Motion are not compatible."
    
    poss = np.concatenate(poss, axis=1)
    eangles = np.concatenate(eangles, axis=1)
    
    trans = poss[:, 0]
    quats = quat.unroll(quat.from_euler(eangles, order))
    
    return fps, trans, quats

def save(
    filepath: Path | str,
    anim: Animation, 
    order: str="zyx",
    ) -> bool:
    
    skel = anim.skel
    trans = anim.trans.to_numpy()
    quats = anim.rots.to_numpy()
    fps = anim.fps
    
    with open(filepath, "w") as f:
        # write hierarchy data.
        f.write("HIERARCHY\n")
        index_order = save_hierarchy(f, skel, 0, order, 0)
        
        # write motion data.
        f.write("MOTION\n")
        f.write("Frames: %d\n" % len(trans))
        f.write("Frame Time: %f\n" % (1.0 / fps))
        save_motion(f, trans, quats, order, index_order)
        f.close()

def save_hierarchy(
    f: TextIOWrapper,
    skel: Skel,
    index: int,
    order: str,
    depth: int,
) -> list[int]:

    def order2xyzrotation(order: str) -> str:
        cmap: dict[str, str] = {
            "x" : "Xrotation",
            "y" : "Yrotation",
            "z" : "Zrotation",   
        }
        return "%s %s %s" % (cmap[order[0]], cmap[order[1]], cmap[order[2]])
    
    joint = skel[index]
    index_order = [index]
    if joint.root:
        f.write("\t" * depth + "ROOT %s\n" % joint.name)
    else:
        f.write("\t" * depth + "JOINT %s\n" % joint.name)
    f.write("\t" * depth + "{\n")
    depth += 1
    offset = joint.offset
    f.write("\t" * depth + \
        "OFFSET %f %f %f\n" % (offset[0], offset[1], offset[2]))
    
    if joint.root:
        f.write("\t" * depth + \
            "CHANNELS 6 Xposition Yposition Zposition %s\n"\
            % order2xyzrotation(order))
    else:
        f.write("\t" * depth + "CHANNELS 3 %s\n"\
            % order2xyzrotation(order))
    
    children_idxs = skel.get_children(index, return_idx=True)
    for child_idx in children_idxs:
        ch_index_order = save_hierarchy(f, skel, child_idx, order, depth)
        index_order.extend(ch_index_order)
    if children_idxs == []:
        f.write("\t" * depth + "End Site\n")
        f.write("\t" * depth + "{\n")
        f.write("\t" * (depth + 1) + "OFFSET %f %f %f\n" \
            % (0, 0, 0))
        f.write("\t" * depth + "}\n")
    
    depth -= 1
    f.write("\t" * depth + "}\n")
    return index_order

def save_motion(
    f: TextIOWrapper,
    trans: np.ndarray,
    quats: np.ndarray,
    order: str,
    index_order: list[int],
) -> None:
    
    def write_position_rotation(
        pos: np.ndarray, 
        rot: np.ndarray,
        ) -> str:
        pos, rot = pos.tolist(), rot.tolist()
        return "%f %f %f %f %f %f " \
            % (pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])

    def write_rotation(rot: np.ndarray) -> str:
        rot = rot.tolist()
        return "%f %f %f " %(rot[0], rot[1], rot[2])
    
    eangles = np.rad2deg(quat.to_euler(quats, order)) # (T, J, 3)
    for i in range(len(trans)):
        for j in index_order:
            if j == 0:
                f.write(
                    "%s" % write_position_rotation(trans[i], eangles[i, j])
                    )
            else:
                f.write(
                    "%s" % write_rotation(eangles[i, j])
                )
        f.write("\n")
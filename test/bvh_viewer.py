from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import taichi as ti

BASEPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASEPATH))

from anitaichi.animation.anim_loader import bvh

@ti.kernel
def get_frame(
    gposs: ti.types.ndarray(),
    cur_pose: ti.template(),
    frame: int
):
    for i in cur_pose:
        cur_pose[i] = gposs[frame, i] / 100

def update_camera(camera, anim, frame):
    root_pos = anim.trans[frame] / 100
    camera_pos = root_pos + np.array([0, 0, -10])

    camera.position(*camera_pos)
    camera.lookat(*root_pos)
    scene.set_camera(camera)

if __name__ == "__main__":
    # Starting taichi on CUDA.
    ti.init(arch=ti.cuda)
    
    window = ti.ui.Window("bvh viewer", (1024, 1024), vsync=True)
    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    bvh_file = BASEPATH / "data/aiming1_subject1.bvh"
    anim = bvh.load(bvh_file)
    anim.ti_fk()
    cur_pose = ti.Vector.field(3, dtype=float, shape=len(anim.skel))
    frame = 0

    while window.running:
        get_frame(anim.global_positions_field, cur_pose, frame)
        scene.particles(cur_pose, radius=0.05, color=(1.0, 0.0, 0.0))
        scene.ambient_light((0.5, 0.5, 0.5))
        update_camera(camera, anim, frame)

        frame += 1
        if frame == len(anim):
            frame = 0
        
        canvas.scene(scene)
        window.show()
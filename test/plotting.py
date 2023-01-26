from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk

BASEPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASEPATH))

from anim import bvh
from anim.motion_matching.database import Database
from anim.motion_matching.mm import create_matching_database

if __name__ == "__main__":
    bvh_folder = BASEPATH / "data"
    bvh_file = "sim_motion.bvh"
    start = 90
    # end = 7086
    end = 2090
    save_fig = True
    
    anim = bvh.load(bvh_folder / bvh_file, start, end)
    db = Database([anim])
    mdb = create_matching_database(
        db=db, method="brute-force",
        w_pos_foot=0.75, w_vel_foot=1, w_vel_hips=1, w_traj_pos=1, w_traj_dir=1.5,
        ignore_end=20, dense_bound_size=16, sparse_bound_size=64,
    )
    bar_lim = np.max(np.abs(mdb.features))
    gposs = db.gpos / 100
    cposs = db.cpos / 100
    parents = db.parents
    
    proj_root_poss = db.proj_root_pos(remove_vertical=True)
    root_dirs = db.root_direction(remove_vertical=True)
    future_20_gposs = db.future_traj_poss(20, cspace=False)
    future_20_gdirs = db.future_traj_dirs(20, cspace=False)
    future_20_cposs = db.future_traj_poss(20, cspace=True)
    future_20_cdirs = db.future_traj_dirs(20, cspace=True)
    
    fig = plt.figure(figsize=(7, 7))
    axg = fig.add_subplot(221, projection="3d")
    axc = fig.add_subplot(223, projection="3d")
    axf = fig.add_subplot(122)
    
    def update(index: int):
        gposes = gposs[index]
        cposes = cposs[index]
        features = mdb.features[index]
        
        proj_root_pos = proj_root_poss[index] / 100
        root_dir = root_dirs[index]
        
        future_20_gpos = future_20_gposs[index] / 100
        future_20_gdir = future_20_gdirs[index]
        future_20_cpos = future_20_cposs[index] / 100
        future_20_cdir = future_20_cdirs[index]
        
        # global space viewer settings
        axg.cla()
        axg.set_title("global")
        axg.grid(axis="y")
        axg.set_yticklabels([])
        axg.yaxis.pane.set_facecolor("gray")
        axg.yaxis.set_major_locator(ticker.NullLocator())
        axg.yaxis.set_minor_locator(ticker.NullLocator())
        axg.view_init(elev=120, azim=-90)
        axg.set_xlim3d(proj_root_pos[0]-3, proj_root_pos[0]+3)
        axg.set_ylim3d(0, 3)
        axg.set_zlim3d(proj_root_pos[1]-3, proj_root_pos[1]+3)
        axg.set_box_aspect((1, 0.5, 1))
        axg.scatter(gposes[:, 0], gposes[:, 1], gposes[:, 2], c="red", s=3, label="joints")
        for i, parent in enumerate(parents):
            if parent != -1:
                axg.plot(
                    [gposes[parent,0], gposes[i,0]],
                    [gposes[parent,1], gposes[i,1]],
                    [gposes[parent,2], gposes[i,2]], 
                    c="red", alpha=0.8
                )
        axg.plot(
            [proj_root_pos[0], proj_root_pos[0]+root_dir[0]],
            [0, 0],
            [proj_root_pos[1], proj_root_pos[1]+root_dir[1]],
            label="root direction", c="blue"
        )
        axg.plot(
            [future_20_gpos[0], future_20_gpos[0]+future_20_gdir[0]],
            [0, 0],
            [future_20_gpos[1], future_20_gpos[1]+future_20_gdir[1]],
            label="future direction 20", c="green"
        )
        axg.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=0)
        
        # character space viewer settings
        axc.cla()
        axc.set_title("character")
        axc.grid(axis="y")
        axc.set_yticklabels([])
        axc.yaxis.pane.set_facecolor("gray")
        axc.yaxis.set_major_locator(ticker.NullLocator())
        axc.yaxis.set_minor_locator(ticker.NullLocator())
        axc.view_init(elev=120, azim=-90)
        axc.set_xlim3d(-3, 3)
        axc.set_ylim3d(0, 3)
        axc.set_zlim3d(-3, 3)
        axc.set_box_aspect((1,0.5,1))
        axc.scatter(cposes[:, 0], cposes[:, 1], cposes[:, 2], c="red", s=3, label="joints")
        for i, parent in enumerate(parents):
            if parent != -1:
                axc.plot(
                    [cposes[parent,0], cposes[i,0]],
                    [cposes[parent,1], cposes[i,1]],
                    [cposes[parent,2], cposes[i,2]], 
                    c="red", alpha=0.8
                )
        axc.plot([0, 0], [0, 0], [0, 1], label="root direction", c="blue")
        axc.plot(
            [future_20_cpos[0], future_20_cpos[0]+future_20_cdir[0]],
            [0, 0],
            [future_20_cpos[1], future_20_cpos[1]+future_20_cdir[1]],
            label="future direction 20", c="green"
        )
        
        # features bar settings
        axf.cla()
        axf.set_title("matching features")
        axf.grid(axis="x")
        axf.set_xlim(-bar_lim, bar_lim)
        axf.barh(np.arange(len(features)), features)
        
        fig.suptitle("frame: {}".format(index+1))
        sys.stdout.write("processed frame: {}\n".format(index+1))
        
    ani = FuncAnimation(fig, update, frames=np.arange(len(db)), interval=1000/anim.fps, repeat=False)
    
    if save_fig:
        fig_path = (BASEPATH / "figs" / bvh_file).with_suffix(".gif")
        ani.save(fig_path)
        sys.stdout.write("Saved figure as " + str(fig_path) + "\n")
    else:
        plt.show()
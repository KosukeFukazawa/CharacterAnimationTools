from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BASEPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASEPATH))

from anim import bvh

if __name__ == "__main__":
    bvh_folder = BASEPATH / "data"
    bvh_file = "sim_motion.bvh"
    start, end = None, None
    elev, azim = 120, -90
    save_fig = False
    
    anim = bvh.load(bvh_folder / bvh_file, start, end)
    poss = anim.gpos / 100
    parents = anim.parents
    
    proj_root_poss = anim.proj_root_pos(remove_vertical=True)
    root_dirs = anim.root_direction(remove_vertical=True)
    future_20_poss = anim.future_traj_poss(20, cspace=False)
    future_20_dirs = anim.future_traj_dirs(20, cspace=False)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    def update(index: int):
        poses = poss[index]
        
        # root direction
        proj_root_pos = proj_root_poss[index] / 100
        root_dir = root_dirs[index]
        
        # future root direction
        future_20_pos = future_20_poss[index] / 100
        future_20_dir = future_20_dirs[index]
        
        # plot settings
        ax.cla()
        ax.set_title("global") 
        ax.grid(axis="y")
        ax.set_yticklabels([])
        ax.yaxis.pane.set_facecolor("gray")
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d(proj_root_pos[0]-3, proj_root_pos[0]+3)
        ax.set_ylim3d(0, 3)
        ax.set_zlim3d(proj_root_pos[1]-3, proj_root_pos[1]+3)
        ax.set_box_aspect((2, 1, 2))
        
        ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c="red", label="joints")
        for i, parent in enumerate(parents):
            if parent != -1:
                ax.plot(
                    [poses[parent,0], poses[i,0]],
                    [poses[parent,1], poses[i,1]],
                    [poses[parent,2], poses[i,2]], 
                    c="red", alpha=0.8
                )
        ax.plot(
            [proj_root_pos[0], proj_root_pos[0]+root_dir[0]],
            [0, 0],
            [proj_root_pos[1], proj_root_pos[1]+root_dir[1]],
            label="root direction", c="blue"
        )
        ax.plot(
            [future_20_pos[0], future_20_pos[0]+future_20_dir[0]],
            [0, 0],
            [future_20_pos[1], future_20_pos[1]+future_20_dir[1]],
            label="20 future direction", c="green"
        )
        ax.legend()
        fig.suptitle("frame: {}".format(index+1))
        sys.stdout.write("Processed frame: {}\n".format(index+1))
    ani = FuncAnimation(fig, update, frames=np.arange(len(anim)), interval=1000/anim.fps, repeat=True)
    
    if save_fig:
        fig_path = (BASEPATH / "figs" / bvh_file).with_suffix(".gif")
        ani.save(fig_path)
        sys.stdout.write("Saved figure as " + str(fig_path) + "\n")
    else:
        plt.show()
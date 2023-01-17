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
    poss = anim.cpos / 100
    parents = anim.parents
    
    future_20_poss = anim.future_traj_poss(20, cspace=True)
    future_20_dirs = anim.future_traj_dirs(20, cspace=True)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    def update(index: int):
        poses = poss[index]
        
        future_20_pos = future_20_poss[index] / 100
        future_20_dir = future_20_dirs[index]
        
        ax.cla()
        ax.set_title("character") 
        ax.grid(axis="y")
        ax.set_yticklabels([])
        ax.yaxis.pane.set_facecolor("gray")
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(0, 3)
        ax.set_zlim3d(-3, 3)
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
            [future_20_pos[0], future_20_pos[0]+future_20_dir[0]],
            [0, 0],
            [future_20_pos[1], future_20_pos[1]+future_20_dir[1]],
            label="20 future direction", c="green"
        )
        ax.legend()
        fig.suptitle("frame: {}".format(index+1))
        sys.stdout.write("processed frame: {}\n".format(index+1))
    ani = FuncAnimation(fig, update, frames=np.arange(len(anim)), interval=1000/anim.fps, repeat=True)
    
    if save_fig:
        fig_path = (BASEPATH / "figs" / bvh_file).with_suffix(".gif")
        ani.save(fig_path)
    else:
        plt.show()
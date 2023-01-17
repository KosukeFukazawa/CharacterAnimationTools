from __future__ import annotations

from pathlib import Path
import pickle

from anim import bvh
from anim.animation import Animation
from anim.motion_matching.mm import create_matching_features

def preprocess_motion_data(BASEPATH: Path, cfg, save_dir: Path):
    dataset_dir = BASEPATH / cfg.dir
    anims = []
    for file, start, end in zip(cfg.files, cfg.starts, cfg.ends):
        anims.append(bvh.load(dataset_dir / file, start, end))
    
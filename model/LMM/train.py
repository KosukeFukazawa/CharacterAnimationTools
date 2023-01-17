"""
train.py
You can define your configulation file like below.
```
python train.py configs/DeepLearning/LMM.toml
```
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import jit, partial, random
from flax import linen as nn
from flax.training import train_state
import optax

BASEPATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASEPATH))

from model.LMM.setting import Config
from model.LMM.decompressor import Compressor, Decompressor
from model.LMM.stepper import Stepper
from model.LMM.projector import Projector

def main(cfg: Config):
    ckpt_dir = cfg.ckpt_dir
    
    match cfg.setting.train_model:
        case "all":
            sys.stdout.write("Train models: decompressor, stepper, projector\n")
            train_decompressor(cfg.setting.decompressor, ckpt_dir) 
            train_stepper(cfg.setting.stepper, ckpt_dir)
            train_projector(cfg.setting.projector, ckpt_dir)
        case "decompressor":
            sys.stdout.write("Train model: decompressor\n")
            train_decompressor(cfg.setting.decompressor, ckpt_dir)
        case "stepper":
            sys.stdout.write("Train model: stepper\n")
            train_stepper(cfg.setting.stepper, ckpt_dir)
        case "projector":
            sys.stdout.write("Train model: projector\n")
            train_projector(cfg.setting.projector, ckpt_dir)
        case _:
            raise ValueError("Invalid value. Please check LMM.toml.")

def train_decompressor(cfg, ckpt_dir):
    pass

def train_stepper(cfg, ckpt_dir):
    pass

def train_projector(cfg, ckpt_dir):
    pass

if __name__ == "__main__":
    setting_path = sys.argv[1]
    cfg = Config(BASEPATH, setting_path)
    
    main(cfg)
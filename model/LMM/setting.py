"""
setting.py
Device setting etc..
"""
from __future__ import annotations

import sys
from pathlib import Path
import shutil

from util.load import pickle_load, toml_load
from model.LMM.preprocessing import preprocess_motion_data

class Config:
    def __init__(self, BASEPATH: Path, setting_path: str) -> None:
        # Load a setting file.
        try:
            if (BASEPATH / setting_path).exists():
                setting_path = BASEPATH / setting_path
                setting = toml_load(setting_path)
            else:
                setting = toml_load(setting_path)
        except:
            setting_path = BASEPATH / "configs/DeepLearning/LMM.toml"
            setting = toml_load(setting_path)
        sys.stdout.write(f"Config file: {str(setting_path)}\n")
        
        self.setting = setting
        self.BASEPATH = BASEPATH
        self.seed = setting.seed
        
        # Device settings (gpu or cpu)
        self.device = setting.device
        # GPU settings.
        if self.device == "gpu" or "cuda":
            if isinstance(setting.gpus, list):
                # if you use multi gpus
                if len(setting.gpus) > 1:
                    self.gpus = setting.gpus
                    self.multi_gpus_setting()
                else:
                # if you use single gpu
                    self.gpus = setting.gpus[0]
                    self.single_gpu_setting()
            else:
                self.gpus = setting.gpus
                self.single_gpu_setting()
        
        # Dataset settings.
        # if you need preprocess
        if setting.need_preprocess:
            save_dir = BASEPATH / setting.processed_dir
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
                sys.stdout.write(f"Create preprocessed folder at: {str(save_dir)}\n")
            self.dataset = preprocess_motion_data(BASEPATH, setting.dataset, save_dir / setting.processed_file_name)
        else:
            self.dataset = pickle_load(save_dir / setting.processed_file_name)
        
        # Save experiment settings.
        ckpt_dir = BASEPATH / setting.checkpoint_dir / setting.exp_name
        # TBD: if exists, load checkpoints and resume the experiment.
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True)
            sys.stdout.write(f"Create checkpoint folder at: {str(ckpt_dir)}\n")
        self.ckpt_dir = ckpt_dir
            
        save_cfg_dir = BASEPATH / setting.save_config_dir
        save_cfg_path = save_cfg_dir / f"{setting.exp_name}.toml"
        if not save_cfg_dir.exists():
            save_cfg_dir.mkdir(parents=True)
            sys.stdout.write(f"Create config folder at: {str(save_cfg_dir)}\n")
        shutil.copy(str(setting_path), str(save_cfg_path))

    def single_gpu_setting(self):
        pass
    
    def multi_gpus_setting(self):
        pass
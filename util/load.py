"""
load.py
"""
from __future__ import annotations

from pathlib import Path
import toml, yaml, pickle, json
from easydict import EasyDict

def toml_load(path: Path):
    with open(path, "r") as f:
        cfg = toml.load(f)
    cfg = EasyDict(cfg)
    return cfg

def yaml_load(path: Path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = EasyDict(cfg)
    return cfg

def pickle_load(path: Path, encoding: str="ASCII"):
    with open(path, "rb") as f:
        cfg = pickle.load(f, encoding=encoding)
    return cfg

def json_load(path: Path):
    with open(path, "r") as f:
        cfg = json.loads(f.read())
    cfg = EasyDict(cfg)
    return cfg
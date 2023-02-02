from __future__ import annotations

import numpy as np
from util import quat

def eye(shape: list[int], dtype=np.float32) -> np.ndarray:
    return np.ones(list(shape) + [8], dtype=dtype) * \
        np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)

def normalize(dq: np.ndarray, eps=1e-8) -> np.ndarray:
    mag = quat.length(dq[...,:4])
    mag = np.where(mag==0, eps, mag)
    return dq / mag[None]

def abs(dq: np.ndarray) -> np.ndarray:
    real = np.where(dq[...,0:1] > 0.0, dq[...,:4], -dq[...,:4])
    dual = np.where(dq[...,4:5] > 0.0, dq[...,4:], -dq[...,4:])
    return np.concatenate([real, dual], axis=-1)

def inv(dq: np.ndarray) -> np.ndarray:
    real = quat.inv(dq[...,:4])
    dual = -quat.mul_inv(quat.inv_mul(dq[...,:4], dq[...,4:]), dq[...,:4])
    return np.concatenate([real, dual], axis=-1)

def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    real = quat.mul(x[...,:4], y[...,:4])
    dual = quat.mul(x[...,:4], y[...,4:]) + quat.mul(x[...,4:], y[...,:4])
    return np.concatenate([real, dual], axis=-1)

def from_trans(trans: np.ndarray) -> np.ndarray:
    dual = np.zeros(trans.shape[:-1] + (4,))
    dual[...,1:] = trans * 0.5
    return np.concatenate([quat.eye(trans.shape[:,-1]), dual], axis=-1)

def from_rot(rot: np.ndarray) -> np.ndarray:
    return np.concatenate([rot, np.zeros(rot.shape[:-1] + (4,))], axis=-1)

def from_rot_and_trans(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    rot = quat.normalize(rot)
    dual = np.zeros(trans.shape[:-1] + (4,))
    dual[...,1:] = trans
    dual = quat.mul(dual, rot) * 0.5
    return np.concatenate([rot, dual], axis=-1)

def to_trans(dq: np.ndarray) -> np.ndarray:
    return 2 * quat.mul_inv(dq[...,4:], dq[...,:4])[...,1:]

def to_rot(dq: np.ndarray) -> np.ndarray:
    return dq[...,:4]

def fk(dq: np.ndarray, parents: list[int]) -> np.ndarray:
    gdq = [dq[...,:1,:]]
    for i in range(1, len(parents)):
        gdq.append(mul(gdq[parents[i]], dq[...,i:i+1,:]))
    return np.concatenate(gdq, axis=-2)
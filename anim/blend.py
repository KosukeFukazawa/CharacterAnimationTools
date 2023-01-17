from __future__ import annotations

import numpy as np

from anim.animation import Animation
from util import quat

# =====================
#  Basic interpolation
# =====================
# Linear Interpolation (LERP) of objects.
def lerp(x, y, t):
    return (1 - t) * x + t * y

# LERP for quaternions.
def quat_lerp(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    assert x.shape == y.shape, "Two quats must be the same shape."
    return quat.normalize(lerp(x, y, t))

# Spherical linear interpolation (SLERP) for quaternions.
def slerp(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    assert x.shape == y.shape, "Two quats must be the same shape."
    if t == 0:
        return x
    elif t == 1:
        return y
    if quat.dot(x, y) < 0:
        y = - y
    ca = quat.dot(x, y)
    theta = np.arccos(np.clip(ca, 0, 1))
    r = quat.normalize(y - x * ca)
    return x * np.cos(theta * t) + r * np.sin(theta * t)


# =========
#  Damping
# =========

# predict `t`
from __future__ import annotations

import numpy as np

# Calculate cross object of two 3D vectors.
def cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def dot(a, b, keepdims=False):
    return np.sum(a*b, axis=-1, keepdims=keepdims)

def length(v, keepdims=False):
    return np.linalg.norm(v, axis=-1, keepdims=keepdims)

def normalize(v):
    lengths = length(v, keepdims=True)
    lengths = np.where(lengths==0, 1e-10, lengths) # avoid 0 divide
    return v / lengths


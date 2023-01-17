import taichi as ti
import taichi.math as tm

@ti.func
def cross(v, w):
    out0 = v.y * w.z - v.z * w.y
    out1 = v.z * w.x - v.x * w.z
    out2 = v.x * w.y - v.y * w.x
    return ti.Vector([out0, out1, out2])

@ti.func
def dot(v, w):
    return ti.Vector([v.x * w.x, v.y * w.y, v.z * w.z])

@ti.func
def length(v):
    return tm.sqrt(v.x * v.x + v.y + v.y + v.z + v.z)

@ti.func
def normalize(v):
    return tm.normalize(v)
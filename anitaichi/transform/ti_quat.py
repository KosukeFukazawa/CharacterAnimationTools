import taichi as ti
import taichi.math as tm

from anitaichi.transform import ti_vec

@ti.func
def eye():
    return ti.Vector([1.0, 0.0, 0.0, 0.0])

@ti.func
def normalize(q):
    return tm.normalize(q)

@ti.func
def inv(q):
    return ti.Vector([-q[0], q[1], q[2], q[3]])

@ti.func
def abs(q):
    return -q if q[0] < 0.0 else q

@ti.func
def mul(x, y):
    qout0 = y[0] * x[0] - y[1] * x[1] - y[2] * x[2] - y[3] * x[3]
    qout1 = y[0] * x[1] + y[1] * x[0] - y[2] * x[3] + y[3] * x[2]
    qout2 = y[0] * x[2] + y[1] * x[3] + y[2] * x[0] - y[3] * x[1]
    qout3 = y[0] * x[3] - y[1] * x[2] + y[2] * x[1] + y[3] * x[0]
    return ti.Vector([qout0, qout1, qout2, qout3])

@ti.func
def mul_vec3(q, v):
    v_quat = ti.Vector([q[1], q[2], q[3]])
    t = 2.0 * ti_vec.cross(v_quat, v)
    return v + q[0] * t + ti_vec.cross(v_quat, t)

@ti.func
def from_angle_and_axis(angle, axis):
    c = tm.cos(angle / 2.0)
    s = tm.sin(angle / 2.0)
    return ti.Vector([c, s * axis.x, s * axis.y, s * axis.z])

@ti.func
def from_euler(euler, order):
    axis = {
        "x": ti.Vector([1.0, 0.0, 0.0]),
        "y": ti.Vector([0.0, 1.0, 0.0]),
        "z": ti.Vector([0.0, 0.0, 1.0])
    }
    q0 = from_angle_and_axis(euler[0], axis[order[0]])
    q1 = from_angle_and_axis(euler[1], axis[order[1]])
    q2 = from_angle_and_axis(euler[2], axis[order[2]])

    return mul(q0, mul(q1, q2))

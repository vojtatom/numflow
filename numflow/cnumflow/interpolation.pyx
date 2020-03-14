# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language=c++
from .types cimport DTYPE
from .decl cimport Dataset3D, interpolate_3d
from .common cimport create_2d_double_numpy
cimport numpy as np
import numpy as np


def interpolate3D(DTYPE[:,:,:,::1] values, DTYPE[::1] x, DTYPE[::1] y, DTYPE[::1] z, DTYPE[:,::1] points):
    cdef Dataset3D dataset
    dataset.dx = x.size
    dataset.dy = y.size
    dataset.dz = z.size
    dataset.ax = &x[0]
    dataset.ay = &y[0]
    dataset.az = &z[0]
    dataset.data = &values[0, 0, 0, 0]

    cdef DTYPE * vals = interpolate_3d(&dataset, &points[0, 0], points.shape[0])
    arr = create_2d_double_numpy(vals, points.shape[0], 3)
    return arr





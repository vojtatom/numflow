# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language=c++
from .types cimport DTYPE
from .decl cimport dataset3D, interpolate_3d
from .common cimport pointer_to_two_d_numpy_array
cimport numpy as np
import numpy as np

def interpolate3D(DTYPE[:,::1] points, DTYPE[:,:,:,::1] values, DTYPE[::1] x, DTYPE[::1] y, DTYPE[::1] z):
    cdef dataset3D dataset
    dataset.dx = x.size
    dataset.dy = y.size
    dataset.dz = z.size
    dataset.ax = &x[0]
    dataset.ay = &y[0]
    dataset.az = &z[0]
    dataset.data = &values[0, 0, 0, 0]
    
    cdef DTYPE * vals = interpolate_3d(&dataset, &points[0, 0], points.shape[0])

    cdef np.npy_intp dims[2]
    dims[0] = points.shape[0]
    dims[1] = 3
    arr = pointer_to_two_d_numpy_array(vals, dims)
    return arr





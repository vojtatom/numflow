# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language=c++
from .types cimport DTYPE, INTDTYPE
from .decl cimport Dataset3D, DataStreamlines, interpolate_3d, integrate_3d, delete_datastreamline
from .common cimport create_2d_double_numpy, create_1d_double_numpy, create_1d_int32_numpy
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
    cdef np.ndarray[DTYPE, ndim=2] arr = create_2d_double_numpy(vals, points.shape[0], 3)
    return arr


def integrate3D(DTYPE[:,:,:,::1] values, DTYPE[::1] x, DTYPE[::1] y, DTYPE[::1] z, DTYPE[:,::1] points, DTYPE t0, DTYPE tbound):
    cdef Dataset3D dataset
    dataset.dx = x.size
    dataset.dy = y.size
    dataset.dz = z.size
    dataset.ax = &x[0]
    dataset.ay = &y[0]
    dataset.az = &z[0]
    dataset.data = &values[0, 0, 0, 0]

    cdef DataStreamlines * streamlines = integrate_3d(&dataset, &points[0, 0], points.shape[1], t0, tbound)

    cdef np.ndarray[DTYPE, ndim=1] pos = create_1d_double_numpy(streamlines[0].y, streamlines[0].dy)
    cdef np.ndarray[DTYPE, ndim=1] vals = create_1d_double_numpy(streamlines[0].f, streamlines[0].df)
    cdef np.ndarray[DTYPE, ndim=1] ts = create_1d_double_numpy(streamlines[0].t, streamlines[0].dt)
    cdef np.ndarray[INTDTYPE, ndim=1] lens = create_1d_int32_numpy(streamlines[0].l, streamlines[0].dl)

    delete_datastreamline(streamlines)

    return pos, vals, ts, lens



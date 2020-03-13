# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from .types cimport DTYPE, INTDTYPE

cdef extern from "cpp/numflow.hpp":
    struct dataset3D:
        INTDTYPE dx
        INTDTYPE dy
        INTDTYPE dz
        DTYPE * ax
        DTYPE * ay
        DTYPE * az
        DTYPE * data
        

    void interpolate(const DTYPE * comp, INTDTYPE * ind, DTYPE * fac, INTDTYPE * grid_size, DTYPE * res)
    void dataset_sort(DTYPE * data, INTDTYPE columns, INTDTYPE rows)
    DTYPE * interpolate_3d(const dataset3D * dataset, DTYPE *points, const INTDTYPE count)


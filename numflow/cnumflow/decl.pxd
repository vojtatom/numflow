# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from .types cimport DTYPE, INTDTYPE

cdef extern from "cpp/numflow.hpp":
    struct Dataset3D:
        INTDTYPE dx
        INTDTYPE dy
        INTDTYPE dz
        DTYPE * ax
        DTYPE * ay
        DTYPE * az
        DTYPE * data

    struct DataMatrix:
        INTDTYPE rows
        INTDTYPE columns
        DTYPE * data
        
    Dataset3D * load_rectilinear_3d(const DataMatrix * mat, DTYPE epsilon)
    DataMatrix * parse_file(const char * filename, const char * sep)

    DTYPE * interpolate_3d(const Dataset3D * dataset, const DTYPE *points, const INTDTYPE count)

    void delete_dataset_3d(Dataset3D * ds)
    void delete_datamatrix(DataMatrix * dm)
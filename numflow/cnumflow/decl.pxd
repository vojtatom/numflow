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

    struct DataStreamlines:
        INTDTYPE * y
        INTDTYPE dy
        INTDTYPE * f
        INTDTYPE df
        INTDTYPE * t
        INTDTYPE dt
        INTDTYPE * l
        INTDTYPE dl


        
    Dataset3D * load_rectilinear_3d(const DataMatrix * mat, DTYPE epsilon)
    DataMatrix * parse_file(const char * filename, const char * sep)
    Dataset3D * construct_level_3d(const Dataset3D * ds, INTDTYPE x, INTDTYPE y, INTDTYPE z)

    DTYPE * interpolate_3d(const Dataset3D * dataset, const DTYPE *points, const INTDTYPE count)
    DataStreamlines * integrate_3d(const Dataset3D *dataset, DTYPE *points,  const INTDTYPE count, 
                                   const DTYPE t0, const DTYPE tbound)

    void delete_dataset_3d(Dataset3D * ds)
    void delete_datamatrix(DataMatrix * dm)
    void delete_datastreamline(DataStreamlines * ds)
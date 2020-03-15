import numpy as np
from ..core import log
from ..exception import NumflowException
import gc

cimport numpy as np
from .types cimport DTYPE, LONGDTYPE, INTDTYPE
from .decl cimport load_rectilinear_3d, parse_file, DataMatrix, Dataset3D, delete_dataset_3d, delete_datamatrix, construct_level_3d
from .common cimport create_2d_double_numpy, create_1d_double_numpy, create_4d_double_numpy


def load_file(file_name, separator): 
    cdef DataMatrix * mat = parse_file(str.encode(file_name), str.encode(separator))

    if mat == NULL:
        raise NumflowException("Error encountered during file reading")

    cdef np.ndarray[DTYPE, ndim=2] arr = create_2d_double_numpy(mat[0].data, mat[0].rows, mat[0].columns)

    delete_datamatrix(mat)
    return arr

cdef c_construct_rectilinear_3d(DTYPE[:,::1] data, DTYPE epsilon):
    cdef DataMatrix mat
    mat.rows = data.shape[0]
    mat.columns = data.shape[1] #6
    mat.data = &data[0, 0]
    
    #values are sorted inside next call
    cdef Dataset3D * dataset = load_rectilinear_3d(&mat, epsilon)

    if dataset == NULL:
        #raise NumflowException("Error encountered during creating dataset")
        return None

    cdef np.ndarray[DTYPE, ndim=1] ax = create_1d_double_numpy(<void *>dataset[0].ax, dataset[0].dx)
    cdef np.ndarray[DTYPE, ndim=1] ay = create_1d_double_numpy(<void *>dataset[0].ay, dataset[0].dy)
    cdef np.ndarray[DTYPE, ndim=1] az = create_1d_double_numpy(<void *>dataset[0].az, dataset[0].dz)
    
    delete_dataset_3d(dataset)

    return ax, ay, az

def construct_rectilinear_3d(data, epsilon):
    axis = c_construct_rectilinear_3d(data, epsilon)

    if axis is None:
        return None, None 

    data = np.ascontiguousarray((data[:,0:3]).reshape(axis[0].size, axis[1].size, axis[2].size, 3)) #not optimal?
    return axis, data


cdef c_build_pyramide_level_3d(DTYPE[:,:,:,::1] values, DTYPE[::1] x,  DTYPE[::1] y,  DTYPE[::1] z, INTDTYPE xi, INTDTYPE yi, INTDTYPE zi):
    cdef Dataset3D dataset
    dataset.dx = x.size
    dataset.dy = y.size
    dataset.dz = z.size
    dataset.ax = &x[0]
    dataset.ay = &y[0]
    dataset.az = &z[0]
    dataset.data = &values[0, 0, 0, 0]

    cdef Dataset3D * level = construct_level_3d(&dataset, xi, yi, zi)

    cdef np.ndarray[DTYPE, ndim=1] ax = create_1d_double_numpy(<void *>level[0].ax, level[0].dx)
    cdef np.ndarray[DTYPE, ndim=1] ay = create_1d_double_numpy(<void *>level[0].ay, level[0].dy)
    cdef np.ndarray[DTYPE, ndim=1] az = create_1d_double_numpy(<void *>level[0].az, level[0].dz)
    cdef np.ndarray[DTYPE, ndim=4] data = create_4d_double_numpy(<void *>level[0].data, level[0].dx, level[0].dy, level[0].dz, 3)
    
    delete_dataset_3d(level)
    return ax, ay, az, data


def build_pyramide_level_3d(data, axis, x, y, z):
    return c_build_pyramide_level_3d(data, axis[0], axis[1], axis[2], x, y, z)
import numpy as np
from ..core import log
from ..exception import NumflowException
import gc

from .types cimport DTYPE, LONGDTYPE, INTDTYPE
from .decl cimport load_rectilinear_3d, parse_file, DataMatrix, Dataset3D, delete_dataset_3d, delete_datamatrix
from .common cimport create_2d_double_numpy, create_1d_double_numpy


### visible from package level
def load_file(file_name, separator): 
    cdef DataMatrix * mat = parse_file(str.encode(file_name), str.encode(separator))

    if mat == NULL:
        raise NumflowException("Error encountered during file reading")

    arr = create_2d_double_numpy(mat[0].data, mat[0].rows, mat[0].columns)

    delete_datamatrix(mat)
    return arr

def c_construct_rectilinear_3d(DTYPE[:,::1] data, DTYPE epsilon):
    cdef DataMatrix mat
    mat.rows = data.shape[0]
    mat.columns = data.shape[1] #6
    mat.data = &data[0, 0]
    
    #values are sorted inside next call
    cdef Dataset3D * dataset = load_rectilinear_3d(&mat, epsilon)

    if dataset == NULL:
        raise NumflowException("Error encountered during creating dataset")

    ax = create_1d_double_numpy(<void *>dataset[0].ax, dataset[0].dx)
    ay = create_1d_double_numpy(<void *>dataset[0].ay, dataset[0].dy)
    az = create_1d_double_numpy(<void *>dataset[0].az, dataset[0].dz)
    
    delete_dataset_3d(dataset)

    return ax, ay, az

def construct_rectilinear_3d(data, epsilon):
    axis = c_construct_rectilinear_3d(data, epsilon)

    data = np.ascontiguousarray((data[:,0:3]).reshape(axis[0].size, axis[1].size, axis[2].size, 3)) #not optimal?
    
    return axis, data

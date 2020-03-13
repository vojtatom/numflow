# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: language=c++

cimport numpy as np
from .types cimport DTYPE, LONGDTYPE, INTDTYPE

from libc.math cimport sqrt

np.import_array()


cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef pointer_to_int_one_d_numpy_array(void * ptr, np.npy_intp * size):
    """
    Creates 1D np.ndarray by encapsulating INTDTYPE * pointer.
        
        :param void *        ptr:  INTDTYPE pointer pointing to beginning
        :param np.npy_intp * size: pointer to 1D array containg info:   
            size[0] - number of points 
    """

    cdef np.ndarray[INTDTYPE, ndim=1] arr = np.PyArray_SimpleNewFromData(1, size, np.NPY_INT32, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr


cdef pointer_to_double_one_d_numpy_array(void * ptr, np.npy_intp * size):
    """
    Creates 2D np.ndarray by encapsulating DTYPE * pointer.
        
        :param void *        ptr:  DTYPE pointer pointing to beginning
        :param np.npy_intp * size: pointer to 1D array containg info:   
            size[0] - number of points 
    """

    cdef np.ndarray[DTYPE, ndim=1] arr = np.PyArray_SimpleNewFromData(1, size, np.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr


cdef pointer_to_two_d_numpy_array(void * ptr, np.npy_intp * size):
    """
    Creates np.ndarray by encapsulating DTYPE * pointer.
        
        :param void *        ptr:  DTYPE pointer pointing to beginning
        :param np.npy_intp * size: pointer to 2D array containg info:   
            size[0] - number of points 
            size[1] - components per point
    """

    cdef np.ndarray[DTYPE, ndim=2] arr = np.PyArray_SimpleNewFromData(2, size, np.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr
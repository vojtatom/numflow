# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy as np
from .types cimport DTYPE, LONGDTYPE, INTDTYPE

from libc.math cimport sqrt

np.import_array()


cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef DTYPE div(DTYPE a, DTYPE b):
    """
    Divides two numbers in Python fashion.

        :param DTYPE a: number A       
        :param DTYPE b: number B  
    """

    if a < 0:
        if b < 0:
            return -a / -b
        return -(-a / b)
    elif b < 0:
        return -(a / -b)
    return a / b


cdef DTYPE norm(DTYPE * vec, int vec_l):
    """
    Calculate norm of a vector.

        :param DTYPE * vec: pointer to vector
        :param int vec_l: length of vector
    """
    cdef int n
    cdef LONGDTYPE sum = 0, a, b
    for n in range(vec_l):
        a = vec[n]
        sum += a * a
    return sqrt(div(sum, vec_l))


cdef DTYPE dot(DTYPE * vec_a, DTYPE * vec_b, int vec_l, int stride_a, int stride_b):
    """
    Calculate dot product of two vectors.
    
        :param DTYPE * vec_a: pointer to vector A
        :param DTYPE * vec_b: pointer to vector B
        :param int vec_l: number of vector components
        :param int stride_a: stride between elements in the vector A
        :param int stride_b: stride between elements in the vector B
    """

    cdef LONGDTYPE d = 0.0, a, b
    cdef int i
    for i in range(vec_l):
        a = vec_a[i * stride_a]
        b = vec_b[i * stride_b]
        d += a * b
    return d


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


cdef pointer_to_float_one_d_numpy_array(void * ptr, np.npy_intp * size):
    """
    Creates 2D np.ndarray by encapsulating DTYPE * pointer.
        
        :param void *        ptr:  DTYPE pointer pointing to beginning
        :param np.npy_intp * size: pointer to 1D array containg info:   
            size[0] - number of points 
    """

    cdef np.ndarray[DTYPE, ndim=1] arr = np.PyArray_SimpleNewFromData(1, size, np.NPY_FLOAT, ptr)
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

    cdef np.ndarray[DTYPE, ndim=2] arr = np.PyArray_SimpleNewFromData(2, size, np.NPY_FLOAT, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr
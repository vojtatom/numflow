# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from .types cimport DTYPE

cdef pointer_to_double_one_d_numpy_array(void * ptr, np.npy_intp * size)

cdef pointer_to_int_one_d_numpy_array(void * ptr, np.npy_intp * size)

cdef pointer_to_two_d_numpy_array(void * ptr, np.npy_intp * size)

# cython: language_level=3
cimport numpy as np
from .types cimport DTYPE

cdef DTYPE div(DTYPE a, DTYPE b)

cdef DTYPE norm(DTYPE * vec, int vec_l)

cdef DTYPE dot(DTYPE * vec_a, DTYPE * vec_b, int vec_l, int stride_a, int stride_b)

cdef pointer_to_float_one_d_numpy_array(void * ptr, np.npy_intp * size)

cdef pointer_to_int_one_d_numpy_array(void * ptr, np.npy_intp * size)

cdef pointer_to_two_d_numpy_array(void * ptr, np.npy_intp * size)

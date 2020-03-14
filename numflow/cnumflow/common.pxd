# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from .types cimport DTYPE, INTDTYPE

cdef create_1d_int32_numpy(void * ptr, INTDTYPE d1)

cdef create_1d_double_numpy(void * ptr, INTDTYPE d1)

cdef create_2d_double_numpy(void * ptr, INTDTYPE d1, INTDTYPE d2)

cdef create_4d_double_numpy(void * ptr, INTDTYPE d1, INTDTYPE d2, INTDTYPE d3, INTDTYPE d4)

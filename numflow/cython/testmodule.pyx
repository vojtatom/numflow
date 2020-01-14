# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.stdlib cimport malloc, realloc, free
from libc.math cimport fabs, fmax, fmin, pow
cimport numpy as np
cimport cython

cdef ctest():
	cdef int a = 42
	return a

def test():
	cdef int a = ctest()    
	print(a)

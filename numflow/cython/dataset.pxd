# cython: language_level=3

cimport numpy as np
from .types cimport DTYPE

DEF max_variable_dims = 10

#data component
cdef struct s_SData:
	int dim
	const DTYPE * data

ctypedef s_SData SData


#rectilinear dataset
cdef class RectilinearDataset:
	cdef object npgrid
	cdef const DTYPE ** grid
	cdef int grid_dim
	cdef int * grid_l
	cdef int num_points

	cdef object variables
	cdef SData active_variable

	cdef void activate_variable(self, object variable_name)
	cpdef interpolate(self, object variable_name, object points)


	cdef void _add_axis_grid(self, const DTYPE[::1] points)
	cdef void _activate_variable(self, int dim, const DTYPE[::1] comp)
	cdef DTYPE * _interpolate(self, const DTYPE[:,::1] points)
	cdef int _ndinterpolate(self, const DTYPE * points, int points_l, DTYPE * output)
	cdef int _indices(self, int grid_i, DTYPE p, DTYPE * fac, int * ind)
	cdef void _n3_interpolate(self, int * ind, DTYPE * fac, DTYPE * res)

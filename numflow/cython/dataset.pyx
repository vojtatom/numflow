# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libc.stdlib cimport malloc, realloc, free
from libc.math cimport fabs, fmax, fmin, pow
cimport numpy as np
cimport cython

from .common cimport div, pointer_to_two_d_numpy_array
from .types cimport DTYPE

from ..exception import NumflowException
import numpy as np

np.import_array()


DEF max_space_dims = 10
DEF max_variable_dims = 10



cdef class Dataset:
    def __init__(self):
        pass






cdef class RectilinearDataset:
    def __init__(self, object grid, object variables):
        self.grid_dim = 0
        self.npgrid = grid
        self.variables = variables
        self.num_points = 1
        
        #init grid
        cdef int dim = len(grid)
        self.grid = <const DTYPE **> malloc(dim * sizeof(const DTYPE *))
        self.grid_l = <int *> malloc(dim * sizeof(int))
        
        for c in grid:
            self._add_axis_grid(c)


    def __dealloc__(self):
        free(self.grid)
        free(self.grid_l)


    cdef void _add_axis_grid(self, const DTYPE[::1] coords):
        self.grid[self.grid_dim] = &coords[0]
        self.grid_l[self.grid_dim] = coords.size
        
        self.grid_dim += 1
        self.num_points *= coords.size

        if (max_space_dims < self.grid_dim):
            raise NumflowException("Number of spatial coordinates exceeded")


    cpdef interpolate(self, object variable_name, object points):
        #check if variable exists
        if variable_name not in self.variables:
            raise NumflowException("Variable with name '{}' does not exist.".format(variable_name))

        #check for points shape
        if len(points.shape) != 2:
            raise NumflowException("Points have to be specified as 2D array: got '{}', expected '2''.".format(len(points.shape[1])))

        #check for points shape again
        if points.shape[1] != self.grid_dim:
            raise NumflowException("Spatial coordinates shape mismatch: got '{}', expected '{}''.".format(points.shape[1], self.grid_dim))

        #reformat points just in case
        points = points.astype(np.float32)

        #actiavte variable
        cdef int dim = self.variables[variable_name].shape[0] / self.num_points
        if dim > max_variable_dims:
            raise NumflowException("Maximal number of variable dimensions exceeded.")
        self._activate_variable(dim, self.variables[variable_name])

        #perform interpolation
        cdef DTYPE * a = self._interpolate(points)
        
        #pack the result
        cdef np.npy_intp dims[2]
        dims[0] = points.shape[0]
        dims[1] = self.active_variable.dim
        arr = pointer_to_two_d_numpy_array(a, dims)
        return arr


    cdef void activate_variable(self, object variable_name):
        #calculate number of coordinate, set active variable
        cdef int dim = self.variables[variable_name].shape[0] / self.num_points
        self._activate_variable(dim, self.variables[variable_name])


    cdef void _activate_variable(self, int dim, const DTYPE[::1] comp):
        #activate variable for interpolation
        self.active_variable.dim = dim
        self.active_variable.data = &comp[0]


    cdef DTYPE * _interpolate(self, const DTYPE[:,::1] points):
        #perform C interpolation
        cdef int points_l = points.shape[0]
        cdef DTYPE * output = <DTYPE *> malloc(points_l * self.active_variable.dim * sizeof(DTYPE))
        self._ndinterpolate(&points[0, 0], points_l, output)
        return output


    cdef int _ndinterpolate(self, const DTYPE * points, int points_l, DTYPE * output):
        #setup local indices
        cdef int   ind[2 * max_space_dims]
        cdef DTYPE fac[2 * max_space_dims]
        cdef DTYPE res[max_variable_dims]
        cdef int outside

        ### iterate over points
        for i in range(points_l): #here points
            outside = 0
            
            ### locate index in grid
            for d in range(self.grid_dim):
                p = points[i * self.grid_dim + d]
                outside += self._indices(d, p, &fac[d], &ind[d])

            ### perform interpolation
            print(self.grid_dim, outside)
            if self.grid_dim == 3:
                if outside:
                    output[i * 3] = 0
                    output[i * 3 + 1] = 0
                    output[i * 3 + 2] = 0
                else:
                    self._n3_interpolate(ind, fac, &output[i * 3])
            else:
                #TODO implement arbitrary number of dimensions
                return 1

        return 0


    cdef int _indices(self, int grid_i, DTYPE p, DTYPE * fac, int * ind):
        cdef const DTYPE * g  = self.grid[grid_i]
        cdef int           gl = self.grid_l[grid_i]
        cdef DTYPE minv = g[0]#fmin(g[0], g[gl - 1])
        cdef DTYPE maxv = g[gl - 1]#fmax(g[0], g[gl - 1])

        print(minv, maxv)

        cdef int low = 0
        cdef int high = gl - 1
        cdef int middle = <int>((gl - 1) * (p - minv) / (maxv - minv))

        ### check if comp are inside volume
        if p < minv or p > maxv:
            return 1

        ### try to predict the index on uniform
        if p >= g[middle] and p <= g[middle + 1]:
            low = middle
            high = middle + 1 
        else:
            ### if not guessed, perform binary search
            ### has to have more than one layer!!
            middle = (high - low) // 2
            while high - low != 1:
                if p < g[middle]:
                    high = middle
                else:
                    low = middle
                middle =  low + (high - low) // 2  
        
        #[0] is dereference here
        ind[0] = low
        fac[0] = div((p - g[low]), (g[high] - g[low]))
        return 0


    cdef void _n3_interpolate(self, int * ind, DTYPE * fac, DTYPE * res):
        """
        Cython implmentation of trilinear interpolation.
            :param int *     ind:    array of indices on grids
            :param DTYPE *   fac:    factors along idnividual axes

        """
        cdef const DTYPE * comp = self.active_variable.data
        cdef DTYPE c00[3] 
        cdef DTYPE c01[3] 
        cdef DTYPE c10[3] 
        cdef DTYPE c11[3] 
        cdef DTYPE c0[3] 
        cdef DTYPE c1[3]
        cdef int zy   = self.grid_l[1] * self.grid_l[2] * 3
        cdef int zyx0 = zy * ind[0] * 3
        cdef int zyx1 = zy * (ind[0] + 1) * 3
        cdef int zy0  = self.grid_l[2] * ind[1] * 3
        cdef int zy1  = self.grid_l[2] * (ind[1] + 1) * 3
        cdef int zy0ind2 = zy0 + ind[2]
        cdef int zy1ind2 = zy1 + ind[2]

        #TODO REWRITE WITH SSE?
        for i in range(3):
            c00[i] = comp[zyx0 + zy0ind2 + i]     * (1.0 - fac[0]) + comp[zyx1 + zy0ind2 + i]     * fac[0]
            c01[i] = comp[zyx0 + zy0ind2 + 3 + i] * (1.0 - fac[0]) + comp[zyx1 + zy0ind2 + 3 + i] * fac[0]
            c10[i] = comp[zyx0 + zy1ind2 + i]     * (1.0 - fac[0]) + comp[zyx1 + zy1ind2 + i]     * fac[0]
            c11[i] = comp[zyx0 + zy1ind2 + 3 + i] * (1.0 - fac[0]) + comp[zyx1 + zy1ind2 + 3 + i] * fac[0]
            c0[i]  = c00[i] * (1.0 - fac[1]) + c10[i] * fac[1]
            c1[i]  = c01[i] * (1.0 - fac[1]) + c11[i] * fac[1]
            res[i]  = c0[i] * (1.0 - fac[2]) +  c1[i] * fac[2]









    

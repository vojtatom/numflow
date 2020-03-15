import numpy as np
from .cnumflow import interpolate3D
from .exception import NumflowException
from scipy.interpolate import RegularGridInterpolator


class RectilinearDataset:
    def __init__(self, axis, data):
        self.axis = axis
        self.data = data
        self.type = "c"
    

    def __call__(self, points):
        points = np.ascontiguousarray(np.array(points, dtype=np.float64))

        if points.ndim != 2:
            raise NumflowException("Expected array of points with 2 dimensions, got {}".format(points.dim))

        dims = len(self.axis)
        if points.shape[1] != dims:
            raise NumflowException("Mismatch dataset dimensions, got {}, expected {}".format(points.shape[1], dims))

        if dims == 3:
            data = interpolate3D(self.data, self.axis[0], self.axis[1], self.axis[2], points)
        elif dims == 2:
            #TODO implement 2D
            raise NumflowException("Dimensions 2 interpolation: TO BE IMPLEMENTED")
        else:
            raise NumflowException("Unsupported number of dimensions")

        return data
    

class ScipyRectilinearDataset:
    def __init__(self, axis, data):
        self.interpolator = RegularGridInterpolator(axis, data, bounds_error=False, fill_value=[0, 0, 0])
        self.axis = axis
        self.data = data
        self.type = "scipy"

    def __call__(self, points):
        return self.interpolator(points)
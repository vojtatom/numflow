import numpy as np
from ..core import log
from ..exception import NumflowException
import gc

from .types cimport DTYPE, LONGDTYPE, INTDTYPE
from .decl cimport dataset_sort


cdef parse_line(line, sep):
    line = line.strip()
    return [ float(i) for i in line.split(sep) ]

cdef cload_file(file_name, separator):
    num_lines = sum(1 for line in open(file_name))

    with open(file_name, "r") as file:
        line = file.readline()
        dimensions = len(line.split(separator))

        if dimensions % 2 == 1:
            raise NumflowException("Unsuported number of dataset columns: {}".format(dimensions))
        
        data = np.empty((num_lines, dimensions))
        file.seek(0)

        stat, lstat = 0, 0
        for i, l in enumerate(file):
            data[i, :] = parse_line(l, separator)
            stat = int(i / num_lines * 100)
            if lstat != stat:
                lstat = stat
                print('{}%'.format(lstat))

        return data

def load_file(file_name, separator): 
    return cload_file(file_name, separator)


cdef pydataset_sorting(DTYPE[:,::1] data):
    dataset_sort(&data[0, 0], data.shape[1], data.shape[0])


cdef cconstruct_rectilinear(data, decimal_tolerance, mode):
    axis = []
    resolution = []
    dim = data.shape[1] // 2

    cdef int i, j

    for i in range(dim):
        j = i + dim
        uniques, counts = np.unique(data[:, j].round(decimals=decimal_tolerance), return_counts=True) 
        _, axcounts = np.unique(counts, return_counts=True)

        if len(axcounts) != 1:
            return None, None

        axis.append(np.ascontiguousarray(uniques.astype(np.float64)))
        resolution.append(axcounts[0]) # get from single element array

    pydataset_sorting(data)

    data = np.ascontiguousarray((data[:,0:dim]).reshape(*resolution, dim)) #not optimal?
    return axis, data


def construct_rectilinear(data, decimal_tolerance, mode):
    return cconstruct_rectilinear(data, decimal_tolerance, mode)

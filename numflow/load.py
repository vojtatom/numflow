from .exception import NumflowException
from .dataset import RectilinearDataset, ScipyRectilinearDataset
import numpy as np
import gc
from .cnumflow import construct_rectilinear_3d, load_file, sample_dataset_3d


def load(filename, separator=",", points_clustering_tolerance=0.0001, mode="scipy"):
    """Loads dataset from .npy or .csv file in expected format.
    Expected format: 4 or 6 columns with format, a single line looks like:

            x-value, y-value, z-name, x-coordinate, y-coordinate, z-coordinate

    The method automatically detects rectilinear datasets. 
    
    Arguments:
        filename {str} -- filename with suffix .csv or .npy
    
    Keyword Arguments:
        separator {str} -- csv separator, ignored for .npy files (default: {","})
        points_clustering_tolerance {int} -- epsilon delta to be 
        taken into account when scanning for rectilinear datasets (default: {4})
        mode {str} -- mode of dataset, supported values: scipy or c (default: {"scipy"})
    
    Raises:
        NumflowException: raised for unsupported or misformatted files
    
    Returns:
        scipy.interpolator.RegularGridInterpolator -- default interpolator
    """
    if mode not in ["scipy", "c", "both"]:
       raise NumflowException("Unknown mode: {}".format(mode)) 
    
    if filename.endswith(".npy"):
        data = np.load(filename)
    elif filename.endswith(".csv"):
        data = load_file(filename, separator)
    else:
       raise NumflowException("Unknown file format: {}".format(filename)) 


    if data.ndim != 2:
        raise NumflowException("Unsuported number of dimensions: {}".format(data.ndim))
    
    if data.shape[1] != 6:
        raise NumflowException("Unsuported number of dataset columns: {}".format(data.shape[1]))


    #try constructing rectilinear
    gc.collect()
    axis, data = construct_rectilinear_3d(data, points_clustering_tolerance)

    if axis is None:
        #TO BE IMPROVED
        raise NumflowException("Only rectilinear datasets supported")
    else:
        #pyramide = build_pyramide3D(axis, data)
        if mode == "c":
            return RectilinearDataset(axis, data)
        elif mode == "scipy":
            return ScipyRectilinearDataset(axis, data)
        #elif mode == "both":
        return RectilinearDataset(axis, data), ScipyRectilinearDataset(axis, data)


def sample_rectilinear_dataset(dataset, x, y, z):
    pos, vals = sample_dataset_3d(dataset.data, dataset.axis, x, y, z)
    print(pos)
    print(vals)

    return pos, vals
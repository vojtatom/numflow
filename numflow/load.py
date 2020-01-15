import base64

import numpy as np

from .exception import NumflowException
from .cython import RectilinearDataset

def load_rectilinear(layout, format, file):
    space_dim = len(format)
    grid = []


    for _ in range(space_dim):
        s = file.readline().strip()
        r = base64.decodebytes(s)
        grid.append(np.ascontiguousarray(np.frombuffer(r, dtype=np.float32)))

    format = [ int(b) for b in format ]
    num_values = np.prod(format)
    variables = {}
    
    for title, count in zip(layout[::2], layout[1::2]):
        s = file.readline().strip()
        r = base64.decodebytes(s)
        values = np.frombuffer(r, dtype=np.float32)
        #print(num_values, int(count))
        values = np.reshape(values, (*format, int(count)))
        #print(values.shape)
        variables[title] = values

    for i in range(len(grid)):
        #test for flips
        if grid[i][0] > grid[i][1]:
            grid[i] = np.ascontiguousarray(np.flip(grid[i]))

            #flip data
            for key in variables:
                variables[key] = np.flip(variables[key], axis=i)

    for key in variables:
        variables[key] = np.ascontiguousarray(np.flip(variables[key], axis=i).flatten())

    return RectilinearDataset(grid, variables)



def load(filename):
    with open(filename, "rb") as file:
        # read layout information
        layout = [ t.decode("utf-8")  for t in file.readline().strip().split(b" ") ]
        format = [ t.decode("utf-8")  for t in file.readline().strip().split(b" ") ]
        print(layout, format)


        # check validiy
        if layout[0] != "layout:" or format[0] != "format:":
            raise NumflowException("Data file format unknown")

        del layout[0]
        del format[0]

        # number of variables in the data file
        num_variables = len(layout) / 2
        if num_variables == 0:
            raise NumflowException(
                "Unexpected number of variables: {}".format(num_variables))


        if format[0] == "pointcloud":
            raise NumflowException("Data format not implemented")
        elif format[0] == "rectilinear":
            del format[0]
            return load_rectilinear(layout, format, file)
        
        
        
        

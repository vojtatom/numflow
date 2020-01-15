

from numflow.load import load
import numpy as np

dataset = load("rectilinear.numdat")
values = dataset.interpolate("velocity", np.array([[1000000, 1000000, -1000000], [2000000, 2000000, -2000000]]))

print(values)
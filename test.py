#!/bin/python3
from numflow import load
import numpy as np
import time


#file_name = "sun.csv"
#separator= ","
#decimal_tolerance = 4



points = (np.random.rand(10000, 3) - [0, 0, 1]) * 1000000
print(points)
#cdata, sdata = load("data.npy", mode="both")
cdata, sdata = load("sun.csv", mode="both")

start = time.time()
svalues = sdata(points)
end = time.time()
print(end - start)
start = time.time()
cvalues = cdata(points)
end = time.time()
print(end - start)

print(cvalues)
print(np.allclose(svalues, cvalues))

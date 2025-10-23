import numpy
from matplotlib import pyplot as plt
from scipy.optimize import rosen

x = numpy.arange(-2, 2, .01)
y = numpy.arange(-1, 3, .01)
X, Y = numpy.meshgrid(x, x)
z = rosen((X, Y))

#example bounds = [(-2,2),(-1,3)]

def brute_force(func, bounds, step=0.01):
    grids = [np.arange(b[0], b[1], step) for b in bounds] #for given bounds, makes arrays of all points per dimension to evaluate
    mesh = np.meshgrid(*grids) #sorts points into a mesh
    points = np.stack([m.flatten() for m in mesh], axis=1) #flattens mesh into a discrete list to go through

    best_val = np.inf
    best_point = None
    for p in points:
        val = func(p)
        if val < best_val:
            best_val = val
            best_point = p
    return np.array(best_point), best_val

def grad_desc():
    #smth

def newton():
    #smth

def neld_mead():
    #smth

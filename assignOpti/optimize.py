import numpy
from matplotlib import pyplot as plt
from scipy.optimize import rosen

x = numpy.arange(-2, 2, .01)
y = numpy.arange(-1, 3, .01)
X, Y = numpy.meshgrid(x, x)
z = rosen((X, Y))

def brute_force():
    #smth

def grad_desc():
    #smth

def newton():
    #smth

def neld_mead():
    #smth

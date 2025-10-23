import numpy
from matplotlib import pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np

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
    0

def newton(x0, df, ddf, n_steps, step_size):
    # x0 is initial guess (array)
    # df is the first derivative of the function (func)
    # ddf is the second derivative of the function (func)
    # n_steps is the number of steps to perform (num)
    x = np.zeros((n_steps+1,len(x0)))
    x[0,:] = x0
    for i in range(n_steps):
        # x_k+1 = x_k + t = x_k - f'(x_k)/f"(x_k)
        x[i+1,:] = x[i,:] - step_size * np.dot(df(x[i,:]), 1/ddf(x[i,:]))
    return x

#x0 is init guess
def neld_mead(func, x0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tolerance=1e-10, max_iter=1000):
    n = len(x0) #dimensions
    simplex = [x0]
    for i in range(n):
        y = x0.copy()
        y[i] += 0.05  #small perturbation for each dimension
        simplex.append(y)
    simplex = np.array(simplex)
    
    path = [x0.copy()]

    for iteration in range(max_iter):
        vals = np.array([func(x) for x in simplex])
        order = np.argsort(vals)
        simplex = simplex[order] #reorders simplex from best to worst func value
        vals = vals[order]

        if np.std(vals) < tolerance:
            break

        x_best = simplex[0]
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)

        #reflection
        x_reflect = x_centroid + alpha * (x_centroid - x_worst)
        f_reflect = func(x_reflect)

        if vals[0] <= f_reflect < vals[-2]:
            simplex[-1] = x_reflect #accept reflection
        elif f_reflect < vals[0]:
            #expansion
            x_expand = x_centroid + gamma * (x_reflect - x_centroid)
            if func(x_expand) < f_reflect:
                simplex[-1] = x_expand
            else:
                simplex[-1] = x_reflect
        else:
            #contraction
            x_contract = x_centroid + rho * (x_worst - x_centroid)
            if func(x_contract) < vals[-1]:
                simplex[-1] = x_contract
            else:
                #shrink
                simplex = simplex[0] + sigma * (simplex - simplex[0])

        path.append(simplex[0].copy())

    return np.array(path), func(simplex[0]), iteration + 1
    
x_newton = newton([-1,-0.5], rosen_der, rosen_hess, 2, 1)

plt.pcolormesh(X, Y, z, norm='log', vmin=1e-3)
c = plt.colorbar()
plt.plot(x_newton[:,0],x_newton[:,1],'-o')
plt.show()

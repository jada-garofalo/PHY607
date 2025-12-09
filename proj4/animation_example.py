import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



x = np.linspace(0,2*np.pi, num = 101)# numpy array of x-values, shape (N,) for N spatial points
t = np.linspace(0,2*np.pi,num=101)# Array of t-values, shape (M,) for M time points
T,X = np.meshgrid(t, x)
w = np.sin(X)*np.cos(T)# Array of w(x,t). Shape (M,N) for N spatial points & M time points

fig, ax = plt.subplots()
line = ax.plot(x, w[0,:], color = "C0")[0]
points = ax.scatter(x,w[0,:], color = "C0")
ax.set(xlim = (0, 2*np.pi), ylim = [-1.1,1.1])


def update(frame):
    data = np.stack([x, w[frame,:]]).T
    points.set_offsets(data)
    
    ax.set_title(f"T = {t[frame]}")
    line.set_ydata(w[frame,:])
    return (points, line)
    
ani = animation.FuncAnimation(fig = fig, func = update, frames = len(t), interval = 30)
plt.show()

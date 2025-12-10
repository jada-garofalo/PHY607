import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 1 # Bar length


# Animate solution
x = # numpy array of x-values, shape (N,) for N spatial points
t = # Array of t-values, shape (M,) for M time points
w = # Array of w(x,t). Shape (M,N) for N spatial points & M time points
frame_rate = 30 # Animation frame rate

epsilon = 0.001 # A small offset to adjust the bounds of the animation window
fig, ax = plt.subplots()
line = ax.plot(x, w[0,:], color = "C0")[0]
points = ax.scatter(x,w[0,:], color = "C0")
ax.set(xlim = [-epsilon,L+epsilon], ylim = [np.min(w)-epsilon, np.max(w)+epsilon])

def update(frame):
    data = np.stack([x, w[frame,:]]).T
    points.set_offsets(data)
    
    ax.set_title(f"T = {t[frame]}")
    line.set_ydata(w[frame,:])
    return (points, line)
    
ani = animation.FuncAnimation(fig = fig, func = update, frames = len(t), interval = 1000/frame_rate)
ani.save(filename="example.gif", fps = frame_rate, writer="pillow")
plt.show()

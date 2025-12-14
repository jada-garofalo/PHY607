import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import comb

# --------------------------------------------------------
# Problem setup
# --------------------------------------------------------
L = 1.0            # beam length
N = 101            # number of spatial points
dx = L / (N - 1)
x = np.linspace(0, L, N)

Tmax = 5.0         # total simulation time
c2 = 50
M = 5000           # number of time samples
dt =  Tmax / (1000*(M - 1))
t = np.linspace(0, Tmax, M)

# --------------------------------------------------------
# Helper: nth-order centered finite difference
# Using the formula provided
# --------------------------------------------------------
def nth_derivative(w, n, dx):
    # result array
    out = np.zeros_like(w)

    # half-step grid offsets:
    # shift = (n - 2*i)/2 * dx  (index shift = (n - 2*i)/2)
    for i in range(n + 1):
        shift = int((n - 2*i) // 2)

        # coefficient
        coef = ((-1)**i) * comb(n, i)

        # shifted version of w with zero padding
        if shift > 0:
            out[shift:] += coef * w[:-shift]
        elif shift < 0:
            out[:shift] += coef * w[-shift:]
        else:
            out += coef * w
    
    out = out / (dx**n)
    return out


# --------------------------------------------------------
# PDE is w_tt = - w_xxxx
# Time stepping: forward Euler for velocity and position
# --------------------------------------------------------
def step_euler(w, v, dx, dt):
    w_xxxx = c2 * nth_derivative(w, 4, dx)
    v_new = v + dt * (-w_xxxx)
    w_new = w + dt * v_new
    return w_new, v_new


# --------------------------------------------------------
# fixed-fixed (clamped-clamped) BCs:
#   w(0)=0, w_x(0)=0  => w[0]=0 and w[1]=0
#   w(L)=0, w_x(L)=0  => w[-1]=0 and w[-2]=0
# enforce v same way.
# --------------------------------------------------------
def apply_boundary_conditions(w, v):
    w[0] = w[1] = 0.0
    w[-1] = w[-2] = 0.0
    v[0] = v[1] = 0.0
    v[-1] = v[-2] = 0.0
    return w, v


# --------------------------------------------------------
# Initial conditions
# Smooth, low-mode shape
# --------------------------------------------------------
w = np.zeros((M, N))

# initial displacement: smooth bump exciting mode 2 slightly
w0 = 0.01 * np.sin(np.pi*L*x)
v0 = np.zeros_like(w0)

w[0,:] = w0
v = v0.copy()


# --------------------------------------------------------
# Time evolution loop
# --------------------------------------------------------
for k in range(1, M):
    w_new, v_new = step_euler(w[k-1,:], v, dx, dt)
    w_new, v_new = apply_boundary_conditions(w_new, v_new)
    w[k,:] = w_new
    v = v_new


# --------------------------------------------------------
# Animation
# --------------------------------------------------------
frame_rate = 100 # Animation frame rate
frame_skip = 10

epsilon = 0.001 # A small offset to adjust the bounds of the animation window
fig, ax = plt.subplots()
line = ax.plot(x, w[0,:], color = "C0")[0]
points = ax.scatter(x,w[0,:], color = "C0")
ax.set(xlim = [-epsilon,L+epsilon], ylim = [np.min(w)-epsilon, np.max(w)+epsilon])

def update(frame):
    data = np.stack([x, w[frame*frame_skip,:]]).T
    points.set_offsets(data)

    ax.set_title(f"T = {t[frame*frame_skip]:.3f}")
    line.set_ydata(w[frame*frame_skip,:])
    return (points, line)

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(t)//frame_skip,
    interval = 1000/frame_rate
)
ani.save(filename="example.gif", fps = frame_rate, writer="pillow")
plt.show()


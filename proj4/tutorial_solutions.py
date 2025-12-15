import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import comb

# --------------------------------------------------------
# Problem setup (feel free to play around with these values!)
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
# nth-order centered finite difference
# Using the formula provided, write a function to take the nth spatial derivative
# format: nth_derivative()
# --------------------------------------------------------
def nth_derivative(w, n, dx):
    # result array
    out = np.zeros_like(w)

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


# ========================================================
# Crank–Nicolson evolution (energy-stable demonstration)
# ========================================================


from scipy.linalg import solve_banded


def biharmonic_matrix(N, dx):
    n = N - 4
    A = np.zeros((n, n))

    for i in range(n):
        A[i,i] = 6.0
        if i-1 >= 0:
            A[i,i-1] = -4.0
        if i+1 < n:
            A[i,i+1] = -4.0
        if i-2 >= 0:
            A[i,i-2] = 1.0
        if i+2 < n:
            A[i,i+2] = 1.0

    return A / dx**4


def step_crank_nicolson(w, v, A, dt, c2):
    """
    Advance one timestep using Crank–Nicolson.
    Operates on interior points only.
    """
    I = np.eye(len(w))

    # Linear system for v^{n+1}
    lhs = I + (dt**2 / 4) * c2 * A
    rhs = (I - (dt**2 / 4) * c2 * A) @ v - dt * c2 * A @ w

    v_new = np.linalg.solve(lhs, rhs)
    w_new = w + 0.5 * dt * (v + v_new)

    return w_new, v_new


A = biharmonic_matrix(N, dx)

w_cn = np.zeros((M, N))
v_cn = np.zeros(N)

w_cn[0,:] = w0
v_cn[:]   = v0

# extract interior
w_int = w0[2:-2].copy()
v_int = v0[2:-2].copy()


for k in range(1, M):
    w_int, v_int = step_crank_nicolson(w_int, v_int, A, dt, c2)

    w_cn[k,2:-2] = w_int
    v_cn[2:-2]   = v_int

    # enforce clamped BCs explicitly
    w_cn[k,0] = w_cn[k,1] = 0.0
    w_cn[k,-1] = w_cn[k,-2] = 0.0
    v_cn[0] = v_cn[1] = 0.0
    v_cn[-1] = v_cn[-2] = 0.0


plt.figure(figsize=(7,4))
plt.plot(x, w[-1], label="Forward Euler", lw=2)
plt.plot(x, w_cn[-1], "--", label="Crank–Nicolson", lw=2)
plt.xlabel("x")
plt.ylabel("w(x, T)")
plt.title("Beam displacement at final time")
plt.legend()
plt.tight_layout()
plt.show()


# crank-nicolson animation
frame_rate = 100 # Animation frame rate
frame_skip = 50

epsilon = 0.001 # A small offset to adjust the bounds of the animation window
fig, ax = plt.subplots()
line2 = ax.plot(x, w_cn[0,:], color = "C0")[0]
points2 = ax.scatter(x, w_cn[-1], label="Crank–Nicolson", lw=2)
ax.set(xlim = [-epsilon,L+epsilon], ylim = [np.min(w)-epsilon, np.max(w)+epsilon])
ax.set_xlabel("x position")
ax.set_ylabel("displacement w(x)") 

def update2(frame):
    data = np.stack([x, w[frame*frame_skip,:]]).T
    points2.set_offsets(data)

    ax.set_title(f"T = {t[frame*frame_skip]:.3f}")
    line2.set_ydata(w[frame*frame_skip,:])
    return (points2, line2)

ani = animation.FuncAnimation(
    fig = fig,
    func = update2,
    frames = len(t)//frame_skip,
    interval = 1000/frame_rate
)
#ani.save(filename="cn.gif", fps = frame_rate, writer="pillow")
plt.show()

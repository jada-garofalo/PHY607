import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
m = 1
t0 = 0
k = 0.25
x0 = 0
v0 = 1

def F(x):
    F = -k * x
    return F

def symplectic(x, v, t, dt):
    x_sym = [x]
    v_sym = [v]
    e_sym = [((0.5*m*(v**2)) + (0.5*k*(x**2)))]
    t_sym = [t]
    while t<100:
        v = v + F(x) * dt
        x = x + v * dt
        t = t + dt
        x_sym.append(x)
        v_sym.append(v)
        e_sym.append((0.5*m*(v**2)) + (0.5*k*(x**2)))
        t_sym.append(t)
    return x_sym, v_sym, e_sym, t_sym

def euler_explicit(x, v, t, dt):
    x_eul = [x]
    v_eul = [v]
    e_eul = [((0.5*m*(v**2)) + (0.5*k*(x**2)))]
    t_eul = [t]
    while t<100:
        x = x + v * dt
        v = v + F(x) * dt
        t = t + dt
        x_eul.append(x)
        v_eul.append(v)
        e_eul.append((0.5*m*(v**2)) + (0.5*k*(x**2)))
        t_eul.append(t)
    return x_eul, v_eul, e_eul, t_eul

def rk2(x, v, t, dt):
    x_rk2 = [x]
    v_rk2 = [v]
    e_rk2 = [((0.5*m*(v**2)) + (0.5*k*(x**2)))]
    t_rk2 = [t]
    while t<100:
        x = x + 0.5 * v * dt
        v = v + F(x) * dt
        x = x + 0.5 * v * dt
        t = t + dt
        x_rk2.append(x)
        v_rk2.append(v)
        e_rk2.append((0.5*m*(v**2)) + (0.5*k*(x**2)))
        t_rk2.append(t)
    return x_rk2, v_rk2, e_rk2, t_rk2

x_sym, v_sym, e_sym, t_sym = symplectic(x0, v0, t0, dt)
x_eul, v_eul, e_eul, t_eul = euler_explicit(x0, v0, t0, dt)
x_rk2, v_rk2, e_rk2, t_rk2 = rk2(x0, v0, t0, dt)

plt.plot(t_sym, e_sym, label='symplectic', color='blue')
plt.plot(t_eul, e_eul, label='euler explicit', color='green')
plt.plot(t_rk2, e_rk2, label='rk2', color='black')
plt.xlabel('time')
plt.ylabel('energy')
plt.legend()
plt.title('energy vs time for symplectic, euler explicit, and rk2')
plt.show()

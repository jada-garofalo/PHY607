import matplotlib.pyplot as plt
import numpy as np

#global defs / initialization
g = 9.81 #grav acel
C_drag = 0 #drag force coeff
t = 0 #initial time
dt = 0.001 #time step
m = 1000 #cow mass kg
r = np.array([0,1000])
v = np.array([1,100])
history = {
        "r": [],
        "v": [],
        "E": []
}

def get_force(v):
    F_grav = np.array([0, -m * g])
    v_mag = np.linalg.norm(v)
    F_drag = -C_drag * v_mag * v
    F = F_grav + F_drag
    return F

def position_velocity_update(r, v, F, dt):
    a = F/m
    v_new = v + a*dt
    r_new = r + v*dt
    return r_new, v_new
    
def get_energies(r, v):
    PE_new = 0.5*m*(v[0]**2 + v[1]**2)
    KE_new = m*g*r[1]
    E_new = PE_new + KE_new
    return PE_new, KE_new, E_new

while r[1]>0:
    F = get_force(v)
    r_new, v_new = position_velocity_update(r, v, F, dt)
    KE_new, PE_new, E_new = get_energies(r, v)
    
    history["r"].append(r_new)
    history["v"].append(v_new)
    history["E"].append(E_new)
    
    r = r_new
    v = v_new
    t = t+dt

plot_selection = input('Type "position", "velocity", or "energy" to choose plot variable:')
plot_time = np.array(range(0, len(history["E"])))*dt

if plot_selection == "position":
    plot_position = np.array(history["r"])
    plt.plot(plot_time, plot_position[:,0], plot_time, plot_position[:,1])
    plt.ylabel("position")
    plt.title("Cow Position vs Time")
    plt.legend(["x position", "y position"])
    
elif plot_selection == "velocity":
    plot_velocity = np.array(history["v"])
    plt.plot(plot_time, plot_velocity[:,0], plot_time, plot_velocity[:,1])
    plt.ylabel("velocity")
    plt.title("Cow Velocity vs Time")
    plt.legend(["x velocity", "y velocity"])

elif plot_selection == "energy":
    plt.plot(plot_time, history["E"])
    plt.ylabel("energy")
    plt.title("Cow Energy vs Time")

else:
    print("invalid plot selection")
    
plt.xlabel("time")
plt.show()


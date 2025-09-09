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
        "E": [],
        "PE": [],
        "KE": []
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

f = open("position_nodrag.out", "w")
f.write(f"{t} {r[0]} {r[1]}\n")

while dt <= 0.1:
    while r[1]>0:
        F = get_force(v)
        r_new, v_new = position_velocity_update(r, v, F, dt)
        KE_new, PE_new, E_new = get_energies(r, v)
        
        history["r"].append(r_new)
        history["v"].append(v_new)
        history["E"].append(E_new)
        history["PE"].append(PE_new)
        history["KE"].append(KE_new)
        
        r = r_new
        v = v_new
        t = t+dt
        if dt==0.001:
            f.write(f"{t} {r[0]} {r[1]}\n")
    
    f.close()

    plot_selection = input('Type "position", "velocity", or "energy" to choose plot variable:')
    plot_time = np.linspace(0,t,num=len(history["E"]))

    plot_position = np.array(history["r"])
    if plot_selection == "position":
        plt.plot(plot_time, plot_position[:,0], plot_time, plot_position[:,1])
        plt.ylabel("position (m)")
        plt.title("Cow Position vs Time")
        plt.legend(["x position", "y position"])
        plt.xlabel("time (s)")
        plt.show()

    elif plot_selection == "velocity":
        plot_velocity = np.array(history["v"])
        plt.plot(plot_time, plot_velocity[:,0], plot_time, plot_velocity[:,1])
        plt.ylabel("velocity (m/s)")
        plt.title("Cow Velocity vs Time")
        plt.legend(["x velocity", "y velocity"])
        plt.xlabel("time (s)")
        plt.show()

    elif plot_selection == "energy":
        plt.plot(plot_time, history["E"])
        plt.ylabel("energy (J)")
        plt.title("Cow Energy vs Time")
        plt.xlabel("time (s)")
        plt.show()

    else:
        print("invalid plot selection")
        
    plt.plot(plot_position[:,0],plot_position[:,1], label="Numerical")
    
    x_analytical = 1*plot_time
    y_analytical = 1000+(100/1)*x_analytical-0.5*(g/1**2)*x_analytical**2
    plt.plot(x_analytical, y_analytical, ls='--', label="Analytical")
    plt.ylabel("y position (m)")
    plt.xlabel("x position (m)")
    plt.title(f"Cow Path, dt = {dt}")
    plt.legend()
    plt.show()

    plt.plot(plot_time, history["E"], plot_time, history["PE"], plot_time, history["KE"])
    plt.ylabel("energy (J)")
    plt.xlabel("time (s)")
    plt.title(f"Cow Energy vs Time, dt = {dt}")
    plt.legend(["Total Energy", "Potential Energy", "Kinetic Energy"])
    plt.show()
    
    dt = dt*10


while r[1]>0:
    F = get_force(v) #Change this to whatever Jada calls it
    r_new, v_new = position_velocity_update(r, v, F, dt)
    KE_new, PE_new, E_new = get_energies(r, v) #Change this to whatever Will calls it
    
    history["r"] = history["r"].append(r_new) #change 'history' to whatever Jada calls it
    history["v"] = history["v"].append(v_new)
    history["E"] = history["E"].append(E_new)
    
    r = r_new
    v = v_new
    t = t+dt


# for all parts of the plotting section below:
  # 'history' name can be changed accordingly
  # contents of 'history; for each key is assumed to be in numpy array form
  # 'np.' can be changed to whatever numpy is initialized as
  # 'plt.' can be changed to whatever matplotlib.pyplot is initialized as
  
plot_selection = input('Type "position", "velocity", or "energy" to choose plot variable:')
plot_time = np.array(range(0, len(history["E"])))*dt

if plot_selection == "position":
    plt.plot(plot_time, history["r"][:,0], plot_time, history["r"][:,1])
    plt.ylabel("position")
    plt.title("Cow Position vs Time")
    plt.legend(["x position", "y position"])
    
elif plot_selection == "velocity":
    plt.plot(plot_time, history["v"][:,0], plot_time, history["v"][:,1])
    plt.ylabel("velocity")
    plt.title("Cow Velocity vs Time")
    plt.legend(["x velocity", "y velocity"])

elif plot_selection == "energy":
    plt.plot(plot_time, history["E"][:,0], plot_time, history["E"][:,1])
    plt.ylabel("energy")
    plt.title("Cow Energy vs Time")

else
    print("invalid plot selection")
    
plt.xlabel("time")


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

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


def position_velocity_update(r, v, F, dt):
    a = F/m
    v_new = v + a*dt
    r_new = r + v*dt
    return r_new, v_new

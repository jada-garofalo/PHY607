def position_velocity_update(r, v, F, dt):
    a = F/m
    v_new = v + a*dt
    r_new = r + v*dt
    return r_new, v_new

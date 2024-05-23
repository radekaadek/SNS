import numpy as np

def topo(h_el, el_sat):
    h = h_el - 31.36
    p0 = 1013.25
    t0 = 291.15
    Rh0 = 0.5
    p = p0 * (1 - 0.0000226*h)**(5.225)
    t = t0 - 0.0065*h
    Rh = Rh0 * np.exp(-0.0006396)
    e = 6.11 * Rh * 10**((7.5*(t-273.15))/(t-35.85))
    deltaT_d0 = 0.002277*p
    deltaT_w0 = 0.002277*(1255/t + 0.05)*e
    md = 1/(np.sin(np.deg2rad(np.sqrt(el_sat ** 2 + 6.25))))
    mw = 1/(np.sin(np.deg2rad(np.sqrt(el_sat ** 2 + 2.25))))
    deltaT = md * deltaT_d0 + mw * deltaT_w0
    return deltaT


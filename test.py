import numpy as np
from numpy import pi

from poliastro.core.hyper import hyp2f1b
from poliastro.core.stumpff import c2, c3
from poliastro.core.util import cross, norm
from astropy import units as u, constants as c
from poliastro.iod.vallado import lambert as lambert_v
from numba import jit

""" At @jit on it shows Maximum iterations reached in short=False
At @jit off it shows incompatible dimensions error """


 
@jit                         
def vallado(k, r0, r, tof, short, numiter, rtol):
    
    if short:
        t_m = +1
    else:
        t_m = -1

    norm_r0 = np.dot(r0, r0) ** 0.5
    norm_r = np.dot(r, r) ** 0.5
    norm_r0_times_norm_r = norm_r0 * norm_r
    norm_r0_plus_norm_r = norm_r0 + norm_r

    cos_dnu = np.dot(r0, r) / norm_r0_times_norm_r

    A = t_m * (norm_r * norm_r0 * (1 + cos_dnu)) ** 0.5

    if A == 0.0:
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")

    psi = 0.0
    psi_low = -4 * np.pi
    psi_up = 4 * np.pi

    count = 0

    while count < numiter:
        y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
        if A > 0.0:
            # Readjust xi_low until y > 0.0
            # Translated directly from Vallado
            while y < 0.0:
                psi_low = psi
                psi = (
                    0.8
                    * (1.0 / c3(psi))
                    * (1.0 - norm_r0_times_norm_r * np.sqrt(c2(psi)) / A)
                )
                y = norm_r0_plus_norm_r + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5

        xi = np.sqrt(y / c2(psi))
        tof_new = (xi ** 3 * c3(psi) + A * np.sqrt(y)) / np.sqrt(k)
        

        # Convergence check
        if np.abs((tof_new - tof) / tof) < rtol:
            break
        count += 1
        # Bisection checkhttps://gist.github.com/NanuSai/2086447dd7b5243cf0dee45a9d00da82

    f = 1 - y / norm_r0
    g = A * np.sqrt(y / k)

    gdot = 1 - y / norm_r

    v0 = (r - f * r0) / g
    v = (gdot * r - r0) / g

    return v0, v


if __name__ == "__main__":
    k = c.GM_earth.to(u.km ** 3/ u.s ** 2).value
    r0 = [10000., 0, 0] * u.km
    rf = [8000., -5000, 0] * u.km
    tof = (2 * u.hour).to(u.s)
    short = False
    numiter = 10**3
    rtol = 1e-6
    v_long = vallado(k, r0, rf, tof, short=short, numiter=numiter, rtol=rtol)
    v_short = vallado(k,r0,rf,tof,short=True,numiter=numiter,rtol=rtol)
    print(v_long)
    print(v_short)
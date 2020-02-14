import numpy as np
from numpy import pi

from poliastro.core.hyper import hyp2f1b
from poliastro.core.stumpff import c2, c3
from poliastro.core.util import cross, norm
from astropy import units as u, constants as c
from numba import jit

@jit
def lol():
    tof_new = 35 * u.second
    tof = 36 * u.km
    print(tof-tof_new)
    return


lol()
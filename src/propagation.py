import numpy as np

'''
This file contain the functions to transform perturbation force in RSW coordinate into the dynamics of Keplerian/Equinoctial elements.

The dynamics of Keplerian/Equinoctial elements can be written as:

d (element)/ dt = A(element,mu) * F_RSW(element) + b(element,mu)

where A is a (6x3) matrix and b is a (6x1) vector, and mu is the central body's gravitational parameter.

'''

########################################################### Equinoctial ###########################################################
'''
The A matrix (6,3) to transform RSW forces into Equinoctial element time derivatives
'''
def RSW2equi_A(y, mu):
    sqrtpmu = np.sqrt(y[0] / mu)
    cosL = np.cos(y[5])
    sinL = np.sin(y[5])
    w = 1 + y[1] * cosL + y[2] * sinL
    s = 1 + y[3]**2 + y[4]**2

    return (np.array([
        [0, 2 * y[0] / w, 0],
        [sinL, ((w + 1) * cosL + y[1]) / w, -(y[3] * sinL - y[4] * cosL) / w * y[2]],
        [-cosL, ((w + 1) * sinL + y[2]) / w, (y[3] * sinL - y[4] * cosL) / w * y[1]],
        [0, 0, s / 2 / w * cosL],
        [0, 0, s / 2 / w * sinL],
        [0, 0, (y[3] * sinL - y[4] * cosL) / w]
    ]) * sqrtpmu)

'''
The b vector (6,) to transform RSW forces into Equinoctial element time derivatives
'''
def RSW2equi_b(y, mu):
    return np.array([0, 0, 0, 0, 0, np.sqrt(mu * y[0]) * ((1 + y[1] * np.cos(y[5]) + y[2] * np.sin(y[5])) / y[0])**2])

########################################################### Keplerian ###########################################################
'''
The A matrix (6,3) to transform RSW forces into Keplerian element time derivatives
'''
def RSW2kepl_A(y, mu):
    cosnu = np.cos(y[5])
    sinnu = np.sin(y[5])

    cosu = np.cos(y[4] + y[5])
    sinu = np.sin(y[4] + y[5])

    h = np.sqrt(mu * y[0] * (1 - y[1]**2))
    p = y[0]*(1-y[1]**2)
    p_r = 1+y[1]*cosnu
    r = p / p_r
    p_h = h / mu
    p_he = p_h / y[1]

    return np.array([
        [2*y[0]**2/h*y[1]*sinnu, 2*y[0]**2/h*p_r, 0],
        [p_h*sinnu,p_h*(cosnu+(y[1]+cosnu)/p_r),0],
        [0, 0, r*cosu/h],
        [0, 0, r*sinu/h/np.sin(y[2])],
        [-p_he*cosnu, p_he*(1+1/p_r)*sinnu, -r*sinu/np.tan(y[2])/h],
        [p_he*cosnu, -p_he*(1+1/p_r)*sinnu, 0]
    ])

'''
The b vector (6,) to transform RSW forces into Keplerian element time derivatives
'''
def RSW2kepl_b(y, mu):
    return np.array([0, 0, 0, 0, 0, np.sqrt(mu * y[0] * (1 - y[1]**2))/((y[0]*(1-y[1]**2)/(1+y[1]*np.cos(y[5])))**2)])

######################################################################################################################
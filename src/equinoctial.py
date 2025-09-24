import numpy as np
from scipy.special import legendre

def zonal_RSW_equi(y, J_array, mu, Re):
    cosL = np.cos(y[5])
    sinL = np.sin(y[5])
    w = 1 + y[1]*cosL + y[2]*sinL
    r = y[0] / w
    mu_r2 = mu / r**2
    Re_r = Re / r

    S = 1 + y[3]**2 + y[4]**2
    FR = 0.0
    FT = 0.0
    FN = 0.0
    alpha = 2 * (y[3]*sinL - y[4]*cosL) / S
    T_pre = 2 * (y[3]*cosL + y[4]*sinL) / S
    N_pre = (1 - y[3]**2 - y[4]**2) / S

    if alpha < -1 or alpha > 1:
        raise ValueError('alpha out of range')

    for i, J in enumerate(J_array):
        k = i + 2
        Pk = legendre(k)(alpha)
        Pk_1 = legendre(k-1)(alpha)
        FR += mu_r2 * J * Re_r**k * (k+1) * Pk
        legendre_prime = (-k*alpha*Pk + k*Pk_1) / (1 - alpha**2)
        FT -= mu_r2 * J * Re_r**k * T_pre * legendre_prime
        FN -= mu_r2 * J * Re_r**k * N_pre * legendre_prime

    return np.array([FR, FT, FN])

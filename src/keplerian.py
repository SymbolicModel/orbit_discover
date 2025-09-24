import numpy as np
from scipy.special import legendre

def zonal_RSW_kepl(y, J_array, mu, Re):
    cosu = np.cos(y[4]+y[5])
    sinu = np.sin(y[4]+y[5])
    sini = np.sin(y[2])
    cosi = np.cos(y[2])

    p = y[0]*(1-y[1]**2)
    p_r = 1+y[1]*np.cos(y[5])
    r = p / p_r
    mu_r2 = mu / r**2
    Re_r = Re / r

    FR = 0.0
    FT = 0.0
    FN = 0.0
    alpha = sini*sinu
    T_pre = sini*cosu
    N_pre = cosi

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
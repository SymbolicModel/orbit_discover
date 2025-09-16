import numpy as np

'''
This file contain the functions to convert between [r,v] in Cartesian Coordinate, Keplerian and Equinoctial elements.

Note that the conversion to/from [r,v] to Keplerian/Equinoctial elements require knowledge of the gravitational parameter (mu) of the central body. However, conversion between Keplerian and Equinoctial elements does not require mu.

'''
############# Between Keplerian and Equinoctial Elements ################
def keplerian2equinoctial(kepl):
    # Keplerian Elements in the format [a, e, incl, raan, argp, nu]:  array-like, shape (..., 6)
        # a: semi-major axis (scalar / numpy vector)
        # e: eccentricity (scalar / numpy vector)
        # incl: inclination (scalar / numpy vector)
        # raan: right ascension of ascending node (scalar / numpy vector)
        # argp: argument of perigee (scalar / numpy vector)
        # nu: true anomaly (scalar / numpy vector)

    # Output: equinoctial elements (1D numpy vector if inputs are scalar, (...,6) array if inputs are vectors)

    kepl = np.asarray(kepl)
    a = kepl[..., 0]
    e = kepl[..., 1]
    incl = kepl[..., 2]
    raan = kepl[..., 3]
    argp = kepl[..., 4]
    nu = kepl[..., 5]

    p = a * (1 - e ** 2)
    f = e * np.cos(argp + raan)
    g = e * np.sin(argp + raan)
    h = np.tan(incl / 2) * np.cos(raan)
    k = np.tan(incl / 2) * np.sin(raan)
    L = np.mod(argp + raan + nu, 2 * np.pi)
    
    return np.array([p, f, g, h, k, L]).T

def equinoctial2keplerian(y):
    # y: array-like, shape (..., 6)
    # Returns: a, e, i, omega, raan, nu (all in radians)
    y = np.asarray(y)
    p = y[..., 0]
    f = y[..., 1]
    g = y[..., 2]
    h = y[..., 3]
    k = y[..., 4]
    L = y[..., 5]

    a = p / (1 - f**2 - g**2)
    e = np.sqrt(f**2 + g**2)
    i = 2 * np.arctan(np.sqrt(h**2 + k**2))
    raan = np.arctan2(k, h)
    omega = np.arctan2(g, f) - raan
    nu = L - np.arctan2(g, f)
    # Ensure angles are in [0, 2*pi)
    omega = np.mod(omega, 2 * np.pi)
    nu = np.mod(nu, 2 * np.pi)
    raan = np.mod(raan, 2 * np.pi)
    return np.array([a, e, i, raan, omega, nu]).T


############# Between Cartesian [r,v] and Keplerian Elements ################
def ijk2keplerian(r, v, mu):
    '''From Cartesian [r,v] to Keplerian Elements'''
    # r: position vector (3D), shape (...,3)
    # v: velocity vector (3D), shape (...,3)
    
    r = np.asarray(r)
    v = np.asarray(v)

    # Calculate specific angular momentum
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)

    # Calculate eccentricity vector
    e = (np.cross(v, h) / mu) - (r / np.linalg.norm(r,axis=-1,keepdims=True))
    e_mag = np.linalg.norm(e, axis=-1)

    # Calculate semi-major axis
    energy = 0.5 * np.linalg.norm(v, axis=-1) ** 2 - mu / np.linalg.norm(r, axis=-1)
    a = -mu / (2 * energy)

    # Calculate inclination
    incl = np.arccos(h[...,2] / np.linalg.norm(h, axis=-1))

    # Calculate raan
    raan = np.arctan2(n[...,1], n[...,0])

    # Calculate argument of perigee
    argp = np.arccos(np.linalg.vecdot(e, n,axis=-1)  / e_mag / np.linalg.norm(n, axis=-1))
    argp[e[..., 2] < 0] = 2 * np.pi - argp[e[..., 2] < 0]

    # Calculate true anomaly
    nu = np.arccos(np.linalg.vecdot(e, r, axis=-1) / e_mag / np.linalg.norm(r, axis=-1))
    rv_pm = np.linalg.vecdot(v, r, axis=-1) < 0
    nu[rv_pm] = 2 * np.pi - nu[rv_pm]

    return np.array([a, e_mag, incl, raan, argp, nu]).T

def keplerian2ijk(y, mu):
    '''From Keplerian Elements to Cartesian [r,v]'''
    # y: Keplerian element; array-like, shape (..., 6)
    # Returns: position vector (...,3), velocity vector (...,3)
    y = np.asarray(y)
    a = y[..., 0]
    e = y[..., 1]
    incl = y[..., 2]
    raan = y[..., 3]
    argp = y[..., 4]
    nu = y[..., 5]

    # Calculate the semi-latus rectum
    p = a * (1 - e ** 2)

    r = np.zeros((len(a), 3))
    v = np.zeros((len(a), 3))

    for i in range(len(a)):
        # Calculate position in the perifocal frame
        r_perifocal = np.array([
            p[i] * np.cos(nu[i]) / (1 + e[i] * np.cos(nu[i])),
            p[i] * np.sin(nu[i]) / (1 + e[i] * np.cos(nu[i])),
            0
        ])

        # Calculate velocity in the perifocal frame
        v_perifocal = np.array([
            np.sqrt(mu / p[i]) * -np.sin(nu[i]),
            np.sqrt(mu / p[i]) * (e[i] + np.cos(nu[i])),
            0
        ])

        # Rotation matrices
        R3_W = np.array([
            [np.cos(raan[i]), -np.sin(raan[i]), 0],
            [np.sin(raan[i]), np.cos(raan[i]), 0],
            [0, 0, 1]
        ])
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(incl[i]), -np.sin(incl[i])],
            [0, np.sin(incl[i]), np.cos(incl[i])]
        ])
        R3_w = np.array([
            [np.cos(argp[i]), -np.sin(argp[i]), 0],
            [np.sin(argp[i]), np.cos(argp[i]), 0],
            [0, 0, 1]
        ])

        # Total rotation matrix
        R = R3_W @ R1_i @ R3_w

        # Transform to the inertial frame
        r_i = R @ r_perifocal
        v_i = R @ v_perifocal

        r[i] = r_i
        v[i] = v_i

    return r, v

def ijk2equinoctial(r, v, mu):
    # r: position vector (3D), shape (...,3)
    # v: velocity vector (3D), shape (...,3)
    # Returns: equinoctial elements

    kepl = ijk2keplerian(r, v, mu)
    return  keplerian2equinoctial(kepl)

def equinoctial2ijk(equi, mu):
    # equi: array-like, shape (..., 6)
    # Returns: position vector (3D), velocity vector (3D)

    kepl = equinoctial2keplerian(equi)
    return  keplerian2ijk(kepl, mu)

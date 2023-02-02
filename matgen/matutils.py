"""
"""
import math
from typing import Dict, Iterable, List
import numpy as np
import random
from scipy import sparse, linalg, stats
import logging

# from matgen.base import Grain

def _get_IJV_from_neighbors(_cells: Dict) -> tuple[List]:
    """Get I, J, V lists of the adjacency matrix from a dictionary of cells.

    Cells can be vertices, edges, faces or polyhedra of a corresponding
    base class.

    Parameters
    ----------
    _cells
        A dictionary of cells. Keys - cell ids, values - cell objects
        which have `n_ids` attribute.
    
    Returns
    -------
    tuple
        A tuple of lists in the form of (I, J, V) where I - row index,
        J - column index of elements of the adjacency matrix with nonzero
        values. All elements of V is equal to 1. Index of an element is
        (element_id - 1). 
    """

    I = []
    J = []
    V = []
    for cell_id, cell in _cells.items():
        for n_id in cell.n_ids:
            I.append(cell_id - 1)
            J.append(n_id - 1)
            V.append(1)
    
    return (I, J, V)


def _get_IJV_from_incidence(_cells: Dict) -> tuple[List]:
    """Get I, J, V lists of the incidence matrix from a dictionary of cells.

    Cells can be vertices, edges, faces or polyhedra of a corresponding
    base class.

    Parameters
    ----------
    _cells
        A dictionary of cells. Keys - cell ids, values - cell objects
        which have `signed_incident_ids` attribute.
    
    Returns
    -------
    tuple
        A tuple of lists in the form of (I, J, V) where I - row index,
        J - column index of elements of the incidence matrix with nonzero
        values. All elements of V is equal to 1. Index of an element is
        (element_id - 1). Rows correspond to (k - 1)-cells, while columns to
        k-cells.
    """

    I = []
    J = []
    V = []
    for cell_id, cell in _cells.items():
        for signed_inc_id in cell.signed_incident_ids:
            I.append(cell_id - 1)
            J.append(abs(signed_inc_id) - 1)
            if signed_inc_id > 0:
                V.append(1)
            else:
                V.append(-1)
    
    return (I, J, V)


def load_matrix_coo(filename, matrix_shape=None) -> sparse.coo_matrix:
    """
    
    """
    M_sparse = np.loadtxt(filename, dtype='int')
    I = np.array([row[0] for row in M_sparse])
    J = np.array([row[1] for row in M_sparse])
    V = np.array([row[2] for row in M_sparse])

    M_coo = sparse.coo_matrix((V,(I,J)), shape=matrix_shape)

    return M_coo


def calculate_L(B1: sparse.coo_matrix, B2: sparse.coo_matrix):
    """
    """
    return B1.transpose() @ B1 + B2 @ B2.transpose()


def entropy(*args):
    """
    S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # calculate entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 0 or p == 1:
            return 0
        elif p > 0 and p < 1:
            return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
    elif len(j_array) > 1:
        nonzeros = j_array[j_array > 0]
        return - np.sum(nonzeros * np.log2(nonzeros))


def entropy_m(*args):
    """
    mean part of S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # check zero elements
    if np.any(j_array == 0):
        logging.warning('One or more j is equal to 0')
        return np.inf

    # calculate mean entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 1:
            return np.inf
        elif p > 0 and p < 1:
            return - math.log2(p * (1 - p)) / 2
    elif len(j_array) > 1:
        return - np.log2(np.prod(j_array)) / len(j_array)


def entropy_s(*args):
    """
    deviatoric part of S
    """
    # input arguments may be a scalar, a tuple or several scalars
    if len(args) == 1 and isinstance(args[0], Iterable):
        j_array = np.array(args[0])
    else:
        j_array = np.array(args)

    # check sum of input parameters
    if len(j_array) > 1 and not math.isclose(j_array.sum(), 1):
        logging.warning('Sum is not equal to 1')

    # check zero elements
    if np.any(j_array == 0):
        logging.warning('One or more j is equal to 0')
        return - np.inf
    
    # calculate deviatoric entropy
    if len(j_array) == 1:
        p = j_array[0]
        if p == 1:
            return - np.inf
        elif p > 0 and p < 1:
            q = 1 - p
            return - (p - q) / 2 * math.log2(p / q)
    elif len(j_array) > 1:
        Ss = 0
        for k in range(len(j_array)):
            jk = j_array[k]
            for l in range(k + 1, len(j_array)):
                jl = j_array[l]
                Ss += (jk - jl) * math.log2(jk / jl)
        Ss = Ss / len(j_array)
        return - Ss


def get_d_tuple(j_tuple: tuple) -> tuple:
    """
    """
    if j_tuple[0] == 1:
        return 0, 0, 0
    else:
        denom = 1 - j_tuple[0]
        d1 = j_tuple[1] / denom
        d2 = j_tuple[2] / denom
        d3 = j_tuple[3] / denom
        return d1, d2, d3


def ori_mat(ori: tuple, oridesc: str ='euler-bunge:active') -> np.ndarray:
    """
    returns orientation matrix
    """
    
    if oridesc == 'euler-bunge:active':
        f1 = math.radians(ori[0])
        F = math.radians(ori[1])
        f2 = math.radians(ori[2])
        return _R_from_Euler(f1, F, f2)
    elif oridesc == 'euler-roe:active':
        f1 = math.radians(ori[0] + 90)
        F = math.radians(ori[1])
        f2 = math.radians(ori[2] - 90)
        return _R_from_Euler(f1, F, f2)
    elif oridesc == 'rodrigues:active':
        tt = math.sqrt(ori[0] * ori[0] + ori[1] * ori[1] + ori[2] * ori[2])
        t = 2 * math.atan(tt)
        ux = ori[0] / tt
        uy = ori[1] / tt
        uz = ori[2] / tt
        return _R_from_Rodrigues(t, (ux, uy, uz))
    elif oridesc == 'quaternion:active':
        return _R_from_quaternion(ori)


def _R_from_Euler(f1: float, F: float, f2: float):
    """
    """
    cos = math.cos
    sin = math.sin

    R11 = cos(f1)*cos(f2) - sin(f1)*sin(f2)*cos(F)
    R12 = sin(f1)*cos(f2) + cos(f1)*sin(f2)*cos(F)
    R13 = sin(f2)*sin(F)

    R21 = - cos(f1)*sin(f2) - sin(f1)*cos(f2)*cos(F)
    R22 = - sin(f1)*sin(f2) + cos(f1)*cos(f2)*cos(F)
    R23 = cos(f2)*sin(F)

    R31 = sin(f1)*sin(F)
    R32 = - cos(f1)*sin(F)
    R33 = cos(F)

    R = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])
    return R


def _R_from_Rodrigues(t: float, u: tuple) -> np.ndarray:
    """
    """
    ux, uy, uz = u

    cos = math.cos
    sin = math.sin

    R11 = cos(t) + ux*ux*(1 - cos(t))
    R21 = uy*ux*(1 - cos(t)) + uz*sin(t)
    R31 = uz*ux*(1 - cos(t)) - uy*sin(t)

    R12 = ux*uy*(1 - cos(t)) - uz*sin(t)
    R22 = cos(t) + uy*uy*(1 - cos(t))
    R32 = uz*uy*(1 - cos(t)) + ux*sin(t)

    R13 = ux*uz*(1 - cos(t)) + uy*sin(t)
    R23 = uy*uz*(1 - cos(t)) - ux*sin(t)
    R33 = cos(t) + uz*uz*(1 - cos(t))

    R = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])
    return R.T


def _R_from_quaternion(q_tuple: tuple):
    """
    """
    qr, qi, qj, qk = q_tuple

    R11 = 1 - 2*(qj*qj + qk*qk)
    R21 = 2*(qi*qj + qk*qr)
    R31 = 2*(qi*qk - qj*qr)

    R12 = 2*(qi*qj - qk*qr)
    R22 = 1 - 2*(qi*qi + qk*qk)
    R32 = 2*(qj*qk + qi*qr)

    R13 = 2*(qi*qk + qj*qr)
    R23 = 2*(qj*qk - qi*qr)
    R33 = 1 - 2*(qi*qi + qj*qj)

    R = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])

    return R

def calculate_disorient(R1, R2, crysym: str = 'cubic'):
    """
    симметрия кристалла имеет значение - какие еще бывают варианты?
    """
        
    # _g = R1 @ np.linalg.inv(R2)
    _g = R1 @ linalg.inv(R2)
    costheta = (_g[0, 0] + _g[1, 1] + _g[2, 2] - 1) / 2
    if abs(costheta) <= 1:
        theta_min = math.acos(costheta)
    elif costheta > 1:
        theta_min = 0
    elif costheta < -1:
        theta_min = math.acos(-1)
         
    if crysym == 'cubic':
        for Os1 in Osym:
            for Os2 in Osym:
                g = Os1 @ _g @ Os2
                costheta = (g[0, 0] + g[1, 1] + g[2, 2] - 1) / 2
                if abs(costheta) <= 1:
                    theta = math.acos(costheta)
                elif costheta > 1:
                    theta = 0
                elif costheta < -1:
                    theta = math.acos(-1)
                # theta = math.acos((g[0, 0] + g[1, 1] + g[2, 2] - 1) / 2)
                if theta < theta_min:
                    theta_min = theta
    
    return math.degrees(theta_min)


def calculate_disorient_quatern(
    R1: np.ndarray,
    R2: np.ndarray,
    crysym: str = 'cubic',
    angle: bool = False
) -> tuple:
    """
    симметрия кристалла имеет значение - какие еще бывают варианты?
    """
        
    # _g = R1 @ np.linalg.inv(R2)
    _A = R1 @ linalg.inv(R2)
    q0 = math.sqrt(1 + _A[0, 0] + _A[1, 1] + _A[2, 2]) / 2
    A_min = _A
    # q1 = (_A[2, 1] - _A[1, 2]) / (4 * q0)
    # q2 = (_A[0, 2] - _A[2, 0]) / (4 * q0)
    # q3 = (_A[1, 0] - _A[0, 1]) / (4 * q0)
    
    # if q0 <= 1:
    #     theta_min = 2 * math.acos(q0)
    # elif q0 > 1:
    #     theta_min = 0
         
    if crysym == 'cubic':
        for Os1 in Osym:
            for Os2 in Osym:
                A = Os1 @ _A @ Os2
                try:
                    q = math.sqrt(1 + A[0, 0] + A[1, 1] + A[2, 2]) / 2
                except ValueError:
                    q = 0
                # if q0 <= 1:
                #     theta = 2 * math.acos(q0)
                # elif q0 > 1:
                #     theta = 0

                if math.acos(q) < math.acos(q0):
                    q0 = q
                    A_min = A
                    # q1 = (A[2, 1] - A[1, 2]) / (4 * q0)
                    # q2 = (A[0, 2] - A[2, 0]) / (4 * q0)
                    # q3 = (A[1, 0] - A[0, 1]) / (4 * q0)
    if not math.isclose(q0, 0):
        q1 = (A_min[2, 1] - A_min[1, 2]) / (4 * q0)
        q2 = (A_min[0, 2] - A_min[2, 0]) / (4 * q0)
        q3 = (A_min[1, 0] - A_min[0, 1]) / (4 * q0)
    else:
        q1 = math.sqrt(1 + A[0, 0] - A[1, 1] - A[2, 2]) / 2
        q2 = (A_min[0, 1] + A_min[1, 0]) / (4 * q1)
        q3 = (A_min[0, 2] + A_min[2, 0]) / (4 * q1)
    
    if angle:
        theta = 2 * math.acos(q0)
        return (q0, q1, q2, q3), math.degrees(theta)

    return (q0, q1, q2, q3)


def dis_angle(g1: 'Grain', g2: 'Grain', quaternions=False) -> float:
    """
    """
    if g1.crysym != g2.crysym:
        raise ValueError("Crysym of g1 and g2 don't match")
    R1 = ori_mat(g1.ori, g1.oridesc)
    R2 = ori_mat(g2.ori, g2.oridesc)

    if not quaternions:
        return calculate_disorient(R1, R2, g1.crysym)
    else:
        return calculate_disorient_quatern(R1, R2, g1.crysym)


# Cubic symmetry axes
Osym = np.array([
    [
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ],
    [
        [ 1, 0, 0],
        [ 0, 0, -1],
        [ 0, 1, 0]
    ],
    [
        [ 1, 0, 0],
        [ 0, -1, 0],
        [ 0, 0, -1]
    ],
    [
        [ 1, 0, 0],
        [ 0, 0, 1],
        [ 0, -1, 0]
    ],
    [
        [ 0, -1, 0],
        [ 1, 0, 0],
        [ 0, 0, 1]
    ],
    [
        [ 0, 0, 1],
        [ 1, 0, 0],
        [ 0, 1, 0]
    ],
    [
        [ 0, 1, 0],
        [ 1, 0, 0],
        [ 0, 0, -1]
    ],
    [
        [ 0, 0, -1],
        [ 1, 0, 0],
        [ 0, -1, 0]
    ],
    [
        [ -1, 0, 0],
        [ 0, -1, 0],
        [ 0, 0, 1]
    ],
    [
        [ -1, 0, 0],
        [ 0, 0, -1],
        [ 0, -1, 0]
    ],
    [
        [ -1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, -1]
    ],
    [
        [ -1, 0, 0],
        [ 0, 0, 1],
        [ 0, 1, 0]
    ],
    [
        [ 0, 1, 0],
        [ -1, 0, 0],
        [ 0, 0, 1]
    ],
    [
        [ 0, 0, 1],
        [ -1, 0, 0],
        [ 0, -1, 0]
    ],
    [
        [ 0, -1, 0],
        [ -1, 0, 0],
        [ 0, 0, -1]
    ],
    [
        [ 0, 0, -1],
        [ -1, 0, 0],
        [ 0, 1, 0]
    ],
    [
        [ 0, 0, -1],
        [ 0, 1, 0],
        [ 1, 0, 0]
    ],
    [
        [ 0, 1, 0],
        [ 0, 0, 1],
        [ 1, 0, 0]
    ],
    [
        [ 0, 0, 1],
        [ 0, -1, 0],
        [ 1, 0, 0]
    ],
    [
        [ 0, -1, 0],
        [ 0, 0, -1],
        [ 1, 0, 0]
    ],
    [
        [ 0, 0, -1],
        [ 0, -1, 0],
        [ -1, 0, 0]
    ],
    [
        [ 0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0]
    ],
    [
        [ 0, 0, 1],
        [ 0, 1, 0],
        [ -1, 0, 0]
    ],
    [
        [ 0, 1, 0],
        [ 0, 0, -1],
        [ -1, 0, 0]
    ]
])


def _check_chi2(ns: np.ndarray, ms: np.ndarray) -> tuple:
    """
    ns and ms contain frequencies of classes (categories)
    number of bins is equal to number of classes (categories)
    """
    if len(ns) != len(ms):
        raise ValueError("Numbers of bins must be equal")
    else:
        df = len(ns) - 1
    N = ns.sum()
    M = ms.sum()

    mask = (ns != 0) + (ms != 0)
    ns = ns[mask]
    ms = ms[mask]
    s = ((M * ns - N * ms) * (M * ns - N * ms) / (ns + ms)).sum() / N / M
    p_value = 1 - stats.chi2.cdf(s, df=df)

    return s, p_value


def check_distr_diff(
    angles1: np.ndarray,
    angles2: np.ndarray,
    alpha: float = 0.001
):
    """
    LAGBS and HABGS - two categories
    LAGBS if < 15
    HAGBs if >= 15
    """

    ns = np.array([(angles1 < 15).sum(), (angles1 >= 15).sum()])
    ms = np.array([(angles2 < 15).sum(), (angles2 >= 15).sum()])

    _, p_value = _check_chi2(ns, ms)

    if p_value < alpha:
        return p_value, True # difference is significant
    else:
        return p_value, False # difference is not significant


def _hellinger(P: np.ndarray, Q: np.ndarray):
    """
    """
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if not isinstance(Q, np.ndarray):
        Q = np.array(Q)

    if not np.all(P >= 0) or not np.all(Q >= 0):
        raise ValueError
    if len(P) != len(Q):
        raise ValueError

    return np.sqrt(((np.sqrt(P) - np.sqrt(Q)) ** 2).sum()) / math.sqrt(2)

def hellinger_distance(
    angles1: np.ndarray,
    angles2: np.ndarray
):
    """
    """
    P = np.array([(angles1 < 15).sum(), (angles1 >= 15).sum()]) / len(angles1)
    Q = np.array([(angles2 < 15).sum(), (angles2 >= 15).sum()]) / len(angles2)

    return P, Q, _hellinger(P, Q)


def _Xfunc(x):
    return (np.sqrt(2) - 1) / np.sqrt(1 - (np.sqrt(2) - 1)**2 / np.tan(x / 2)**2)


def _Yfunc(x):
    return (np.sqrt(2) - 1)**2 / np.sqrt(3 - 1 / np.tan(x / 2)**2)


def mackenzie(angles: np.ndarray, precision: int = 5):
    """
    degrees (with 2 decimal digits)

    TODO: try np.piecewise

    """
    angles_rad = np.radians(np.round(angles, 2))
    p = np.zeros_like(angles_rad, dtype=float)

    # 0 <= angle <= 45
    mask = (angles >= 0)&(angles <= 45)
    x = angles_rad[mask]
    p[mask] = (2/15)*(1 - np.cos(x))

    # 45 < angle <= 60
    mask = (angles > 45)&(angles <= 60)
    x = angles_rad[mask]
    p[mask] = (2/15)*(3*(np.sqrt(2) - 1)*np.sin(x) - 2*(1 - np.cos(x)))

    # 60 < angle <= 60.72
    mask = (angles > 60)&(angles <= 60.72)
    x = angles_rad[mask]
    p[mask] = (2/15)*(
        (3*(np.sqrt(2) - 1) + 4/np.sqrt(3))*np.sin(x) - 6*(1 - np.cos(x))
    )

    # 60.73 <= angle <= 62.8
    mask = (angles >= 60.73)&(angles <= 62.8)
    x = angles_rad[mask]
    X = _Xfunc(x)
    Y = _Yfunc(x)
    p[mask] = (2 / 15) * (
        (3*(np.sqrt(2) - 1) + 4/np.sqrt(3))*np.sin(x) - 6*(1 - np.cos(x))
    ) - (8/5/math.pi)*(
        (2*(np.sqrt(2) - 1)*np.arccos(X/np.tan(x/2))) +\
            (1/np.sqrt(3))*np.arccos(Y/np.tan(x/2))
    )*np.sin(x) + (8/5/math.pi)*(
        2*np.arccos((np.sqrt(2) + 1)*X/np.sqrt(2)) +\
            np.arccos((np.sqrt(2) + 1)*Y/np.sqrt(2))
    )*(1 - np.cos(x))

    return np.round(p, precision)

def _cdf1(angles):
    """
    angles in degrees
    all angles must be less than or equal to 45
    """
    x = np.radians(angles)
    return 180/math.pi*2/15*(x - np.sin(x))

def _cdf2(angles):
    """
    angles in degrees
    all angles must be greater than 45 and less than or equal to 60
    """
    x = np.radians(angles)
    return 180/math.pi*2/15*(3*(1 - np.sqrt(2))*np.cos(x) - 2*(x - np.sin(x)))

def mackenzie_cdf(angles, precision: int = 5):
    """
    angles in degrees
    less than or equal to 60
    """
    prob = np.zeros_like(angles, dtype=float)
    
    mask = (angles >= 0)&(angles <= 45)
    x = angles[mask]
    prob[mask] = _cdf1(x)
    
    mask = (angles > 45)&(angles <= 60)
    x = angles[mask]
    prob[mask] = _cdf2(x) - _cdf2(45) + _cdf1(45)

    mask = (angles > 45)&(angles <= 60)
    x = angles[mask]
    prob[mask] = _cdf2(x) - _cdf2(45) + _cdf1(45)
    
    mask = (angles > 60)&(angles <= 61)
    x = angles[mask]
    prob[mask] = 0.99850 # approx, not exact

    mask = (angles >= 62)
    x = angles[mask]
    prob[mask] = 1.0 # approx, not exact
    
    return np.round(prob, precision)

def mackenzie_pmf(angles):
    """
    """
    return np.diff(mackenzie_cdf(angles))


def _in_polygon(x: float, y: float, xp: list, yp: list) -> bool:
    """Find whether a point (x, y) is inside the polygon with (xp, yp)
    coordinates of vertices.

    Parameters
    ----------
    x
        x-coordinate of a point
    y
        y-coordinate of a point
    xp
        A list of x-coordinates of the polygon vertices
    yp
        A list of y-coordinates of the polygon vertices

    Returns
    -------
    bool
        True if a point is inside the polygon, False otherwise.

    Notes
    -----
    See https://shorturl.at/FLR69 for details.
    """
    c = 0
    for i in range(len(xp)):
        if ((
            (yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])
            ) and (
                x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) +\
                xp[i]
            )):
            c = 1 - c
    return True if c == 1 else False


def _get_random(min_value: float, max_value: float) -> float:
    """
    """
    return random.uniform(min_value, max_value)


def get_random_point_on_face(
    a: float,
    b: float,
    c: float,
    d: float,
    xp: list,
    yp: list,
    zp: list
) -> tuple:
    """Get (x, y, z) coordinates of a random point on a given face with
    the equation :math:`ax + by + cz = d`.

    Parameters
    ----------
    a, b, c, d
        See Notes
    xp
        A list of x-coordinates of the face vertices
    yp
        A list of y-coordinates of the face vertices
    zp
        A list of z-coordinates of the face vertices
    
    Returns
    -------
    (x, y, z)
        A tuple of the new random point coordinates

    Notes
    -----
    Parameters d, a, b, c are the parameters of the equation of a face
    :math:`ax + by + cz = d` with :math:`a^2 + b^2 + c^2 = 1`. See details
    https://neper.info/doc/fileformat.html
    """

    xmin = np.min(xp)
    xmax = np.max(xp)
    ymin = np.min(yp)
    ymax = np.max(yp)    
    zmin = np.min(zp)
    zmax = np.max(zp)    
    
    
    if c == 0 and b == 0:
        while True:
            y = _get_random(ymin, ymax)
            z = _get_random(zmin, zmax)
            if _in_polygon(y, z, yp, zp):
                break
        x = (d - b * y - c * z) / a
    elif c == 0:
        while True:
            x = _get_random(xmin, xmax)
            z = _get_random(zmin, zmax)
            if _in_polygon(x, z, xp, zp):
                break
        y = (d - a * x - c * z) / b
    else:
        while True:
            x = _get_random(xmin, xmax)
            y = _get_random(ymin, ymax)
            if _in_polygon(x, y, xp, yp):
                break
        z = (d - a * x - b * y) / c

    return (x, y, z)


def get_random_point_on_edge(xp, yp):
    """
    """
    if math.isclose(xp[0], xp[1]):
        x = xp[0]
        y = _get_random(yp[0], yp[1])
        # if isclose(y, yp[0]):
        #     y = _get_random(yp[0], yp[1])
    else:
        x = _get_random(xp[0], xp[1])
        # if isclose(x, xp[0]):
        #     x = _get_random(xp[0], xp[1])    
        y = (yp[1] - yp[0]) / (xp[1] - xp[0]) * (x - xp[0]) + yp[0]

    return (x, y)










# """
# https://networkx.org/documentation/stable/tutorial.html

# https://docs.scipy.org/doc/scipy/reference/sparse.html
# https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html#module-scipy.sparse.csgraph
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg
# """

# from collections import Counter
# from collections.abc import Iterable
# import os
# from random import SystemRandom
# from typing import Dict, List, Tuple, Union
# from math import (
#     log2, sqrt, cos, sin, acos, asin, radians, degrees, isclose
# )
# import numpy as np
# from scipy import sparse
# import networkx as nx
# from scipy.spatial import Delaunay
# import logging

# def get_graph_from_A_file(f: open) -> nx.Graph:
#     """
#     A - adjacency matrix
#     """
#     A_sparse = np.loadtxt(f, dtype='int')
#     IJ = [(row[0], row[1]) for row in A_sparse]
#     G = nx.Graph()
#     G.add_edges_from(IJ)

#     return G

# def get_graph_from_A_file(f: open) -> nx.Graph:
#     """
#     A - adjacency matrix
#     """
#     A_sparse = np.loadtxt(f, dtype='int')
#     IJ = [(row[0], row[1]) for row in A_sparse]
#     G = nx.Graph()
#     G.add_edges_from(IJ)

#     return G


# def get_G_from_cells(_cells: Dict):
#     """
#     from Adj
#     """
#     I, J, _ = _get_IJV_from_neighbors(_cells)
#     IJ = [(i + 1, j + 1) for i, j in zip(I, J)]
#     G = nx.Graph()
#     G.add_edges_from(IJ)
    
#     return G


# def get_A_from_cells(_cells: Dict):
#     """
#     """
#     I, J, V = _get_IJV_from_neighbors(_cells)
#     A_coo = sparse.coo_matrix((V,(I,J)))
#     return A_coo


# def get_B_from_cells(_cells: Dict):
#     """
#     """
#     I, J, V = _get_IJV_from_incidence(_cells)
#     B_coo = sparse.coo_matrix((V,(I,J)))
#     return B_coo


# def save_A(
#         c,
#         work_dir: str = '.'):
#     """
#     """
#     if not os.path.exists(work_dir):
#         os.mkdir(work_dir)
#     # Save A0.txt
#     filename = os.path.join(work_dir, 'A0.txt')
#     I, J, V = _get_IJV_from_neighbors(c._vertices)
#     np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

#     # Save A1.txt
#     filename = os.path.join(work_dir, 'A1.txt')
#     I, J, V = _get_IJV_from_neighbors(c._edges)
#     np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

#     # Save A2.txt
#     filename = os.path.join(work_dir, 'A2.txt')
#     I, J, V = _get_IJV_from_neighbors(c._faces)
#     np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

#     # Save A3.txt
#     if c.dim == 3:
#         filename = os.path.join(work_dir, 'A3.txt')
#         I, J, V = _get_IJV_from_neighbors(c._polyhedra)
#         np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

# def save_B(
#         c,
#         work_dir: str = '.'):
#     """
#     """
#     if not os.path.exists(work_dir):
#         os.mkdir(work_dir)
#     # Save B1.txt
#     filename = os.path.join(work_dir, 'B1.txt')
#     I, J, V = _get_IJV_from_incidence(c._vertices)
#     np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

#     # Save B2.txt
#     filename = os.path.join(work_dir, 'B2.txt')
#     I, J, V = _get_IJV_from_incidence(c._edges)
#     np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

#     # Save B3.txt
#     if c.dim == 3:
#         filename = os.path.join(work_dir, 'B3.txt')
#         I, J, V = _get_IJV_from_incidence(c._faces)
#         np.savetxt(filename, [*zip(I, J, V)], fmt='%d')


# def _tri_area_2D(points_coo):
#     """
#     """
#     if len(points_coo) != 3:
#         raise ValueError('Not triangle')
#     points_coo = np.array(points_coo)
#     xs = points_coo[:, 0]
#     ys = points_coo[:, 1]
#     # S = xs[0] * (ys[1] - ys[2]) +\
#     #     xs[1] * (ys[2] - ys[0]) +\
#     #     xs[2] * (ys[0] - ys[1])
    
#     S = (xs[1] - xs[0]) * (ys[2] - ys[0]) -\
#         (xs[2] - xs[0]) * (ys[1] - ys[0])

#     return abs(S) / 2
    
# def face_area_2D(c, f_id):
#     """
#     """
#     v_ids = c.get_one('f', f_id).v_ids
#     vs = c.get_many('v', v_ids)
#     points = np.array([v.coord2D for v in vs])
#     d = Delaunay(points)
#     area = 0
#     for t in d.simplices:
#         area += _tri_area_2D(points[t])
#     return area

# def edge_length_2D(c, e_id):
#     """
#     change interface?
#     """
#     v_ids = c.get_one('e', e_id).v_ids
#     vs = c.get_many('v', v_ids)
#     points = np.array([v.coord2D for v in vs])
    
#     xs = points[:, 0]
#     ys = points[:, 1]

#     l2 = (xs[0] - xs[1])*(xs[0] - xs[1]) + (ys[0] - ys[1])*(ys[0] - ys[1])
#     return sqrt(l2)


# def metastability(S, Smax, Smin, Srand):
#     """
#     """    
#     if S >= Srand and Smax != Srand:
#         return (S - Srand) / (Smax - Srand)
#     elif S < Srand and Smin != Srand:
#         return (S - Srand) / (Srand - Smin)


# def S_rand(p):
#     """
#     """
#     jr0 = (1 - p)**3
#     jr1 = 3 * p * (1 - p)**2
#     jr2 = 3 * p**2 * (1 - p)
#     jr3 = p**3

#     return entropy(jr0, jr1, jr2, jr3)


# def get_vor_entropy(vor):
#     """
#     S = - sum(Pn * log(Pn))
#     """
#     regions = vor.regions
#     n_sides = np.array([len(r) for r in regions])
#     inner_regions = [-1 not in r and len(r) >= 3 for r in regions]
#     n_sides_inner = n_sides[inner_regions]
#     d = Counter(n_sides_inner)
#     N_int = len(n_sides_inner)

#     S = - np.array([d[k] / N_int * log2(d[k] / N_int) for k in d.keys()]).sum()

#     return S
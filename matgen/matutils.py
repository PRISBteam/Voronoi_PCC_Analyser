"""
"""
import math
from typing import Dict, Iterable, List, Tuple
import numpy as np
from scipy import sparse, linalg
import logging

# from matgen.base import Grain

def _get_IJV_from_neighbors(_cells: Dict) -> Tuple[List]:
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


def _get_IJV_from_incidence(_cells: Dict) -> Tuple[List]:
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


def get_d_tuple(j_tuple: Tuple) -> Tuple:
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


def ori_mat(ori: Tuple, oridesc: str ='euler-bunge:active') -> np.ndarray:
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


def _R_from_Rodrigues(t: float, u: Tuple) -> np.ndarray:
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


def dis_angle(g1: 'Grain', g2: 'Grain') -> float:
    """
    """
    if g1.crysym != g2.crysym:
        raise ValueError("Crysym of g1 and g2 don't match")
    R1 = ori_mat(g1.ori, g1.oridesc)
    R2 = ori_mat(g2.ori, g2.oridesc)
    return calculate_disorient(R1, R2, g1.crysym)


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
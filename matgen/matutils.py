"""
"""
import math
from typing import Dict, Iterable, List, Tuple
import numpy as np
from scipy import sparse
import logging

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


# def load_matrix_coo(f: open, matrix_shape=None) -> sparse.coo_matrix:
#     """
#     TODO: change behavior ? (delete "- 1")?
#     """
#     A_sparse = np.loadtxt(f, dtype='int')
#     I = np.array([row[0] for row in A_sparse]) - 1
#     J = np.array([row[1] for row in A_sparse]) - 1
#     V = np.array([row[2] for row in A_sparse])

#     A_coo = sparse.coo_matrix((V,(I,J)), shape=matrix_shape)

#     return A_coo


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

# def _get_IJV_from_neighbors(_cells: Dict) -> Tuple[List]:
#     """
#     index of an element is element_id - 1
#     """

#     I = []
#     J = []
#     V = []
#     for cell_id, cell in _cells.items():
#         for n_id in cell.n_ids:
#             I.append(cell_id - 1)
#             J.append(n_id - 1)
#             V.append(1)
    
#     return (I, J, V)

# def _get_IJV_from_incidence(_cells: Dict) -> Tuple[List]:
#     """
#     index of an element is element_id - 1
#     """

#     I = []
#     J = []
#     V = []
#     for cell_id, cell in _cells.items():
#         for inc_id in cell.incident_cells:
#             I.append(cell_id - 1)
#             J.append(inc_id - 1)
#             V.append(1)
    
#     return (I, J, V)


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



# def _ori_mat(ori, oridesc: str ='euler-bunge:active'):
#     """
#     """
    
#     if oridesc == 'euler-bunge:active':
#         f1 = radians(ori[0])
#         F = radians(ori[1])
#         f2 = radians(ori[2])
#     elif oridesc == 'euler-roe:active':
#         f1 = radians(ori[0] + 90)
#         F = radians(ori[1])
#         f2 = radians(ori[2] - 90)

#     #print(f1, F, f2)

#     g11 = cos(f1)*cos(f2) - sin(f1)*sin(f2)*cos(F)
#     #print(g11)
#     g12 = sin(f1)*cos(f2) + cos(f1)*sin(f2)*cos(F)
#     g13 = sin(f2)*sin(F)

#     g21 = - cos(f1)*sin(f2) - sin(f1)*cos(f2)*cos(F)
#     g22 = - sin(f1)*sin(f2) + cos(f1)*cos(f2)*cos(F)
#     g23 = cos(f2)*sin(F)

#     g31 = sin(f1)*sin(F)
#     g32 = - cos(f1)*sin(F)
#     g33 = cos(F)

#     g = np.array([
#         [g11, g12, g13],
#         [g21, g22, g23],
#         [g31, g32, g33]
#     ])

#     return g


# def calculate_theta_2D(c, e_id, crysym: str = 'cubic'):
#     """
#     симметрия кристалла имеет значение - какие еще бывают варианты?
#     """
#     f_ids = c.get_one('e', e_id).f_ids
#     if len(f_ids) == 1:
#         return -1.0
#     elif len(f_ids) == 2:
#         f1, f2 = c.get_many('f', f_ids)
#         g1 = _ori_mat(f1.ori, oridesc=f1.ori_format)
#         g2 = _ori_mat(f2.ori, oridesc=f2.ori_format)
        
#         _g = g1 @ np.linalg.inv(g2)
#         theta_min = acos((_g[0, 0] + _g[1, 1] + _g[2, 2] - 1) / 2)
        
#         if crysym == 'cubic': # TODO: добавить крисим в хар-ки комплекса
#             for Os1 in Osym:
#                 for Os2 in Osym:
#                     g = Os1 @ _g @ Os2
#                     theta = acos((g[0, 0] + g[1, 1] + g[2, 2] - 1) / 2)
#                     if theta < theta_min:
#                         theta_min = theta
       
#     return degrees(theta_min)



# Osym = np.array([
#     [
#         [ 1, 0, 0],
#         [ 0, 1, 0],
#         [ 0, 0, 1]
#     ],
#     [
#         [ 1, 0, 0],
#         [ 0, 0, -1],
#         [ 0, 1, 0]
#     ],
#     [
#         [ 1, 0, 0],
#         [ 0, -1, 0],
#         [ 0, 0, -1]
#     ],
#     [
#         [ 1, 0, 0],
#         [ 0, 0, 1],
#         [ 0, -1, 0]
#     ],
#     [
#         [ 0, -1, 0],
#         [ 1, 0, 0],
#         [ 0, 0, 1]
#     ],
#     [
#         [ 0, 0, 1],
#         [ 1, 0, 0],
#         [ 0, 1, 0]
#     ],
#     [
#         [ 0, 1, 0],
#         [ 1, 0, 0],
#         [ 0, 0, -1]
#     ],
#     [
#         [ 0, 0, -1],
#         [ 1, 0, 0],
#         [ 0, -1, 0]
#     ],
#     [
#         [ -1, 0, 0],
#         [ 0, -1, 0],
#         [ 0, 0, 1]
#     ],
#     [
#         [ -1, 0, 0],
#         [ 0, 0, -1],
#         [ 0, -1, 0]
#     ],
#     [
#         [ -1, 0, 0],
#         [ 0, 1, 0],
#         [ 0, 0, -1]
#     ],
#     [
#         [ -1, 0, 0],
#         [ 0, 0, 1],
#         [ 0, 1, 0]
#     ],
#     [
#         [ 0, 1, 0],
#         [ -1, 0, 0],
#         [ 0, 0, 1]
#     ],
#     [
#         [ 0, 0, 1],
#         [ -1, 0, 0],
#         [ 0, -1, 0]
#     ],
#     [
#         [ 0, -1, 0],
#         [ -1, 0, 0],
#         [ 0, 0, -1]
#     ],
#     [
#         [ 0, 0, -1],
#         [ -1, 0, 0],
#         [ 0, 1, 0]
#     ],
#     [
#         [ 0, 0, -1],
#         [ 0, 1, 0],
#         [ 1, 0, 0]
#     ],
#     [
#         [ 0, 1, 0],
#         [ 0, 0, 1],
#         [ 1, 0, 0]
#     ],
#     [
#         [ 0, 0, 1],
#         [ 0, -1, 0],
#         [ 1, 0, 0]
#     ],
#     [
#         [ 0, -1, 0],
#         [ 0, 0, -1],
#         [ 1, 0, 0]
#     ],
#     [
#         [ 0, 0, -1],
#         [ 0, -1, 0],
#         [ -1, 0, 0]
#     ],
#     [
#         [ 0, -1, 0],
#         [ 0, 0, 1],
#         [ -1, 0, 0]
#     ],
#     [
#         [ 0, 0, 1],
#         [ 0, 1, 0],
#         [ -1, 0, 0]
#     ],
#     [
#         [ 0, 1, 0],
#         [ 0, 0, -1],
#         [ -1, 0, 0]
#     ]
# ])


# def entropy(*args):
#     """
#     S
#     """
#     # input arguments may be a scalar, a tuple or several scalars
#     if len(args) == 1 and isinstance(args[0], Iterable):
#         j_array = np.array(args[0])
#     else:
#         j_array = np.array(args)

#     # check sum of input parameters
#     if len(j_array) > 1 and not isclose(j_array.sum(), 1):
#         logging.warning('Sum is not equal to 1')

#     # calculate entropy
#     if len(j_array) == 1:
#         p = j_array[0]
#         if p == 0 or p == 1:
#             return 0
#         elif p > 0 and p < 1:
#             return - (p*log2(p) + (1 - p)*log2(1 - p))
#     elif len(j_array) > 1:
#         nonzeros = j_array[j_array > 0]
#         return - np.sum(nonzeros * np.log2(nonzeros))


# def entropy_m(*args):
#     """
#     mean part of S
#     """
#     # input arguments may be a scalar, a tuple or several scalars
#     if len(args) == 1 and isinstance(args[0], Iterable):
#         j_array = np.array(args[0])
#     else:
#         j_array = np.array(args)

#     # check sum of input parameters
#     if len(j_array) > 1 and not isclose(j_array.sum(), 1):
#         logging.warning('Sum is not equal to 1')

#     # check zero elements
#     if np.any(j_array == 0):
#         logging.warning('One or more j is equal to 0')
#         return np.inf

#     # calculate mean entropy
#     if len(j_array) == 1:
#         p = j_array[0]
#         if p == 1:
#             return np.inf
#         elif p > 0 and p < 1:
#             return - log2(p * (1 - p)) / 2
#     elif len(j_array) > 1:
#         return - np.log2(np.prod(j_array)) / len(j_array)


# def entropy_s(*args):
#     """
#     deviatoric part of S
#     """
#     # input arguments may be a scalar, a tuple or several scalars
#     if len(args) == 1 and isinstance(args[0], Iterable):
#         j_array = np.array(args[0])
#     else:
#         j_array = np.array(args)

#     # check sum of input parameters
#     if len(j_array) > 1 and not isclose(j_array.sum(), 1):
#         logging.warning('Sum is not equal to 1')

#     # check zero elements
#     if np.any(j_array == 0):
#         logging.warning('One or more j is equal to 0')
#         return - np.inf
    
#     # calculate deviatoric entropy
#     if len(j_array) == 1:
#         p = j_array[0]
#         if p == 1:
#             return - np.inf
#         elif p > 0 and p < 1:
#             q = 1 - p
#             return - (p - q) / 2 * log2(p / q)
#     elif len(j_array) > 1:
#         Ss = 0
#         for k in range(len(j_array)):
#             jk = j_array[k]
#             for l in range(k + 1, len(j_array)):
#                 jl = j_array[l]
#                 Ss += (jk - jl) * log2(jk / jl)
#         Ss = Ss / len(j_array)
#         return - Ss


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


# def get_entropy(c):
#     """
#     добавить проверку != 0
#     """
#     S = 0
#     for jtype in range(4):
#         j = c.get_j_fraction(jtype)
#         if j != 0:
#             S -= j * log2(j)
#     return S

# def get_m_entropy(c):
#     """
#     добавить проверку != 0
#     """
#     Sm = 1
#     for jtype in range(4):
#         j = c.get_j_fraction(jtype)
#         if j != 0:
#             Sm *= j
#     Sm = log2(Sm) / 4

#     return Sm

# def get_s_entropy(c):
#     """
#     добавить проверку != 0
#     """
#     Ss = 0
#     for k in range(4):
#         jk = c.get_j_fraction(k)
#         if jk != 0:
#             for l in range(k, 4):
#                 jl = c.get_j_fraction(l)
#                 if jl != 0:
#                     Ss += (jk - jl) * log2(jk / jl)
#     Ss = Ss / 4

#     return Ss

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
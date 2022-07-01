"""
https://networkx.org/documentation/stable/tutorial.html

https://docs.scipy.org/doc/scipy/reference/sparse.html
https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html#module-scipy.sparse.csgraph
https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg
"""

import os
from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import sparse
import networkx as nx


def load_matrix_coo(f: open, matrix_shape=None) -> sparse.coo_matrix:
    """
    TODO: change behavior ? (delete "- 1")?
    """
    A_sparse = np.loadtxt(f, dtype='int')
    I = np.array([row[0] for row in A_sparse]) - 1
    J = np.array([row[1] for row in A_sparse]) - 1
    V = np.array([row[2] for row in A_sparse])

    A_coo = sparse.coo_matrix((V,(I,J)), shape=matrix_shape)

    return A_coo


def get_graph_from_A_file(f: open) -> nx.Graph:
    """
    A - adjacency matrix
    """
    A_sparse = np.loadtxt(f, dtype='int')
    IJ = [(row[0], row[1]) for row in A_sparse]
    G = nx.Graph()
    G.add_edges_from(IJ)

    return G

def get_graph_from_A_file(f: open) -> nx.Graph:
    """
    A - adjacency matrix
    """
    A_sparse = np.loadtxt(f, dtype='int')
    IJ = [(row[0], row[1]) for row in A_sparse]
    G = nx.Graph()
    G.add_edges_from(IJ)

    return G

def _get_IJV_from_neighbors(_cells: Dict) -> Tuple[List]:
    """
    index of an element is element_id - 1
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
    """
    index of an element is element_id - 1
    """

    I = []
    J = []
    V = []
    for cell_id, cell in _cells.items():
        for inc_id in cell.incident_cells:
            I.append(cell_id - 1)
            J.append(inc_id - 1)
            V.append(1)
    
    return (I, J, V)


def get_G_from_cells(_cells: Dict):
    """
    from Adj
    """
    I, J, _ = _get_IJV_from_neighbors(_cells)
    IJ = [(i + 1, j + 1) for i, j in zip(I, J)]
    G = nx.Graph()
    G.add_edges_from(IJ)
    
    return G


def get_A_from_cells(_cells: Dict):
    """
    """
    I, J, V = _get_IJV_from_neighbors(_cells)
    A_coo = sparse.coo_matrix((V,(I,J)))
    return A_coo

def get_B_from_cells(_cells: Dict):
    """
    """
    I, J, V = _get_IJV_from_incidence(_cells)
    B_coo = sparse.coo_matrix((V,(I,J)))
    return B_coo


def save_A(
        c,
        work_dir: str = '.'):
    """
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    # Save A0.txt
    filename = os.path.join(work_dir, 'A0.txt')
    I, J, V = _get_IJV_from_neighbors(c._vertices)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

    # Save A1.txt
    filename = os.path.join(work_dir, 'A1.txt')
    I, J, V = _get_IJV_from_neighbors(c._edges)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

    # Save A2.txt
    filename = os.path.join(work_dir, 'A2.txt')
    I, J, V = _get_IJV_from_neighbors(c._faces)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

    # Save A3.txt
    if c.dim == 3:
        filename = os.path.join(work_dir, 'A3.txt')
        I, J, V = _get_IJV_from_neighbors(c._polyhedra)
        np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

def save_B(
        c,
        work_dir: str = '.'):
    """
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    # Save B1.txt
    filename = os.path.join(work_dir, 'B1.txt')
    I, J, V = _get_IJV_from_incidence(c._vertices)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

    # Save B2.txt
    filename = os.path.join(work_dir, 'B2.txt')
    I, J, V = _get_IJV_from_incidence(c._edges)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

    # Save B3.txt
    if c.dim == 3:
        filename = os.path.join(work_dir, 'B3.txt')
        I, J, V = _get_IJV_from_incidence(c._faces)
        np.savetxt(filename, [*zip(I, J, V)], fmt='%d')

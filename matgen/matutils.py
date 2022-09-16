"""
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy import sparse

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
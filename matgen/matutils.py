"""
"""
from typing import Dict, List, Tuple


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
        which have `incident_ids` attribute.
    
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
        for inc_id in cell.incident_ids:
            I.append(cell_id - 1)
            J.append(inc_id - 1)
            V.append(1)
    
    return (I, J, V)
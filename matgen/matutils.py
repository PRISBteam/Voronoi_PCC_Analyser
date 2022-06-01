"""
https://networkx.org/documentation/stable/tutorial.html

https://docs.scipy.org/doc/scipy/reference/sparse.html
https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html#module-scipy.sparse.csgraph
https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg
"""

import numpy as np
from scipy import sparse
import networkx as nx

def load_matrix_coo(f: open, matrix_shape=None) -> sparse.coo_matrix:
    """
    """
    A_sparse = np.loadtxt(f, dtype='int')
    I = np.array([row[0] for row in A_sparse]) - 1
    J = np.array([row[1] for row in A_sparse]) - 1
    V = np.array([row[2] for row in A_sparse])

    A_coo = sparse.coo_matrix((V,(I,J)), shape=matrix_shape)

    return A_coo


def get_graph_from_A(f: open) -> nx.Graph:
    """
    A - adjacency matrix
    """
    A_sparse = np.loadtxt(f, dtype='int')
    IJ = [(row[0], row[1]) for row in A_sparse]
    G = nx.Graph()
    G.add_edges_from(IJ)

    return G

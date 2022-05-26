"""
dtype='int'

calculate degrees another way (feasible with big arrays)
"""

import argparse
import os
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import seaborn

def construct_dense_matrix(f, matrix_shape=None):
    """
    """
    A_sparse = np.loadtxt(f, dtype='int')
    I = np.array([row[0] for row in A_sparse]) - 1
    J = np.array([row[1] for row in A_sparse]) - 1
    V = np.array([row[2] for row in A_sparse])

    A_dense = sparse.coo_matrix((V,(I,J)), shape=matrix_shape).toarray()

    return A_dense


def construct_degree_matrix(A_dense):
    """
    """
    D = np.diag(np.sum(A_dense, axis=0))

    return D


def main() -> None:
    start = time.perf_counter_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", 
        help="Filepath to file with matrix in COO sparse format (I, J, V)"
    )
    parser.add_argument(
        "-s", "--matrix-shape", dest='matrix_shape',
        help="Shape of the matrix, i.e '(8,8)' or 8,8"
    )
    args = parser.parse_args()
 
    with open(args.filepath, "r", encoding="utf-8") as f:
        if args.matrix_shape:
            matrix_shape = tuple(map(int, args.matrix_shape.lstrip('(').rstrip(')').split(',')))
            print(matrix_shape)
        else:
            matrix_shape=None
        A_dense = construct_dense_matrix(f, matrix_shape)
        D = construct_degree_matrix(A_dense)
        np.savetxt(args.filepath.rstrip('.out') + '_dense.out', A_dense, fmt='%d')
        np.savetxt(args.filepath.rstrip('.out') + '_degree.out', D, fmt='%d')
        seaborn.heatmap(A_dense, cmap='Reds')
        plt.savefig(args.filepath.rstrip('.out') + '_dense.png')

        print('Time elapsed:', (time.perf_counter_ns() - start) / 1000000, 'ms')


if __name__ == "__main__":
    main()

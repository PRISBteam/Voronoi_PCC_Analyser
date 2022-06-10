"""Module to create sparse matrices from Neper output.

Arguments:
---------
filepath
    Filepath to .tess file with a tesselation (Neper output)
destination_folder
    Path to folder which will contain generated matrices. In case
    there is no need for a folder, type "."
-o, optional
    Use orientation (negative edges and faces) or not.

Functions
-------

"""

import argparse
from ast import Return
import os
import time
from typing import Dict, TextIO
import numpy as np

def extract_seeds(        
        filename: str = 'complex.tess',
        directory: str = '.') -> None:
    """
    """
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**cell' in line:
                n = int(f.readline().rstrip('\n').lstrip())
            if '*seed' in line:
                with open(os.path.join(directory, 'seeds.txt'), 'w') as f_seeds:
                    for i in range(n):
                        row = f.readline().split()
                        f_seeds.write(' '.join(row[1:4]) + '\n')
                return


def _write_matrices(
        arcs: np.array,
        node: str,
        is_signed: bool,
        d: Dict,
        A_out: TextIO,
        B_out: TextIO) -> Dict:
    """
    """
    for arc in arcs:
        _arc = abs(arc)
        if is_signed and arc < 0:
            B_out.write(str(_arc) + ' ' + node + ' -1\n')
        else:
            B_out.write(str(_arc) + ' ' + node + ' 1\n')
        if _arc in d.keys():
            A_out.write(d[_arc] + ' ' + node + ' 1\n')
            A_out.write(node + ' ' + d[_arc] + ' 1\n')
        else:
            d[_arc] = node

    return d


def write_matrices(
        filename: str = 'complex.tess',
        directory: str = '.',
        is_signed: bool = False) -> None:
    """Write A and B matrices from a .tess file.

    Writes following matrices:
    A0 - adjacency matrix for 0-cells (vertices)
    A1 - adjacency matrix for 1-cells (edges)
    A2 - adjacency matrix for 2-cells (faces)
    A3 - adjacency matrix for 3-cells (polyhedra)

    B01 - incidence matrix (0-cells are row indexes, 1-cells are columns)
    B12 - incidence matrix (1-cells are row indexes, 2-cells are columns)
    B23 - incidence matrix (2-cells are row indexes, 3-cells are columns)

    All matrices are in sparse COO format, i.e. each row of the file contains
    three elements - i (matrix row index), j (matrix column index), v (value).
    Indexes start from 1.

    In 2D case there are no A3 and B32 matrices.

    Matrices are stored in *.txt files.

    Number of cells are stored in 'number_of_cells.txt' file.
    Each row corresponds to number of cells: first row is number of 0-cells,
    second is number of 1-cells and so on.

    Components of a normal vector for each face are stored in 'Normal.txt'.
    Format: face_id a b c

    Parameters
    ----------
    filename

    directory

    is_signed

    Returns
    -------
    None
    """

    nc_filename = 'number_of_cells.txt'
    euler_c = 0
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**vertex' in line:
                n = int(f.readline().rstrip('\n').lstrip())
                euler_c += n
                with open(os.path.join(directory, nc_filename), 'w') as f_n:
                    f_n.write(str(n) + '\n')
            if '**edge' in line:
                n = int(f.readline().rstrip('\n').lstrip())
                euler_c -= n
                with open(os.path.join(directory, nc_filename), 'a') as f_n:
                    f_n.write(str(n) + '\n')
                d = {}
                with open(os.path.join(directory, 'B1.txt'), 'w') as B_out,\
                        open(os.path.join(directory, 'A1.txt'), 'w') as A_out,\
                        open(os.path.join(directory, 'A0.txt'), 'w') as A0_out:
                    for i in range(n):
                        row = f.readline().rstrip('\n').lstrip().split()
                        A0_out.write(row[1] + ' ' + row[2] + ' 1\n')
                        A0_out.write(row[2] + ' ' + row[1] + ' 1\n')
                        arcs = np.array(row[1:-1], dtype=int)
                        node = row[0]
                        d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
            if '**face' in line:
                n = int(f.readline().rstrip('\n').lstrip())
                euler_c += n
                with open(os.path.join(directory, nc_filename), 'a') as f_n:
                    f_n.write(str(n) + '\n')
                d = {}
                with open(os.path.join(directory, 'B2.txt'), 'w') as B_out,\
                        open(os.path.join(directory, 'A2.txt'), 'w') as A_out,\
                        open(os.path.join(directory, 'normals.txt'), 'w') as N_out:
                    for i in range(n):
                        node = f.readline().split()[0]
                        row = f.readline().split()
                        face_normal = f.readline().split()
                        face_row = face_normal[1:]
                        _ = f.readline()
                        N_out.write(node + ' ' + ' '.join(face_row) + '\n')
                        arcs = np.array(row[1:], dtype=int)
                        d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
            if '**polyhedron' in line:
                n = int(f.readline().rstrip('\n').lstrip())
                euler_c -= n
                with open(os.path.join(directory, nc_filename), 'a') as f_n:
                    f_n.write(str(n) + '\n')
                d = {}
                with open(os.path.join(directory, 'B3.txt'), 'w') as B_out,\
                        open(os.path.join(directory, 'A3.txt'), 'w') as A_out:
                    for i in range(n):
                        row = f.readline().split()
                        node = row[0]
                        arcs = np.array(row[2:], dtype=int)
                        d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
            if '**domain' in line:
                break
        
    if not euler_c == 1:
        print('Euler characteristic is not equal to 1!')

def main() -> None:
    start = time.perf_counter_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default='complex.tess',
        help="Filepath to .tess file with a tesselation (Neper output)"
    )
    parser.add_argument(
        "--dir",
        default='.',
        help="Path to folder which will contain generated matrices"
    )
    parser.add_argument(
        "-o", dest='is_signed',
        action='store_true',
        help="Orientation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    extract_seeds(args.file, args.dir)
    write_matrices(args.file, args.dir, args.is_signed)
    
    print('Time elapsed:', (time.perf_counter_ns() - start) / 1000000, 'ms')


if __name__ == "__main__":
    main()

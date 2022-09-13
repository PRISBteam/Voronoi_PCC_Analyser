"""Stand-alone module to create sparse matrices from Neper output.

Examples

    $ python sparsemat.py --file filename.tess --dir my_dir

"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from matgen import base


def extract_seeds(
    filename: str,
    work_dir: str = '.'
) -> Dict:
    """Extract seeds from .tess file and save as seeds.txt.

    Parameters
    ----------
    filename
        Filepath to .tess file with a tesselation (Neper output).
    work_dir
        Path to folder which will contain a new file seeds.txt.

    Returns
    -------
    poly_seeds
        Dictionary of seeds form .tess file. Keys are grain ids,
        values - lists of three seed coordinates. If the complex
        is 2D, third coordinate (z) is equal to 0.

    """
    poly_seeds = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**cell' in line:
                n = int(f.readline().rstrip('\n').lstrip())
            if '*seed' in line:
                filename_s = os.path.join(work_dir, 'seeds.txt')
                with open(filename_s, 'w') as f_seeds:
                    for i in range(n):
                        row = f.readline().split()
                        f_seeds.write(' '.join(row[1:4]) + '\n')
                        poly_seeds[int(row[0])] = [*map(float, row[1:4])]
                return poly_seeds


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


def _save_A(_cells: Dict, filename: str) -> None:
    """Save adjacency matrix into a file.

    Parameters
    ----------
    _cells
        A dictionary of cells. Keys - cell ids, values - cell objects
        which have `n_ids` attribute.
    filename
        Filename of a new file with the adjacency matrix in sparse format.    
    """
    I, J, V = _get_IJV_from_neighbors(_cells)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')


def _save_B(_cells: Dict, filename: str):
    """Save incidence matrix into a file.

    Parameters
    ----------
    _cells
        A dictionary of cells. Keys - cell ids, values - cell objects
        which have `incident_ids` attribute.
    filename
        Filename of a new file with the incidence matrix in sparse format.
    """
    I, J, V = _get_IJV_from_incidence(_cells)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')


def parse_tess_file(filename) -> Tuple:
    """Parse .tess file and produce a tuple of cell dictionaries.

    Parameters
    ----------
    filename
        Filepath to .tess file with a tesselation (Neper output).
    
    Returns
    -------
    dim
        Complex dimension - can be 2 or 3.
    tuple
        A tuple of dictionaries with vertices, edges, faces (and
        polyhedra in 3D case).

    """
    _vertices = {}
    _edges = {}
    _faces = {}
    _polyhedra = {}
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            if '**general' in line:
                dim = int(file.readline().split()[0])
                if dim not in [2, 3]:
                    raise ValueError(
                        f'Dimension must be 2 or 3, not {dim}'
                    )
            if '**vertex' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    v_id = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    _vertices[v_id] = base.Vertex(v_id, x, y, z)
            
            if '**edge' in line:
                n = int(file.readline().rstrip('\n'))
                for _ in range(n):
                    row = file.readline().split()
                    e_id = int(row[0])
                    v1_id = int(row[1])
                    v2_id = int(row[2])
                    v_ids = [v1_id, v2_id]
                    _vertices[v1_id].add_incident_cell(e_id)
                    _vertices[v1_id].add_neighbor(v2_id)
                    _vertices[v2_id].add_incident_cell(e_id)
                    _vertices[v2_id].add_neighbor(v1_id)
                    _edges[e_id] = base.Edge(e_id, v_ids)
                # Add neighbors to edges from common vertices
                for v in _vertices.values():
                    for e_id in v.incident_ids:
                        s = set(v.incident_ids)
                        s.difference_update([e_id])
                        _edges[e_id].add_neighbors(list(s))
            
            if '**face' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    f_id = int(row[0])
                    v_ids = []
                    for k in range(2, int(row[1]) + 2):
                        v_ids.append(int(row[k]))
                    face = base.Face(f_id, v_ids)
                    
                    row = file.readline().split()
                    e_ids = []
                    for k in range(1, int(row[0]) + 1):
                        e_id = abs(int(row[k]))
                        e_ids.append(e_id)
                        _edges[e_id].add_incident_cell(f_id)
                    face.add_edges(e_ids)
                    
                    row = file.readline().split()
                    face.add_equation(
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    )
                    _ = file.readline()
                    
                    _faces[f_id] = face
                # Add neighbors to faces from common edges
                for e in _edges.values():
                    for f_id in e.incident_ids:
                        s = set(e.incident_ids)
                        s.difference_update([f_id])
                        _faces[f_id].add_neighbors(list(s))
            
            if '**polyhedron' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    p_id = int(row[0])
                    f_ids = []
                    v_ids = []
                    e_ids = []
                    for k in range(2, int(row[1]) + 2):
                        f_id = abs(int(row[k]))
                        f_ids.append(f_id)
                        v_ids += _faces[f_id].v_ids
                        e_ids += _faces[f_id].e_ids
                        _faces[f_id].add_incident_cell(p_id)
                    f_ids = list(set(f_ids))
                    poly = base.Poly(p_id, f_ids)
                    poly.add_vertices(v_ids)
                    poly.add_edges(e_ids)
                    _polyhedra[p_id] = poly
                # Add neighbors to polyhedra from common faces
                for f in _faces.values():
                    for p_id in f.incident_ids:
                        s = set(f.incident_ids)
                        s.difference_update([p_id])
                        _polyhedra[p_id].add_neighbors(list(s))
            if '**domain' in line:
                if dim == 2:
                    return dim, (_vertices, _edges, _faces)
                else:
                    return dim, (_vertices, _edges, _faces, _polyhedra)


def save_matrices(filename, work_dir: str = '.') -> None:
    """Save new files.

    A0 - adjacency matrix for 0-cells (vertices)
    A1 - adjacency matrix for 1-cells (edges)
    A2 - adjacency matrix for 2-cells (faces)
    A3 - adjacency matrix for 3-cells (polyhedra)

    B1 - incidence matrix (0-cells are row indexes, 1-cells are columns)
    B2 - incidence matrix (1-cells are row indexes, 2-cells are columns)
    B3 - incidence matrix (2-cells are row indexes, 3-cells are columns)

    In 2D case there are no A3 and B3 matrices.

    Matrices are stored in *.txt files.

    Number of cells are stored in 'number_of_cells.txt' file.
    Each row corresponds to number of cells: first row is number of 0-cells,
    second is number of 1-cells and so on.

    Components of a normal vector for each face are stored in 'normals.txt'.
    Format: face_id a b c
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    dim, cell_dicts = parse_tess_file(filename)

    if dim == 2:
        A_filenames = ['A0.txt', 'A1.txt', 'A2.txt']
        B_filenames = ['B1.txt', 'B2.txt']
    else:
        A_filenames = ['A0.txt', 'A1.txt', 'A2.txt', 'A3.txt']
        B_filenames = ['B1.txt', 'B2.txt', 'B3.txt']
    
    for A_filename, _cells in zip(A_filenames, cell_dicts):
        filename = os.path.join(work_dir, A_filename)
        _save_A(_cells, filename)

    for B_filename, _cells in zip(B_filenames, cell_dicts[:-1]):
        filename = os.path.join(work_dir, B_filename)
        _save_B(_cells, filename)

    nc_filename = os.path.join(work_dir, 'number_of_cells.txt')
    with open(nc_filename, 'w') as file:
        for _cells in cell_dicts:
            file.write(f'{len(_cells.keys())}\n')
        
    normals_filename = os.path.join(work_dir, 'normals.txt')
    _faces = cell_dicts[2]
    with open(normals_filename, 'w') as file:
        for face in _faces.values():
            file.write(f'{face.id} {face.a} {face.b} {face.c}\n')


def main() -> None:
    """
    """
    start = time.time()
    
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

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    _ = extract_seeds(args.file, args.dir)
    save_matrices(args.file, args.dir)

    print('Time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()

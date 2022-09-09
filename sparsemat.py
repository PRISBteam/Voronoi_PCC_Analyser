"""Stand-alone module to create sparse matrices from Neper output.

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
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from matgen import base


def extract_seeds(        
        filename: str,
        work_dir: str = '.') -> None:
    """
    """
    poly_seeds = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**cell' in line:
                n = int(f.readline().rstrip('\n').lstrip())
            if '*seed' in line:
                with open(os.path.join(work_dir, 'seeds.txt'), 'w') as f_seeds:
                    for i in range(n):
                        row = f.readline().split()
                        f_seeds.write(' '.join(row[1:4]) + '\n')
                        poly_seeds[int(row[0])] = [*map(float, row[1:4])]
                return poly_seeds


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
        for inc_id in cell.incident_ids:
            I.append(cell_id - 1)
            J.append(inc_id - 1)
            V.append(1)
    
    return (I, J, V)


def _save_A(_cells: Dict, filename: str):
    """
    """
    I, J, V = _get_IJV_from_neighbors(_cells)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')


def _save_B(_cells: Dict, filename: str):
    """
    """
    I, J, V = _get_IJV_from_incidence(_cells)
    np.savetxt(filename, [*zip(I, J, V)], fmt='%d')


# def save_A(
#         c,
#         work_dir: str = '.'):
#     """
#     """

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

def parse_tess_file(filename):
    """
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


def save_matrices(filename, work_dir: str = '.'):
    """
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

# def _write_matrices(
#         arcs: np.array,
#         node: str,
#         is_signed: bool,
#         d: Dict,
#         A_out,
#         B_out) -> Dict:
#     """
#     """
#     for arc in arcs:
#         _arc = abs(arc)
#         if is_signed and arc < 0:
#             B_out.write(str(_arc) + ' ' + node + ' -1\n')
#         else:
#             B_out.write(str(_arc) + ' ' + node + ' 1\n')
#         if _arc in d.keys():
#             A_out.write(d[_arc] + ' ' + node + ' 1\n')
#             A_out.write(node + ' ' + d[_arc] + ' 1\n')
#         else:
#             d[_arc] = node

#     return d


# def write_matrices(
#         filename: str,
#         directory: str = '.',
#         is_signed: bool = False) -> None:
#     """Write A and B matrices from a .tess file.

#     Writes following matrices:
#     A0 - adjacency matrix for 0-cells (vertices)
#     A1 - adjacency matrix for 1-cells (edges)
#     A2 - adjacency matrix for 2-cells (faces)
#     A3 - adjacency matrix for 3-cells (polyhedra)

#     B01 - incidence matrix (0-cells are row indexes, 1-cells are columns)
#     B12 - incidence matrix (1-cells are row indexes, 2-cells are columns)
#     B23 - incidence matrix (2-cells are row indexes, 3-cells are columns)

#     All matrices are in sparse COO format, i.e. each row of the file contains
#     three elements - i (matrix row index), j (matrix column index), v (value).
#     Indexes start from 1.

#     In 2D case there are no A3 and B32 matrices.

#     Matrices are stored in *.txt files.

#     Number of cells are stored in 'number_of_cells.txt' file.
#     Each row corresponds to number of cells: first row is number of 0-cells,
#     second is number of 1-cells and so on.

#     Components of a normal vector for each face are stored in 'Normal.txt'.
#     Format: face_id a b c

#     Parameters
#     ----------
#     filename

#     directory

#     is_signed

#     Returns
#     -------
#     None
#     """

#     nc_filename = 'number_of_cells.txt'
#     euler_c = 0
#     with open(filename, 'r', encoding="utf-8") as f:
#         for line in f:
#             if '**vertex' in line:
#                 n = int(f.readline().rstrip('\n').lstrip())
#                 euler_c += n
#                 with open(os.path.join(directory, nc_filename), 'w') as f_n:
#                     f_n.write(str(n) + '\n')
#             if '**edge' in line:
#                 n = int(f.readline().rstrip('\n').lstrip())
#                 euler_c -= n
#                 with open(os.path.join(directory, nc_filename), 'a') as f_n:
#                     f_n.write(str(n) + '\n')
#                 d = {}
#                 with open(os.path.join(directory, 'B1.txt'), 'w') as B_out,\
#                         open(os.path.join(directory, 'A1.txt'), 'w') as A_out,\
#                         open(os.path.join(directory, 'A0.txt'), 'w') as A0_out:
#                     for i in range(n):
#                         row = f.readline().rstrip('\n').lstrip().split()
#                         A0_out.write(row[1] + ' ' + row[2] + ' 1\n')
#                         A0_out.write(row[2] + ' ' + row[1] + ' 1\n')
#                         arcs = np.array(row[1:-1], dtype=int)
#                         node = row[0]
#                         d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
#             if '**face' in line:
#                 n = int(f.readline().rstrip('\n').lstrip())
#                 euler_c += n
#                 with open(os.path.join(directory, nc_filename), 'a') as f_n:
#                     f_n.write(str(n) + '\n')
#                 d = {}
#                 with open(os.path.join(directory, 'B2.txt'), 'w') as B_out,\
#                         open(os.path.join(directory, 'A2.txt'), 'w') as A_out,\
#                         open(os.path.join(directory, 'normals.txt'), 'w') as N_out:
#                     for i in range(n):
#                         node = f.readline().split()[0]
#                         row = f.readline().split()
#                         face_normal = f.readline().split()
#                         face_row = face_normal[1:]
#                         _ = f.readline()
#                         N_out.write(node + ' ' + ' '.join(face_row) + '\n')
#                         arcs = np.array(row[1:], dtype=int)
#                         d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
#             if '**polyhedron' in line:
#                 n = int(f.readline().rstrip('\n').lstrip())
#                 euler_c -= n
#                 with open(os.path.join(directory, nc_filename), 'a') as f_n:
#                     f_n.write(str(n) + '\n')
#                 d = {}
#                 with open(os.path.join(directory, 'B3.txt'), 'w') as B_out,\
#                         open(os.path.join(directory, 'A3.txt'), 'w') as A_out:
#                     for i in range(n):
#                         row = f.readline().split()
#                         node = row[0]
#                         arcs = np.array(row[2:], dtype=int)
#                         d = _write_matrices(arcs, node, is_signed, d, A_out, B_out)
#             if '**domain' in line:
#                 break
        
#     if not euler_c == 1:
#         print('Euler characteristic is not equal to 1!')

def main() -> None:
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
    # parser.add_argument(
    #     "-o", dest='is_signed',
    #     action='store_true',
    #     help="Orientation"
    # )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    _ = extract_seeds(args.file, args.dir)
    save_matrices(args.file, args.dir)

    # write_matrices(args.file, args.dir, args.is_signed) - wrong
    # c = core.CellComplex(filename=args.file)
    # c.save_into_files(work_dir=args.dir)

    print('Time elapsed:', time.time() - start, 's')


if __name__ == "__main__":
    main()

"""
Get cells from Neper .tess file in different formats.
1. Cells list representation stores a p-complex using p dictionaries.
The key of each dictionary corresponds to a k-cell and each value contains
the vertex list (0-cells) of the corresponding k-cell. 

Plotting https://likegeeks.com/3d-plotting-in-python/ 

"""

import numpy as np
from typing import Dict, List, Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_v_coordinates(filename: str) -> Dict:
    """
    Get coordinates of vertices from Neper .tess file in the form
    of a dictionary:
    {vertex_id: {'x': x_coord, 'y': y_coord, 'z': z_coord}}

    Parameters
    ----------
    filename
        Name of Neper file.

    Returns
    -------
    vertices
        Dictionary of vertices and their coordinates.
    """
    vertices = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**vertex' in line:
                n = int(f.readline().rstrip('\n'))
                for i in range(n):
                    row = f.readline().split()
                    v_id = int(row[0])
                    vertices[v_id] = {}
                    vertices[v_id]['x'] = float(row[1])
                    vertices[v_id]['y'] = float(row[2])
                    vertices[v_id]['z'] = float(row[3])
                return vertices


def get_edges_from_tess(filename: str) -> Dict:
    """
    Get edges of a complex from Neper .tess file in the form
    of a dictionary:
    {edge_id: [ver_1, ver_2]}

    Parameters
    ----------
    filename
        Name of Neper file.

    Returns
    -------
    edges
        Dictionary of edges and their vertices.
    """
    edges = {}
    with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**edge' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        e_id = int(row[0])
                        edges[e_id] = [int(row[1]), int(row[2])]
                    return edges


def get_faces_from_tess(filename: str) -> Dict:
    """
    Get faces of a complex from Neper .tess file in the form
    of a dictionary:
    {face_id: [ver_1, ver_2 ...]}

    Parameters
    ----------
    filename
        Name of Neper file.

    Returns
    -------
    faces
        Dictionary of faces and their vertices.
    """
    faces = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if '**face' in line:
                n = int(f.readline().rstrip('\n'))
                for i in range(n):
                    row = f.readline().split()
                    f_id = int(row[0])
                    faces[f_id] = []
                    for k in range(2, int(row[1]) + 2):
                        faces[f_id].append(int(row[k]))
                    _ = f.readline()
                    _ = f.readline()
                    _ = f.readline()
                return faces


def get_poly_from_tess(filename: str) -> Dict:
    """
    Get polyhedra of a complex from Neper .tess file in the form
    of a dictionary:
    {poly_id: [ver_1, ver_2 ...]}

    Parameters
    ----------
    filename
        Name of Neper file.

    Returns
    -------
    polyhedra
        Dictionary of polyhedra and their vertices.
    """
    polyhedra = {}
    faces = get_faces_from_tess(filename)
    with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**polyhedron' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        p_id = int(row[0])
                        polyhedra[p_id] = []
                        for k in range(2, int(row[1]) + 2):
                            polyhedra[p_id] += faces[abs(int(row[k]))]
                        polyhedra[p_id] = list(set(polyhedra[p_id]))
                    return polyhedra


def _from_dict_to_coord_tuples(
        vertices: Dict, 
        subset=[]) -> List[Tuple]:
    """
    """
    v_list = []
    keys_list = vertices.keys() if not subset else subset
    for v in keys_list:
        v_list.append(
            (vertices[v]['x'], 
             vertices[v]['y'], 
             vertices[v]['z'])
        )
    return v_list


def _transform_coord(
        vertices: Dict, 
        vertices_subset: List = []) -> Tuple[List]:
    """
    Transform coordinates of vertices in the form (xs, ys, zs),
    where xs, ys and zs are the lists of x-, y- and z-coordinates
    respectively.

    Parameters
    ----------
    vertices
        Dictionary of vertices and their coordinates.
    vertices_subset, optional
        List of vertices to transfrom. If not present,
        output will contain all the vertices from the dictionary.

    Returns
    -------
    xs
        List of x-coordinates of vertices.
    ys
        List of y-coordinates of vertices.
    zs
        List of z-coordinates of vertices.
    """
    xs = []
    ys = []
    zs = []
    if not vertices_subset:
        vertices_subset = vertices.keys()
    for v in vertices_subset:
        xs.append(vertices[v]['x'])
        ys.append(vertices[v]['y'])
        zs.append(vertices[v]['z'])
    return (xs, ys, zs)


def create_axis(figsize: Tuple = (8,8)):
    """
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    return fig, ax


def plot_points(
        ax: Axes,
        vertices: Dict,
        vert_id_list: List,
        color=None,
        label: List = None) -> None:
    """
    """
    xs, ys, zs = _transform_coord(vertices, vert_id_list)
    ax.scatter(xs, ys, zs, label=label, c=color)
    if label:
        ax.legend(loc="best")
    plt.show()

def plot_edge(
        ax: Axes,
        vertices: Dict,
        edge: List,
        color: str = None,
        label: List = None) -> None:
    """
    """
    x = np.linspace(vertices[edge[0]]['x'], vertices[edge[1]]['x'], 50)
    y = np.linspace(vertices[edge[0]]['y'], vertices[edge[1]]['y'], 50)
    z = np.linspace(vertices[edge[0]]['z'], vertices[edge[1]]['z'], 50)
    ax.plot(x, y, z, color=color, label=label)
    if label:
        ax.legend(loc="best")
    plt.show()

def plot_face(
        ax: Axes,
        vertices: Dict,
        face: List,
        color: str =None,
        alpha: float = 0.2) -> None:
    """
    """
    vl = _from_dict_to_coord_tuples(vertices, face)
    poly = Poly3DCollection([vl], alpha=alpha, color=color)
    ax.add_collection3d(poly)

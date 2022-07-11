"""
classes
TODO:
1. Add web interface
2. 

Change lists of incident etc. to sets?
"""
import os
from typing import Dict, Iterable, List, Tuple, Union
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#import representation
from matgen import matutils


def _create_ax(dim: int = 2, figsize: Tuple = (8,8)) -> Axes:
    """
    """
    if dim == 2:
        projection = None
        xlim = ylim = (-0.1, 1.1)
    elif dim == 3:
        projection = '3d'
        xlim = ylim = zlim = (0, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if dim == 3:
        ax.set_zlim(*zlim)
    return ax

class Vertex():
    """
    Class Vertex (0-cell).
    
    Attributes
    ----------
    id
        Identifier of a vertex.
    x
        x-coordinate.
    y
        y-coordinate.
    z
        z-coordinate.
    coord
        A tuple of x-, y-, z-coordinates.
    neighbors
        List of neighbouring vertices.
    e_ids
        List of edges which the vetrex incident to.
        
    Methods
    -------
    add_incident_edge
        
    add_incident_edges
        
    get_degree
        
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        """
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.n_ids = []
        self.e_ids = []
        self.is_external = False
        self.junction_type = None
        
    @classmethod
    def from_tess_file(cls, filename: str) -> Dict:
        """
        """      
        _vertices = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**vertex' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        v_id = int(row[0])
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        _vertices[v_id] = cls(v_id, x, y, z)
                    return _vertices
        
    def __str__(self) -> str:
        """
        """
        v_str = "Vertex(id=%d)" % self.id
        
        return v_str
    
    def __repr__(self) -> str:
        """
        """
        v_str = f"Vertex(id={self.id})"
        
        return v_str
    
    @property
    def incident_cells(self) -> List:
        """
        """
        return self.e_ids

    @property
    def coord(self) -> Tuple[float]:
        """
        """
        return (self.x, self.y, self.z)

    @property
    def coord2D(self) -> Tuple[float]:
        """
        """
        return (self.x, self.y)
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.n_ids.append(n_id)
        self.n_ids = list(set(self.n_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.n_ids += n_ids
        self.n_ids = list(set(self.n_ids))
        
    def add_incident_edge(self, e_id: int):
        """
        """
        self.e_ids.append(e_id)
        self.e_ids = list(set(self.e_ids))
        
    def add_incident_edges(self, e_ids: Iterable):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))
        
    def get_degree(self) -> int:
        """
        """
        return len(self.e_ids)

    def plot(
            self,
            dim: int = 2,
            ax: Axes = None,
            figsize: Tuple = (8,8),
            **kwargs) -> Axes:
        """
        """
        if not ax:
            ax = _create_ax(dim, figsize)
        if dim == 2:
            ax.scatter(self.x, self.y, **kwargs)
        elif dim == 3:
            ax.scatter(self.x, self.y, self.z, **kwargs)
        return ax
    
    def set_external(self, is_external: bool = True):
        """
        """
        self.is_external = is_external

    def set_junction_type(self, junction_type: int):
        """
        For 2D cell complex
        junction_type equals number of special incident cells
        """
        self.junction_type = junction_type


class Edge():
    """
    Class Edge (1-cell).
    
    Attributes
    ----------
    id
        Identifier of an edge.
    v_ids
        A list of two vertices (their ids) of the edge.
    neighbors
        List of neighbouring edges.
    incident_faces
        List of faces which the edge belongs to.
        
    Methods
    -------
    add_neighbor
        Add a neighboring edge (by id).
    add_neighbors
        Add a list of neighboring edges (by id).
    add_incident_face
    
    add_incident_faces
    
    set_orientation
    
    get_degree
    
    """
    def __init__(self, id: int, v_ids: Iterable):
        """
        """
        self.id = id
        self.v_ids = v_ids
        self.n_ids = []
        self.f_ids = []
        self.is_special = False
        self.is_external = False
        self.theta = None
        self.junction_type = None

    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _vertices: Dict = {},
            measure: bool = False,
            theta: bool = False,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        """
        _edges = {}
        # if incidence:
        #     v_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**edge' in line:
                    n = int(f.readline().rstrip('\n'))
                    for _ in range(n):
                        row = f.readline().split()
                        e_id = int(row[0])
                        v1_id = int(row[1])
                        v2_id = int(row[2])
                        v_ids = [v1_id, v2_id]
                        if _vertices:
                            _vertices[v1_id].add_incident_edge(e_id)
                            _vertices[v1_id].add_neighbor(v2_id)
                            _vertices[v2_id].add_incident_edge(e_id)
                            _vertices[v2_id].add_neighbor(v1_id)

                        # if incidence:
                        #     if v1_id in v_dict.keys():
                        #         v_dict[v1_id].append(e_id)
                        #     else:
                        #         v_dict[v1_id] = [e_id]
                        #     if v2_id in v_dict.keys():
                        #         v_dict[v2_id].append(e_id)
                        #     else:
                        #         v_dict[v2_id] = [e_id]

                        _edges[e_id] = cls(e_id, v_ids)

                    # if incidence:
                    #     for v_id, vertex in vertices.items():
                    #         vertex.add_incident_edges(v_dict[v_id])
                    break
        if measure:
            filename_m = filename.rstrip('.tess') + '.stedge'
            with open(filename_m, 'r', encoding="utf-8") as file:
                for line in file:
                    row = line.split()
                    e_id = int(row[0])
                    e_length = float(row[1])
                    _edges[e_id].set_length(e_length)
                    if theta:
                        e_theta = float(row[2])
                        _edges[e_id].set_theta(
                            e_theta, 
                            lower_thrd=lower_thrd,
                            upper_thrd=upper_thrd
                        )
        # Add neighbors to edges from common vertices
        for v in _vertices.values():
            for e_id in v.e_ids:
                s = set(v.e_ids)
                s.difference_update([e_id])
                _edges[e_id].add_neighbors(list(s))
        return _edges

    def __str__(self):
        """
        """
        e_str = "Edge(id=%d)" % self.id
        
        return e_str
    
    def __repr__(self):
        """
        """
        e_str = f"Edge(id={self.id})"
        
        return e_str
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.n_ids.append(n_id)
        self.n_ids = list(set(self.n_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.n_ids += n_ids
        self.n_ids = list(set(self.n_ids))
        
    @property
    def incident_cells(self):
        """
        """
        return self.f_ids

    def add_incident_face(self, f_id: int):
        """
        """
        self.f_ids.append(f_id)
        self.f_ids = list(set(self.f_ids))
        
    def add_incident_faces(self, f_ids: Iterable):
        """
        """
        self.f_ids += f_ids
        self.f_ids = list(set(self.f_ids))
    
    def get_degree(self):
        """
        triple junctions ?
        """
        return len(self.f_ids)

    def set_length(self, length: float):
        """
        """
        self.len = length

    def set_theta(
            self,
            theta: float,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        2D
        disorientation angle (in degrees)
        theta must be >= 0 ?
        for external cells theta = -1
        """
        if theta < 0:
            self.set_external(True, dim=2)
            return
        elif self.is_external:
            raise ValueError(f"External doesn't have theta (id={self.id})")
        
        self.theta = theta
        
        if lower_thrd and upper_thrd:
            if  theta >= lower_thrd and theta <= upper_thrd:
                self.set_special(True)
            else:
                self.set_special(False)
        elif lower_thrd:
            if  theta >= lower_thrd:
                self.set_special(True)
            else:
                self.set_special(False)
        elif upper_thrd:
            if  theta <= upper_thrd:
                self.set_special(True)
            else:
                self.set_special(False)


    def set_special(self, is_special: bool = True):
        """
        External cannot be set special
        """
        if is_special and not self.is_external:
            self.is_special = is_special
        elif not is_special:
            self.is_special = is_special
        elif self.is_external:
            raise ValueError('External cannot be set special')


    def set_external(self, is_external: bool = True, dim: int = 2):
        """
        When set external, it is set not special with theta = -1
        Default for edge dim = 2
        """
        self.is_external = is_external
        if dim == 2:
            if is_external:
                self.theta = -1
                self.set_special(False)
            elif self.theta == -1:
                self.theta = None

    def set_junction_type(self, junction_type: int):
        """
        For 3D cell complex
        ? check consistency: external has type None
        """
        self.junction_type = junction_type

    # def plot(
    #         self,
    #         dim: int = 2,
    #         ax: Axes = None,
    #         figsize: Tuple = (8,8),
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         ax = _create_ax(dim, figsize)
        
    #     v1, v2 = self.v_ids # won't work because ids not equal to objects
    #     x_space = np.linspace(v1.x, v2.x, 50)
    #     y_space = np.linspace(v1.y, v2.y, 50)
    #     if dim == 2:
    #         ax.plot(x_space, y_space, **kwargs)
    #     elif dim == 3:
    #         z_space = np.linspace(v1.z, v2.z, 50)
    #         ax.plot(x_space, y_space, z_space, **kwargs)
        
    #     return ax

    # def plot3D(self, ax: Axes = None):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_3D_axis()
    #     x1, x2 = [v.x for v in self.vertices]
    #     y1, y2 = [v.y for v in self.vertices]        
    #     z1, z2 = [v.z for v in self.vertices]

    #     x_space = np.linspace(x1, x2, 50)
    #     y_space = np.linspace(y1, y2, 50)
    #     z_space = np.linspace(z1, z2, 50)
        
    #     ax.plot(x_space, y_space, z_space)



class Face():
    """
    Class Face (2-cell).
    
    Attributes
    ----------
    id
        Identifier of a face.
    vertices
        A list of vertices of the face.
    neighbors
        List of neighbouring faces.
    incident_poly
        List of polyhedra which the face belongs to.
    is_special
        
    Methods
    -------
    add_neighbor
        Add a neighboring face (by id).
    add_neighbors
        Add a list of neighboring faces (by id).
    add_incident_poly
    
    add_incident_polys
    
    set_orientation
        
    get_degree
    
    get_type
    
    set_special
    
    """
    def __init__(self, id: int, v_ids: Iterable):
        """
        """
        self.id = id
        self.v_ids = v_ids
        self.e_ids = []
        self.n_ids = []
        self.p_ids = []
        self.is_special = False
        self.is_external = False
        self.theta = None
        
    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _edges: Dict = {},
            measure: bool = False,
            theta: bool = False,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        """
        _faces = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**face' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        f_id = int(row[0])
                        v_ids = []
                        for k in range(2, int(row[1]) + 2):
                            v_ids.append(int(row[k]))
                        face = cls(f_id, v_ids)
                        
                        row = f.readline().split()
                        e_ids = []
                        for k in range(1, int(row[0]) + 1):
                            e_id = abs(int(row[k]))
                            e_ids.append(e_id)
                            if _edges:
                                _edges[e_id].add_incident_face(f_id)
                        face.add_edges(e_ids)
                        
                        row = f.readline().split()
                        face.add_equation(
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3])
                        )
                        _ = f.readline()
                        
                        _faces[f_id] = face
                    break
        if measure:
            filename_m = filename.rstrip('.tess') + '.stface'
            with open(filename_m, 'r', encoding="utf-8") as file:
                for line in file:
                    row = line.split()
                    f_id = int(row[0])
                    f_area = float(row[1])
                    _faces[f_id].set_area(f_area)
                    if theta:
                        f_theta = float(row[2])
                        _faces[f_id].set_theta(
                            f_theta,
                            lower_thrd=lower_thrd,
                            upper_thrd=upper_thrd
                        )
        for e in _edges.values():
            for f_id in e.f_ids:
                s = set(e.f_ids)
                s.difference_update([f_id])
                _faces[f_id].add_neighbors(list(s))
        return _faces
        
    def __str__(self):
        """
        """
        f_str = "Face(id=%d)" % self.id
        
        return f_str
    
    def __repr__(self):
        """
        """
        f_str = f"Face(id={self.id})"
        
        return f_str
    
    @property
    def incident_cells(self):
        """
        """
        return self.p_ids
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.n_ids.append(n_id)
        self.n_ids = list(set(self.n_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.n_ids += n_ids
        self.n_ids = list(set(self.n_ids))
        
    def add_incident_poly(self, p_id: int):
        """
        """
        self.p_ids.append(p_id)
        self.p_ids = list(set(self.p_ids))
        
    def add_incident_polys(self, p_ids: Iterable):
        """
        """
        self.p_ids += p_ids
        self.p_ids = list(set(self.p_ids))
    
    def add_edges(self, e_ids: Iterable):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))
    
    def add_equation(self, d: float, a: float, b: float, c: float):
        """
        """
        self.d = d
        self.a = a
        self.b = b
        self.c = c
        self.normal = (a, b, c)
    
    def get_degree(self):
        """
        internal / external
        """
        return len(self.p_ids)
    
    def set_seed(self, seed_coord: Tuple):
        """
        """
        self.seed = seed_coord

    def set_area(self, area: float):
        """
        """
        self.area = area

    def set_theta(
            self,
            theta: float,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        3D
        disorientation angle (in degrees)
        theta must be >= 0 ?
        for external cells theta = -1
        """
        if theta < 0:
            self.set_external(True, dim=3)
            return
        elif self.is_external:
            raise ValueError(f"External doesn't have theta (id={self.id})")
        
        self.theta = theta
        
        if lower_thrd and upper_thrd:
            if  theta >= lower_thrd and theta <= upper_thrd:
                self.set_special(True)
            else:
                self.set_special(False)
        elif lower_thrd:
            if  theta >= lower_thrd:
                self.set_special(True)
            else:
                self.set_special(False)
        elif upper_thrd:
            if  theta <= upper_thrd:
                self.set_special(True)
            else:
                self.set_special(False)


    def set_special(self, is_special: bool = True):
        """
        External cannot be set special
        """
        if is_special and not self.is_external:
            self.is_special = is_special
        elif not is_special:
            self.is_special = is_special
        elif self.is_external:
            raise ValueError('External cannot be set special')
        # if is_special:
        #     self.set_external(False)
        #     if self.theta < 0:
        #         self.theta = None


    def set_external(self, is_external: bool = True, dim: int = 3):
        """
        When set external, it is set not special with theta = -1
        Default for face dim = 2
        """
        self.is_external = is_external
        if dim == 3:
            if is_external:
                self.theta = -1
                self.set_special(False)
            elif self.theta == -1:
                self.theta = None


class Poly():
    """
    Class Poly (3-cell).
    
    Attributes
    ----------
    id
        Identifier of a polyhedron.
    vertices
        A list of vertices of the polyhedron.
    neighbors
        List of neighbouring polyhedra.
    faces
        List of faces that are on the boundary of the polyhedron.
    seed
        
        
    Methods
    -------
    add_neighbor
        Add a neighboring polyhedron (by id).
    add_neighbors
        Add a list of neighboring polyhedra (by id).
    set_orientation
        
    get_degree
    """
    def __init__(self, id: int, f_ids: Iterable):
        """
        """
        self.id = id
        self.v_ids = []
        self.e_ids = []
        self.n_ids = []
        self.f_ids = f_ids
        self.is_external = False

    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _faces: Dict,
            measure: bool = False):
        """
        """
        seeds = {}
        ori = {}
        _polyhedra = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**cell' in line:
                    N = int(f.readline().rstrip('\n'))
                if '*seed' in line:
                    for i in range(N):
                        row = f.readline().split()
                        seeds[int(row[0])] = tuple([*map(float, row[1:4])])
                if '*ori' in line:
                    ori_format = f.readline().strip() #.rstrip('\n')
                    for i in range(N):
                        row = f.readline().split()
                        ori[i + 1] = tuple([*map(float, row)])
                if '**polyhedron' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        p_id = int(row[0])
                        f_ids = []
                        v_ids = []
                        e_ids = []
                        for k in range(2, int(row[1]) + 2):
                            f_id = abs(int(row[k]))
                            f_ids.append(f_id)
                            v_ids += _faces[f_id].v_ids
                            e_ids += _faces[f_id].e_ids
                            _faces[f_id].add_incident_poly(p_id)
                        f_ids = list(set(f_ids))
                        poly = cls(p_id, f_ids)
                        poly.v_ids += v_ids
                        poly.v_ids = list(set(poly.v_ids))
                        poly.add_edges(e_ids)
                        poly.set_crystal_ori(ori_format, ori[p_id])
                        poly.set_seed(seeds[p_id])
                        _polyhedra[p_id] = poly
                    break
        if measure:
            filename_m = filename.rstrip('.tess') + '.stpoly'
            with open(filename_m, 'r', encoding="utf-8") as file:
                for line in file:
                    row = line.split()
                    p_id = int(row[0])
                    p_vol = float(row[1])
                    _polyhedra[p_id].set_volume(p_vol)
        for f in _faces.values():
            for p_id in f.p_ids:
                s = set(f.p_ids)
                s.difference_update([p_id])
                _polyhedra[p_id].add_neighbors(list(s))
        return _polyhedra
        
    def __str__(self):
        """
        """
        p_str = "Poly(id=%d)" % self.id
        
        return p_str
    
    def __repr__(self):
        """
        """
        p_str = f"Poly(id={self.id})"
        
        return p_str
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.n_ids.append(n_id)
        self.n_ids = list(set(self.n_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.n_ids += n_ids
        self.n_ids = list(set(self.n_ids))
        
    def add_face(self, f_id: int):
        """
        """
        self.f_ids.append(f_id)
        self.f_ids = list(set(self.f_ids))
        
    def add_faces(self, f_ids: Iterable):
        """
        """
        self.f_ids += f_ids
        self.f_ids = list(set(self.f_ids))
    
    def add_edges(self, e_ids: Iterable):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))
    
    
    def set_seed(self, seed_coord: Tuple):
        """
        """
        self.seed = seed_coord
    
    def set_crystal_ori(self, ori_format: str, ori_components: Tuple):
        """
        """
        self.ori_format = ori_format
        self.ori = ori_components

    def set_volume(self, volume: float):
        """
        """
        self.vol = volume

    def set_external(self, is_external: bool = True):
        """
        """
        self.is_external = is_external

class CellComplex():
    """
    Class CellComplex.
    A class for cell complex.

    Attributes
    ----------

    Methods
    -------

    """
    # Initializes a cell complex from Neper .tess file.
    def __init__(
            self,
            filename: str = 'complex.tess',
            measures: bool = False,
            theta: bool = False,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        """
        start = time.time()
        self.source_file = filename
        self.measures = measures # has measures or not

        # Define the cell complex dimension (can be 2 or 3)
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**general' in line:
                    self.dim = int(f.readline().split()[0])
                    if self.dim not in [2, 3]:
                        raise ValueError(
                            f'Dimension must be 2 or 3, not {self.dim}'
                        )
                    break
        
        # In 2D case edges have misorintation angles (theta)
        # In 3D case faces have misorintation angles (theta)
        if self.dim == 2:
            theta2D = theta
            theta3D = False
        elif self.dim ==3:
            theta2D = False
            theta3D = theta

        # A dictionary
        self._vertices = Vertex.from_tess_file(filename)
        # A list
        self.vertices = [v for v in self._vertices.values()]
        
        # print(len(self._vertices.keys()), 'vertices loaded:',
        #     time.time() - start, 's')
        # A dictionary
        self._edges = Edge.from_tess_file(
            filename = filename, 
            _vertices = self._vertices,
            measure=measures,
            theta=theta2D,
            lower_thrd=lower_thrd,
            upper_thrd=upper_thrd
        )
        # A list
        self.edges = [e for e in self._edges.values()]
        
        # print(len(self._edges.keys()),'edges loaded:',
        #     time.time() - start, 's')
        
        # Add neighbors to edges from common vertices
        # for v in self.vertices:
        #     for e_id in v.e_ids:
        #         s = set(v.e_ids)
        #         s.difference_update([e_id])
        #         self._edges[e_id].add_neighbors(list(s))

        # print('neighbor edges found',
        #     time.time() - start, 's')        
        # A dictionary
        self._faces = Face.from_tess_file(
            filename = filename,
            _edges=self._edges,
            measure=measures,
            theta=theta3D,
            lower_thrd=lower_thrd,
            upper_thrd=upper_thrd
        )
        # A list
        self.faces = [f for f in self._faces.values()]
        
        # print(len(self._faces.keys()), 'faces loaded:',
        #     time.time() - start, 's')
        # Add neighbors to faces from common edges
        # for e in self.edges:
        #     for f_id in e.f_ids:
        #         s = set(e.f_ids)
        #         s.difference_update([f_id])
        #         self._faces[f_id].add_neighbors(list(s))
        
        # print('neighbor faces found',
        #     time.time() - start, 's') 
        
        # In 2D case faces have seeds and orientations
        if self.dim == 2:
            with open(filename, 'r', encoding="utf-8") as file:
                for line in file:
                    if '**cell' in line:
                        N = int(file.readline().rstrip('\n'))
                    if '*seed' in line:
                        for i in range(N):
                            row = file.readline().split()
                            f_id = int(row[0])
                            seed_coord = tuple([*map(float, row[1:3])])
                            self._faces[f_id].set_seed(seed_coord)
                        break
                    # TODO: add ori in 2D case
            
            # Set external edges and vertices
            for e in self.edges:
                if len(e.f_ids) == 1:
                    e.set_external(dim=self.dim)
                    for v_id in e.v_ids:
                        self._vertices[v_id].set_external()
                    for f_id in e.f_ids:
                        self._faces[f_id].set_external(dim=self.dim)
        
        # In 3D there are polyhedra, that have seeds and orientations
        elif self.dim == 3:
            # A dictionary
            self._polyhedra = Poly.from_tess_file(
                filename,
                self._faces,
                measure=measures
            )
            # A list
            self.polyhedra = [p for p in self._polyhedra.values()]
            
            # Add neighbors to polyhedra from common faces
            # for f in self.faces:
            #     for p_id in f.p_ids:
            #         s = set(f.p_ids)
            #         s.difference_update([p_id])
            #         self._polyhedra[p_id].add_neighbors(list(s))
            
            # print('neighbor polyhedra found',
            #     time.time() - start, 's')

            # Set external faces, edges and vertices
            for f in self.faces:
                if len(f.p_ids) == 1:
                    f.set_external(dim=self.dim)
                    for v_id in f.v_ids:
                        self._vertices[v_id].set_external()
                    for e_id in f.e_ids:
                        self._edges[e_id].set_external(dim=self.dim)
                    for p_id in f.p_ids:
                        self._polyhedra[p_id].set_external()
        
        # Set junction types from theta (if known from a file)
        # If lower or upper threshold are known or both
        if lower_thrd or upper_thrd:
            self.set_junction_types()

        self.load_time = round(time.time() - start, 1)
        print('Complex loaded:', self.load_time, 's')
    
    def __str__(self):
        """
        """
        cc_str = f"<class CellComplex> {self.dim}D" +\
        f"\n{self.vernb} vertices" + f"\n{self.edgenb} edges" +\
        f"\n{self.facenb} faces" 
        
        if self.dim == 3:
            cc_str += f"\n{self.polynb} polyhedra"

        return cc_str
    
    def __repr__(self) -> str:
        """
        """
        return self.__str__()

    @property
    def vernb(self):
        """
        """
        return len(self._vertices)

    @property
    def edgenb(self):
        """
        """
        return len(self._edges)

    @property
    def facenb(self):
        """
        """
        return len(self._faces)

    @property
    def polynb(self):
        """
        """
        return len(self._polyhedra)

    def _choose_cell_type(self, cell_type: Union[str, int]):
        """
        """
        if cell_type in ['v', 'vertex', 0, '0']:
            _cells = self._vertices
        elif cell_type in ['e', 'edge', 1, '1']:
            _cells = self._edges
        elif cell_type in ['f', 'face', 2, '2']:
            _cells = self._faces
        elif cell_type in ['p', 'poly', 3, '3']:
            _cells = self._polyhedra
        else:
            raise TypeError('Unknown cell type')
        
        return _cells
    
    def get_one(self, cell_type: Union[str, int], cell_id: int):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        
        return _cells[cell_id]
    
    def get_many(self, cell_type: Union[str, int], cell_ids: Iterable):
        """
        """
        _cells = self._choose_cell_type(cell_type)

        return [_cells[cell_id] for cell_id in cell_ids]

    def get_external_ids(self, cell_type: Union[str, int]):
        """
        """
        _cells = self._choose_cell_type(cell_type)

        return [cell.id for cell in _cells.values() if cell.is_external]

    def get_internal_ids(self, cell_type: Union[str, int]):
        """
        """
        _cells = self._choose_cell_type(cell_type)

        return [cell.id for cell in _cells.values() if not cell.is_external]

    def get_special_ids(self):
        """
        2D - edges can be special
        3D - faces can be special
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('edge')
        elif self.dim == 3:
            _cells = self._choose_cell_type('face')

        return [cell.id for cell in _cells.values() if cell.is_special]

    def get_junction_ids_of_type(self, junction_type: int) -> List:
        """
        2D - vertices are junctions
        3D - edges are junctions
        """
        
        if self.dim == 2:
            _cells = self._choose_cell_type('vertex')
        elif self.dim == 3:
            _cells = self._choose_cell_type('edge')

        junction_ids = []
        for cell in _cells.values():
            if cell.junction_type == junction_type:
                junction_ids.append(cell.id)

        return junction_ids

    def get_spec_fraction(self):
        """
        number_of_special / number_of_internal
        """
        n_spec = len(self.get_special_ids())
        if self.dim == 2:
            n_int = len(self.get_internal_ids('e'))
        elif self.dim == 3:
            n_int = len(self.get_internal_ids('f'))

        frac = n_spec / n_int

        return round(frac, 3)

    def get_ext_fraction(self, cell_type: Union[str, int]):
        """
        number_of_external / number_of_cells
        """
        n_ext = len(self.get_external_ids(cell_type))
        n_cells = len(self._choose_cell_type(cell_type))

        frac = n_ext / n_cells

        return round(frac, 3)

    def get_j_fraction(self, junction_type: int):
        """
        number_of_junction_of_type / number_of_internal
        """
        n_junc = len(self.get_junction_ids_of_type(junction_type))
        if self.dim == 2:
            n_int = len(self.get_internal_ids('v'))
        elif self.dim == 3:
            n_int = len(self.get_internal_ids('e'))

        frac = n_junc / n_int

        return round(frac, 3)
    
    def set_junction_types(self) -> None:
        """
        external has None junction type
        """
        # junction_types = {
        #     0: 'J0',
        #     1: 'J1',
        #     2: 'J2',
        #     3: 'J3',
        #     4: 'U'
        # }

        if self.dim == 2:
            for v in self.vertices:
                v.n_spec_edges = 0
                if v.is_external:
                    v.set_junction_type(None)
                else:
                    for e_id in v.e_ids:
                        if self._edges[e_id].is_special:
                            v.n_spec_edges += 1
                    if v.n_spec_edges > 3:
                        print(
                            f'{v} is incident to ' +
                            f'{v.n_spec_edges} special edges'
                        )
                    v.set_junction_type(v.n_spec_edges)
        elif self.dim == 3:
            for e in self.edges:
                e.n_spec_faces = 0
                if e.is_external:
                    e.set_junction_type(None)
                else:
                    for f_id in e.f_ids:
                        if self._faces[f_id].is_special:
                            e.n_spec_faces += 1
                    if e.n_spec_faces > 3:
                        print(
                            f'{e} is incident to ' +
                            f'{e.n_spec_faces} special faces'
                        )
                    e.set_junction_type(e.n_spec_faces)

    def reset_special(
            self,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('edge')
        elif self.dim == 3:
            _cells = self._choose_cell_type('face')

        for cell in _cells.values():
            if not cell.is_external:
                if lower_thrd and upper_thrd:
                    if  cell.theta >= lower_thrd and cell.theta <= upper_thrd:
                        cell.set_special(True)
                    else:
                        cell.set_special(False)
                elif lower_thrd:
                    if  cell.theta >= lower_thrd:
                        cell.set_special(True)
                    else:
                        cell.set_special(False)
                elif upper_thrd:
                    if  cell.theta <= upper_thrd:
                        cell.set_special(True)
                    else:
                        cell.set_special(False)
                else:
                    cell.set_special(False)
        self.set_junction_types()

    def plot_vertices(
            self,
            v_ids: List = [],
            ax: Axes = None,
            figsize: Tuple = (8,8),
            labels: bool = False,
            **kwargs):
        """
        """
        if not ax:
            ax = _create_ax(self.dim, figsize)
        if v_ids:
            v_list = self.get_many('v', v_ids)
        else:
            v_list = self.vertices
        for v in v_list:
            if labels:
                ax = v.plot(dim=self.dim, ax=ax, label=v.id, **kwargs)
            else:
                ax = v.plot(dim=self.dim, ax=ax, **kwargs)
        if labels:
            ax.legend(loc='best')
        return ax

    
    def plot_edges(
            self,
            e_ids: List = [],
            ax: Axes = None,
            figsize: Tuple = (8,8),
            labels: bool = False,
            **kwargs):
        """
        """
        if not ax:
            ax = _create_ax(self.dim, figsize)
        if e_ids:
            e_list = self.get_many('e', e_ids)
        else:
            e_list = self.edges
        for e in e_list:
            v1, v2 = self.get_many('v', e.v_ids)
            x_space = np.linspace(v1.x, v2.x, 50)
            y_space = np.linspace(v1.y, v2.y, 50)
            if self.dim == 2:
                if labels:
                    ax.plot(x_space, y_space, label=e.id, **kwargs)
                else:
                    ax.plot(x_space, y_space, **kwargs)
            elif self.dim == 3:
                z_space = np.linspace(v1.z, v2.z, 50)
                if labels:
                    ax.plot(x_space, y_space, z_space, label=e.id, **kwargs)
                else:
                    ax.plot(x_space, y_space, z_space, **kwargs)
        if labels:
            ax.legend(loc='best')
        return ax

    # def plot_vertices_3D(
    #         self,
    #         v_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_3D_axis()
    #     if v_ids:
    #         v_list = self.get_many('v', v_ids)
    #     else:
    #         v_list = self.vertices
    #     for v in v_list:
    #         ax = v.plot3D(ax, **kwargs)
        
    #     return ax

    # def plot_edges_3D(
    #         self,
    #         e_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     if not ax:
    #         fig, ax = representation.create_3D_axis()
    #     if e_ids:
    #         e_list = self.get_many('e', e_ids)
    #     else:
    #         e_list = self.edges
    #     for e in e_list:
    #         v1, v2 = self.get_many('v', e.v_ids)

    #         x_space = np.linspace(v1.x, v2.x, 50)
    #         y_space = np.linspace(v1.y, v2.y, 50)
    #         z_space = np.linspace(v1.z, v2.z, 50)
            
    #         ax.plot(x_space, y_space, z_space, **kwargs)
            
    #     return ax
    
    
    def plot_faces(
            self,
            f_ids: List = [],
            ax: Axes = None,
            figsize: Tuple = (8,8),
            labels: bool = False,
            **kwargs):
        """
        labels doesn't work with 3d faces
        """
        if not ax:
            ax = _create_ax(self.dim, figsize)
        if f_ids:
            f_list = self.get_many('f', f_ids)
        else:
            f_list = self.faces
        f_coord_list = []
        for f in f_list:
            v_list = self.get_many('v', f.v_ids)

            if self.dim == 2:
                xs = [v.x for v in v_list]
                ys = [v.y for v in v_list]
                if labels:
                    ax.fill(xs, ys, label=f.id, **kwargs)
                else:
                    ax.fill(xs, ys, **kwargs)
            elif self.dim == 3:
                coord_list = [v.coord for v in v_list]
                f_coord_list.append(coord_list)

        if self.dim == 3:
            poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
            ax.add_collection3d(poly)
        elif labels:
            ax.legend(loc='best')
        return ax

    # def plot_faces_3D(
    #         self,
    #         f_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_3D_axis()
    #     if f_ids:
    #         f_list = self.get_many('f', f_ids)
    #     else:
    #         f_list = self.faces
    #     f_coord_list = []
    #     for f in f_list:
    #         v_list = self.get_many('v', f.v_ids)
    #         coord_list = [v.coord for v in v_list]
    #         f_coord_list.append(coord_list)
    #     poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
    #     ax.add_collection3d(poly)

    #     return ax

    def plot_polyhedra(
            self,
            p_ids: List = [],
            ax: Axes = None,
            figsize: Tuple = (8,8),
            **kwargs):
        """
        """
        if not ax:
            ax = _create_ax(self.dim, figsize)
        if p_ids:
            p_list = self.get_many('p', p_ids)
        else:
            p_list = self.polyhedra
        
        for p in p_list:
            ax = self.plot_faces(f_ids=p.f_ids, ax=ax, **kwargs)

        return ax

    # def plot_vertices_2D(
    #         self,
    #         v_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_2D_axis()
    #     if v_ids:
    #         v_list = self.get_many('v', v_ids)
    #     else:
    #         v_list = self.vertices
    #     for v in v_list:
    #         ax = v.plot2D(ax, **kwargs)
        
    #     return ax

    # def plot_edges_2D(
    #         self,
    #         e_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     if not ax:
    #         fig, ax = representation.create_2D_axis()
    #     if e_ids:
    #         e_list = self.get_many('e', e_ids)
    #     else:
    #         e_list = self.edges
    #     for e in e_list:
    #         v1, v2 = self.get_many('v', e.v_ids)

    #         x_space = np.linspace(v1.x, v2.x, 50)
    #         y_space = np.linspace(v1.y, v2.y, 50)
            
    #         ax.plot(x_space, y_space, **kwargs)
            
    #     return ax


    # def plot_faces_2D(
    #         self,
    #         f_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_2D_axis()
    #     if f_ids:
    #         f_list = self.get_many('f', f_ids)
    #     else:
    #         f_list = self.faces
    #     for f in f_list:
    #         # _ = self.plot_edges_2D(f.e_ids, ax, **kwargs)
    #         v_list = self.get_many('v', f.v_ids)
    #         xs = [v.x for v in v_list]
    #         ys = [v.y for v in v_list]
    #         ax.fill(xs, ys, **kwargs)

    #     return ax

    #     
    #     for f in f_list:
    #         v_list = self.get_many('v', f.v_ids)
    #         coord_list = [v.coord for v in v_list]
    #         f_coord_list.append(coord_list)
    #     poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
    #     ax.add_collection3d(poly)

    def plot_seeds(
            self,
            cell_ids: List = [],
            ax: Axes = None,
            figsize: Tuple = (8,8),
            **kwargs):
        """
        """
        if not ax:
            ax = _create_ax(self.dim, figsize)
        if self.dim == 2 and cell_ids:
            cell_list = self.get_many('f', cell_ids)
        elif self.dim == 2 and not cell_ids:
            cell_list = self.faces
        elif self.dim == 3 and cell_ids:
            cell_list = self.get_many('p', cell_ids)
        elif self.dim == 3 and not cell_ids:
            cell_list = self.polyhedra
        
        xs = []
        ys = []
        if self.dim == 3:
            zs = []
        for cell in cell_list:
            if self.dim == 2:
                x, y = cell.seed
            elif self.dim == 3:
                x, y, z = cell.seed
                zs.append(z)
            xs.append(x)
            ys.append(y)
        if self.dim == 2:
            ax.scatter(xs, ys, **kwargs)
        elif self.dim == 3:
            ax.scatter(xs, ys, zs, **kwargs)
        return ax


    # def plot_seeds_2D(
    #         self,
    #         f_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         ax = plt.subplot(111)
    #         # ax.set_xlim(0, 1)
    #         # ax.set_ylim(0, 1)
    #     if f_ids:
    #         f_list = self.get_many('f', f_ids)
    #     else:
    #         f_list = self.faces
        
    #     xs = []
    #     ys = []
    #     for f in f_list:
    #         x, y = f.seed
    #         xs.append(x)
    #         ys.append(y)
    #     ax.scatter(xs, ys, **kwargs)
    #     return ax

    # def plot_seeds_3D(
    #         self,
    #         p_ids: List = [],
    #         ax: Axes = None,
    #         **kwargs):
    #     """
    #     """
    #     if not ax:
    #         fig, ax = representation.create_2D_axis()
    #     if p_ids:
    #         p_list = self.get_many('p', p_ids)
    #     else:
    #         p_list = self.polyhedra
        
    #     xs = []
    #     ys = []
    #     zs = []
    #     for p in p_list:
    #         x, y, z = p.seed
    #         xs.append(x)
    #         ys.append(y)
    #         zs.append(z)
    #     ax.scatter(xs, ys, zs, **kwargs)
    #     return ax
  
    def get_sparse_A(self, cell_type: Union[str, int]):
        """
        """
        _cells = self._choose_cell_type(cell_type)
   
        return matutils.get_A_from_cells(_cells)

    def get_sparse_B(self, cell_type: Union[str, int]):
        """
        """
        _cells = self._choose_cell_type(cell_type)
   
        return matutils.get_B_from_cells(_cells)

    def get_graph_from_A(self, cell_type: Union[str, int]):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        
        return matutils.get_G_from_cells(_cells)

    def save_to_files(self, work_dir: str = '.', representation=None):
        """
        Can be saved as a set of srapse matrices or cell lists.
        representation: operator form, cells list, neper format?
        A, B
        """
        matutils.save_A(self, work_dir)
        matutils.save_B(self, work_dir)

        nc_filename = os.path.join(work_dir, 'number_of_cells.txt')
        with open(nc_filename, 'w') as file:
            file.write(f'{self.vernb}\n{self.edgenb}\n{self.facenb}')
            if self.dim == 3:
                file.write(f'\n{self.polynb}')
        
        normals_filename = os.path.join(work_dir, 'normals.txt')
        with open(normals_filename, 'w') as file:
            for face in self.faces:
                file.write(f'{face.id} {face.a} {face.b} {face.c}\n')

        seeds_filename = os.path.join(work_dir, 'seeds.txt')
        with open(seeds_filename, 'w') as file:
            if self.dim == 2:
                for face in self.faces:
                    file.write('%.12f %.12f 0.000000000000\n' % face.seed) 
            elif self.dim == 3:
                for poly in self.polyhedra:
                    file.write('%.12f %.12f %.12f\n' % poly.seed) 
  

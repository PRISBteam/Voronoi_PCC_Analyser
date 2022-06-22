"""
classes
- check time
- optimize

"""
from typing import Dict, Iterable, List, Union
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import representation
import matutils

# def get_element_by_id(seq, id):
#     """
#     """
#     for el in seq:
#         if el.id == id:
#             return el

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
    def __init__(self, id: int, x: float, y: float, z: float):
        """
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.coord = (x, y, z) # make property ?
        self.coord2D = (x, y)  # make property ?
        self.neighbor_ids = []
        self.e_ids = []
        
    @classmethod
    def from_tess_file(cls, filename):
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
        
    def __str__(self):
        """
        """
        v_str = "Vertex(id=%d)" % self.id
        
        return v_str
    
    def __repr__(self):
        """
        """
        v_str = f"Vertex(id={self.id})"
        
        return v_str
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.neighbor_ids.append(n_id)
        self.neighbor_ids = list(set(self.neighbor_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.neighbor_ids += n_ids
        self.neighbor_ids = list(set(self.neighbor_ids))
        
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
        
    def get_degree(self):
        """
        """
        return len(self.e_ids)


    def plot3D(self, ax: Axes = None, **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_3D_axis()
        ax.scatter(self.x, self.y, self.z, **kwargs)
        return ax
    
    def plot2D(self, ax: Axes = None, **kwargs):
        """
        """
        if not ax:
            ax = plt.subplot(111)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.scatter(self.x, self.y, **kwargs)
        return ax

    @property
    def incident_cells(self):
        """
        """
        return self.e_ids


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
        self.neighbor_ids = []
        self.f_ids = []

    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _vertices: Dict = {},
            measure: bool = False):
        """
        """
        _edges = {}
        # if incidence:
        #     v_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**edge' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        e_id = int(row[0])
                        v1_id = int(row[1])
                        v2_id = int(row[2])
                        v_ids = [v1_id, v2_id]
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
        self.neighbor_ids.append(n_id)
        self.neighbor_ids = list(set(self.neighbor_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.neighbor_ids += n_ids
        self.neighbor_ids = list(set(self.neighbor_ids))
        
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
    
    def set_orientation(self, orientation):
        """
        """
        pass

    def get_degree(self):
        """
        triple junctions ?
        """
        return len(self.f_ids)

    def set_length(self, length):
        """
        """
        self.len = length

    @property
    def incident_cells(self):
        """
        """
        return self.f_ids

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
        self.neighbor_ids = []
        self.p_ids = []
        self.is_special = False
        
    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _edges: List = [],
            measure: bool = False):
        """
        """
        _faces = {}
        # if incidence:
        #     e_dict = {}
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
                            _edges[e_id].add_incident_face(f_id)
                            # if incidence:
                            #     if e_id in e_dict.keys():
                            #         e_dict[e_id].append(f_id)
                            #     else:
                            #         e_dict[e_id] = [f_id]
                                # edge = get_element_by_id(edges, e_id)
                                # edge.add_incident_face(f_id)
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
                    
                    # if incidence:
                    #     for edge in edges:
                    #         edge.add_incident_faces(e_dict[edge.id])
                    
                    break
        
        if measure:
            filename_m = filename.rstrip('.tess') + '.stface'
            with open(filename_m, 'r', encoding="utf-8") as file:
                for line in file:
                    row = line.split()
                    f_id = int(row[0])
                    f_area = float(row[1])
                    _faces[f_id].set_area(f_area)

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
    
    def add_neighbor(self, n_id: int):
        """
        """
        self.neighbor_ids.append(n_id)
        self.neighbor_ids = list(set(self.neighbor_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.neighbor_ids += n_ids
        self.neighbor_ids = list(set(self.neighbor_ids))
        
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
    
    def add_edges(self, e_ids):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))
    
    def add_equation(self, d, a, b, c):
        """
        """
        self.d = d
        self.a = a
        self.b = b
        self.c = c
        self.normal = (a, b, c)
    
    def set_orientation(self, orientation):
        """
        """
        pass
    
    def get_degree(self):
        """
        internal / external
        """
        return len(self.p_ids)
    
    def get_type(self):
        """
        Check!!!
        """
        if len(self.p_ids) == 1:
            self.type = "external"
        elif len(self.p_ids) == 2:
            self.type = "internal"
        return self.type
    
    def set_special(self, is_special=True):
        """
        """
        self.is_special = is_special
    
    def set_seed2D(self, seed_coord):
        """
        """
        self.seed = seed_coord

    def set_area(self, area):
        """
        """
        self.area = area

    @property
    def incident_cells(self):
        """
        """
        return self.p_ids


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
        self.neighbor_ids = []
        self.f_ids = f_ids

    @classmethod
    def from_tess_file(
            cls,
            filename: str,
            _faces: List,
            measure: bool = False):
        """
        """
        seeds = {}
        ori = {}
        _polyhedra = {}
        # f_dict = {}
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
                            # if f_id in f_dict.keys():
                            #     f_dict[f_id].append(p_id)
                            # else:
                            #     f_dict[f_id] = [p_id]
                            _faces[f_id].add_incident_poly(p_id)
                        f_ids = list(set(f_ids))
                        poly = cls(p_id, f_ids)
                        poly.v_ids += v_ids
                        poly.v_ids = list(set(poly.v_ids))
                        poly.add_edges(e_ids)
                        poly.set_crystal_ori(ori_format, ori[p_id])
                        poly.set_seed(seeds[p_id])
                        _polyhedra[p_id] = poly
                    # f_v_dict = {}
                    # for face in faces:
                    #     face.add_incident_polys(f_dict[face.id])
                    #     f_v_dict[face.id] = face.v_ids
                    # for poly in polyhedra:
                    #     for f_id in poly.faces:
                    #         poly.v_ids += f_v_dict[f_id]
                    #     poly.v_ids = list(set(poly.v_ids))
                    break
        
        if measure:
            filename_m = filename.rstrip('.tess') + '.stpoly'
            with open(filename_m, 'r', encoding="utf-8") as file:
                for line in file:
                    row = line.split()
                    p_id = int(row[0])
                    p_vol = float(row[1])
                    _polyhedra[p_id].set_volume(p_vol)
        
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
        self.neighbor_ids.append(n_id)
        self.neighbor_ids = list(set(self.neighbor_ids))
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.neighbor_ids += n_ids
        self.neighbor_ids = list(set(self.neighbor_ids))
        
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
    
    def add_edges(self, e_ids):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))
    
    
    def set_orientation(self, orientation):
        """
        """
        pass
    
    # def get_degree(self):
    #     """
    #     """
    #     return len(self.f_ids)
    
    def set_seed(self, seed_coord):
        """
        """
        self.seed = seed_coord
    
    def set_crystal_ori(self, ori_format, ori_components):
        self.ori_format = ori_format
        self.ori = ori_components

    def set_volume(self, volume):
        """
        """
        self.vol = volume

class CellComplex():
    """
    """
    def __init__(
            self,
            filename: str = 'complex.tess',
            measures: bool = False):
        """
        """
        start = time.time()
        self.source_file = filename
        self.measures = measures

        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**general' in line:
                    self.dim = int(f.readline().split()[0])
                    break

        self._vertices = Vertex.from_tess_file(filename)
        # May be redundant, or make generator?
        self.vertices = [v for v in self._vertices.values()]
        
        print(len(self._vertices.keys()), 'vertices loaded:',
            time.time() - start, 's')
        
        self._edges = Edge.from_tess_file(
            filename = filename, 
            _vertices = self._vertices,
            measure=measures
        )
        # May be redundant, or make generator?
        self.edges = [e for e in self._edges.values()]
        
        print(len(self._edges.keys()),'edges loaded:',
            time.time() - start, 's')
        
        for v in self.vertices:
            for e_id in v.e_ids:
                s = set(v.e_ids)
                s.difference_update([e_id])
                self._edges[e_id].add_neighbors(list(s))

        print('neighbor edges found',
            time.time() - start, 's')        
        
        self._faces = Face.from_tess_file(
            filename = filename,
            _edges=self._edges,
            measure=measures
        )
        # May be redundant, or make generator?
        self.faces = [f for f in self._faces.values()]
        
        print(len(self._faces.keys()), 'faces loaded:',
            time.time() - start, 's')

        for e in self.edges:
            for f_id in e.f_ids:
                s = set(e.f_ids)
                s.difference_update([f_id])
                self._faces[f_id].add_neighbors(list(s))
        
        print('neighbor faces found',
            time.time() - start, 's') 
        
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
                            self._faces[f_id].set_seed2D(seed_coord)
                        break

        elif self.dim == 3:
            self._polyhedra = Poly.from_tess_file(
                filename,
                self._faces,
                measure=measures
            )
            # May be redundant, or make generator?
            self.polyhedra = [p for p in self._polyhedra.values()]
            
            print(len(self._polyhedra.keys()), 'poly loaded:',
                time.time() - start, 's')

        for f in self.faces:
            for p_id in f.p_ids:
                s = set(f.p_ids)
                s.difference_update([p_id])
                self._polyhedra[p_id].add_neighbors(list(s))
        
        print('neighbor polyhedra found',
            time.time() - start, 's') 

        # # переделать
        # for poly in self.polyhedra:
        #     p_edges = []
        #     p_faces = self.get_many('f', poly.faces)
        #     for p_face in p_faces:
        #         p_edges += p_face.edges
        #     poly.add_edges(list(set(p_edges)))

    
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
    
    def get_many(self, cell_type, cell_ids):
        """
        """
        _cells = self._choose_cell_type(cell_type)

        return [_cells[cell_id] for cell_id in cell_ids]
    
    def plot_vertices_3D(
            self,
            v_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_3D_axis()
        if v_ids:
            v_list = self.get_many('v', v_ids)
        else:
            v_list = self.vertices
        for v in v_list:
            ax = v.plot3D(ax, **kwargs)
        
        return ax

    def plot_edges_3D(
            self,
            e_ids: List = [],
            ax: Axes = None,
            **kwargs):
        if not ax:
            fig, ax = representation.create_3D_axis()
        if e_ids:
            e_list = self.get_many('e', e_ids)
        else:
            e_list = self.edges
        for e in e_list:
            v1, v2 = self.get_many('v', e.v_ids)

            x_space = np.linspace(v1.x, v2.x, 50)
            y_space = np.linspace(v1.y, v2.y, 50)
            z_space = np.linspace(v1.z, v2.z, 50)
            
            ax.plot(x_space, y_space, z_space, **kwargs)
            
        return ax


    def plot_faces_3D(
            self,
            f_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_3D_axis()
        if f_ids:
            f_list = self.get_many('f', f_ids)
        else:
            f_list = self.faces
        f_coord_list = []
        for f in f_list:
            v_list = self.get_many('v', f.v_ids)
            coord_list = [v.coord for v in v_list]
            f_coord_list.append(coord_list)
        poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
        ax.add_collection3d(poly)

        return ax

    def plot_polyhedra_3D(
            self,
            p_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_3D_axis()
        if p_ids:
            p_list = self.get_many('p', p_ids)
        else:
            p_list = self.polyhedra
        
        for p in p_list:
            _ = self.plot_faces_3D(p.f_ids, ax, **kwargs)

        return ax

    def plot_vertices_2D(
            self,
            v_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_2D_axis()
        if v_ids:
            v_list = self.get_many('v', v_ids)
        else:
            v_list = self.vertices
        for v in v_list:
            ax = v.plot2D(ax, **kwargs)
        
        return ax

    def plot_edges_2D(
            self,
            e_ids: List = [],
            ax: Axes = None,
            **kwargs):
        if not ax:
            fig, ax = representation.create_2D_axis()
        if e_ids:
            e_list = self.get_many('e', e_ids)
        else:
            e_list = self.edges
        for e in e_list:
            v1, v2 = self.get_many('v', e.v_ids)

            x_space = np.linspace(v1.x, v2.x, 50)
            y_space = np.linspace(v1.y, v2.y, 50)
            
            ax.plot(x_space, y_space, **kwargs)
            
        return ax


    def plot_faces_2D(
            self,
            f_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_2D_axis()
        if f_ids:
            f_list = self.get_many('f', f_ids)
        else:
            f_list = self.faces
        f_coord_list = []
        for f in f_list:
            # _ = self.plot_edges_2D(f.e_ids, ax, **kwargs)
            v_list = self.get_many('v', f.v_ids)
            xs = [v.x for v in v_list]
            ys = [v.y for v in v_list]
            ax.fill(xs, ys, **kwargs)

        return ax

    #     
    #     for f in f_list:
    #         v_list = self.get_many('v', f.v_ids)
    #         coord_list = [v.coord for v in v_list]
    #         f_coord_list.append(coord_list)
    #     poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
    #     ax.add_collection3d(poly)

    def plot_seeds_2D(
            self,
            f_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            ax = plt.subplot(111)
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
        if f_ids:
            f_list = self.get_many('f', f_ids)
        else:
            f_list = self.faces
        
        xs = []
        ys = []
        for f in f_list:
            x, y = f.seed
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, **kwargs)
        return ax

    def plot_seeds_3D(
            self,
            p_ids: List = [],
            ax: Axes = None,
            **kwargs):
        """
        """
        if not ax:
            fig, ax = representation.create_2D_axis()
        if p_ids:
            p_list = self.get_many('p', p_ids)
        else:
            p_list = self.polyhedra
        
        xs = []
        ys = []
        zs = []
        for p in p_list:
            x, y, z = p.seed
            xs.append(x)
            ys.append(y)
            zs.append(z)
        ax.scatter(xs, ys, zs, **kwargs)
        return ax
  
    def get_sparse_A(self, cell_type):
        """
        """
        _cells = self._choose_cell_type(cell_type)
   
        return matutils.get_A_from_cells(_cells)

    def get_sparse_B(self, cell_type):
        """
        """
        _cells = self._choose_cell_type(cell_type)
   
        return matutils.get_B_from_cells(_cells)

    def get_graph_from_A(self, cell_type):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        
        return matutils.get_G_from_cells(_cells)

    def save_into_files(self, representation=None, work_dir: str = '.'):
        """
        Can be saved as a set of srapse matrices or cell lists.
        representation: operator form, cells list, neper format?
        A, B
        """
        matutils.save_A(self, work_dir)
        matutils.save_B(self, work_dir)

    
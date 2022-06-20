"""
classes
- check time
- optimize
- incidence is time consuming: 
it may be worth to load data from file to sparse matrix format

"""
from typing import Iterable, List
import time

def get_element_by_id(seq, id):
    """
    """
    for el in seq:
        if el.id == id:
            return el

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
    incident_edges
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
        self.coord = (x, y, z)
        self.coord2D = (x, y)
        self.neighbors = []
        self.incident_edges = []
        
    @classmethod
    def from_file(cls, filename):
        """
        """
        vertices = []
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
                        vertices.append(cls(v_id, x, y, z))
                    return vertices

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
        self.neighbors.append(n_id)
        self.neighbors = list(set(self.neighbors))
    
    def add_neighbors(self, n_list: Iterable):
        """
        """
        self.neighbors += n_list
        self.neighbors = list(set(self.neighbors))
        
    def add_incident_edge(self, e_id: int):
        """
        """
        self.incident_edges.append(e_id)
        self.incident_edges = list(set(self.incident_edges))
        
    def add_incident_edges(self, e_list: Iterable):
        """
        """
        self.incident_edges += e_list
        self.incident_edges = list(set(self.incident_edges))
        
    def get_degree(self):
        """
        """
        return len(self.incident_edges)


class Edge():
    """
    Class Edge (1-cell).
    
    Attributes
    ----------
    id
        Identifier of an edge.
    vertices
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
    def __init__(self, id: int, v_list: Iterable):
        """
        """
        self.id = id
        self.vertices = v_list
        self.neighbors = []
        self.incident_faces = []

    @classmethod
    def from_file(
            cls,
            filename: str,
            vertices: List = [],
            incidence: bool = True):
        """
        """
        edges = []
        if incidence:
            v_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**edge' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        e_id = int(row[0])
                        v1_id = int(row[1])
                        v2_id = int(row[2])
                        v_list = [v1_id, v2_id]
                        if incidence:
                            if v1_id in v_dict.keys():
                                v_dict[v1_id].append(e_id)
                            else:
                                v_dict[v1_id] = [e_id]
                            if v2_id in v_dict.keys():
                                v_dict[v2_id].append(e_id)
                            else:
                                v_dict[v2_id] = [e_id]                            
                            # v1 = get_element_by_id(vertices, v1_id)
                            # v1.add_incident_edge(e_id)
                            # v2 = get_element_by_id(vertices, v2_id)
                            # v2.add_incident_edge(e_id)
                        edges.append(cls(e_id, v_list))

                    if incidence:
                        for vertex in vertices:
                            vertex.add_incident_edges(v_dict[vertex.id])
                    return edges

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
        self.neighbors.append(n_id)
        self.neighbors = list(set(self.neighbors))
    
    def add_neighbors(self, n_list: Iterable):
        """
        """
        self.neighbors += n_list
        self.neighbors = list(set(self.neighbors))
        
    def add_incident_face(self, f_id: int):
        """
        """
        self.incident_faces.append(f_id)
        self.incident_faces = list(set(self.incident_faces))
        
    def add_incident_faces(self, f_list: Iterable):
        """
        """
        self.incident_faces += f_list
        self.incident_faces = list(set(self.incident_faces))
    
    def set_orientation(self, orientation):
        """
        """
        pass

    def get_degree(self):
        """
        """
        return len(self.incident_faces)


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
    def __init__(self, id: int, v_list: Iterable):
        """
        """
        self.id = id
        self.vertices = v_list
        self.neighbors = []
        self.incident_poly = []
        self.is_special = False
        
    @classmethod
    def from_file(
            cls,
            filename: str,
            edges: List = [],
            incidence: bool = True):
        """
        """
        faces = []
        if incidence:
            e_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**face' in line:
                    n = int(f.readline().rstrip('\n'))
                    for i in range(n):
                        row = f.readline().split()
                        f_id = int(row[0])
                        v_list = []
                        for k in range(2, int(row[1]) + 2):
                            v_list.append(int(row[k]))
                        face = cls(f_id, v_list)
                        
                        row = f.readline().split()
                        e_list = []
                        for k in range(1, int(row[0]) + 1):
                            e_id = abs(int(row[k]))
                            e_list.append(e_id)
                            if incidence:
                                if e_id in e_dict.keys():
                                    e_dict[e_id].append(f_id)
                                else:
                                    e_dict[e_id] = [f_id]
                                # edge = get_element_by_id(edges, e_id)
                                # edge.add_incident_face(f_id)
                        face.add_edges(e_list)
                        
                        row = f.readline().split()
                        face.add_equation(
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3])
                        )
                        _ = f.readline()
                        
                        faces.append(face)
                    
                    if incidence:
                        for edge in edges:
                            edge.add_incident_faces(e_dict[edge.id])
                    
                    return faces
        
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
        self.neighbors.append(n_id)
        self.neighbors = list(set(self.neighbors))
    
    def add_neighbors(self, n_list: Iterable):
        """
        """
        self.neighbors += n_list
        self.neighbors = list(set(self.neighbors))
        
    def add_incident_poly(self, p_id: int):
        """
        """
        self.incident_poly.append(p_id)
        self.incident_poly = list(set(self.incident_poly))
        
    def add_incident_polys(self, p_list: Iterable):
        """
        """
        self.incident_poly += p_list
        self.incident_poly = list(set(self.incident_poly))
    
    def add_edges(self, e_list):
        """
        """
        self.edges = e_list
    
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
        """
        return len(self.incident_poly)
    
    def get_type(self):
        """
        Check!!!
        """
        if len(self.incident_poly) == 1:
            self.type = "external"
        elif len(self.incident_poly) == 2:
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
    def __init__(self, id: int, f_list: Iterable):
        """
        """
        self.id = id
        self.vertices = []
        self.neighbors = []
        self.faces = f_list

    @classmethod
    def from_file(
            cls,
            filename: str,
            faces: List):
        """
        """
        seeds = {}
        ori = {}
        polyhedra = []
        f_dict = {}
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
                        v_list = []
                        f_list = []
                        for k in range(2, int(row[1]) + 2):
                            f_id = abs(int(row[k]))
                            f_list.append(f_id)
                            if f_id in f_dict.keys():
                                f_dict[f_id].append(p_id)
                            else:
                                f_dict[f_id] = [p_id]
                            # face = get_element_by_id(faces, f_id)
                            # v_list += face.vertices
                            # face.add_incident_poly(p_id)
                        # v_list = list(set(v_list))
                        f_list = list(set(f_list))
                        poly = cls(p_id, f_list)
                        poly.set_crystal_ori(ori_format, ori[p_id])
                        poly.set_seed(seeds[p_id])
                        # poly.add_faces(f_list)
                        polyhedra.append(poly)

                    f_v_dict = {}
                    for face in faces:
                        face.add_incident_polys(f_dict[face.id])
                        f_v_dict[face.id] = face.vertices
                    for poly in polyhedra:
                        for f_id in poly.faces:
                            poly.vertices += f_v_dict[f_id]
                        poly.vertices = list(set(poly.vertices))
                    return polyhedra
        
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
        self.neighbors.append(n_id)
        self.neighbors = list(set(self.neighbors))
    
    def add_neighbors(self, n_list: Iterable):
        """
        """
        self.neighbors += n_list
        self.neighbors = list(set(self.neighbors))
        
    def add_face(self, f_id: int):
        """
        """
        self.faces.append(f_id)
        self.faces = list(set(self.faces))
        
    def add_faces(self, f_list: Iterable):
        """
        """
        self.faces += f_list
        self.faces = list(set(self.faces))
    
    def set_orientation(self, orientation):
        """
        """
        pass
    
    def get_degree(self):
        """
        """
        return len(self.faces)
    
    def set_seed(self, seed_coord):
        """
        """
        self.seed = seed_coord
    
    def set_crystal_ori(self, ori_format, ori_components):
        self.ori_format = ori_format
        self.ori = ori_components

class CellComplex():
    """
    """
    def __init__(
            self,
            filename: str = 'complex.tess',
            incidence: bool = True):
        """
        """
        start = time.time()
        self.source_file = filename

        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                if '**general' in line:
                    self.dim = int(f.readline().split()[0])
                    break

        self.vertices = Vertex.from_file(filename)
        
        print(len(self.vertices), 'vertices loaded:',
            time.time() - start, 's')
        
        self.edges = Edge.from_file(
            filename = filename, 
            vertices = self.vertices,
            incidence=incidence
        )
        
        print(len(self.edges),'edges loaded:', time.time() - start, 's')
        
        self.faces = Face.from_file(
            filename = filename,
            edges=self.edges,
            incidence=incidence)
        
        print(len(self.faces), 'faces loaded:', time.time() - start, 's')
        
        if self.dim == 2:
            seeds = {}
            with open(filename, 'r', encoding="utf-8") as f:
                for line in f:
                    if '**cell' in line:
                        N = int(f.readline().rstrip('\n'))
                    if '*seed' in line:
                        for i in range(N):
                            row = f.readline().split()
                            seeds[int(row[0])] = tuple([*map(float, row[1:3])])
                        break
            for face in self.faces:
                face.set_seed2D(seeds[face.id])
        elif self.dim == 3:
            self.polyhedra = Poly.from_file(filename, self.faces)
            print(len(self.polyhedra), 'poly loaded:',
                time.time() - start, 's')
    
    
    def get_one(self, cell_type: str, id: int):
        """
        """
        if cell_type == 'v' or cell_type == 'vertex':
            cell_list = self.vertices
        elif cell_type == 'e' or cell_type == 'edge':
            cell_list = self.edges
        elif cell_type == 'f' or cell_type == 'face':
            cell_list = self.faces
        elif cell_type == 'p' or cell_type == 'poly':
            cell_list = self.polyhedra
        else:
            raise TypeError('Unknown type cell')
        
        return get_element_by_id(seq=cell_list, id=id)
    
    def get_many(self, cell_type, ids):
        """
        """
        if cell_type == 'v' or cell_type == 'vertex':
            cell_list = self.vertices
        elif cell_type == 'e' or cell_type == 'edge':
            cell_list = self.edges
        elif cell_type == 'f' or cell_type == 'face':
            cell_list = self.faces
        elif cell_type == 'p' or cell_type == 'poly':
            cell_list = self.polyhedra
        else:
            raise TypeError('Unknown type cell')
        
        return [get_element_by_id(cell_list, id) for id in ids]
    
    
    def save_into_files(self, representation):
        """
        Can be saved as a set of srapse matrices or cell lists.
        representation: operator form, cells list, neper format?
        """
        pass

    
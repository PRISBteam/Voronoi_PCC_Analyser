"""Base classes for complex analysis.
"""
import io
from typing import Dict, Iterable, List, Tuple
from matgen import matutils
# import numpy as np


class Cell():
    """
    """
    def __init__(self, id: int):
        """
        """

        self.id = id
        self.n_ids = [] # neighbour ids
        self.is_external = False

    def __str__(self):
        """
        """
        cell_str = self.__class__.__name__ + "(id=%d)" % self.id
        
        return cell_str
    
    def __repr__(self) -> str:
        """
        """
        return self.__str__()

    def add_neighbor(self, n_id: int):
        """
        """
        if n_id not in self.n_ids:
            self.n_ids.append(n_id)
    
    def add_neighbors(self, n_ids: Iterable):
        """
        """
        self.n_ids += n_ids
        self.n_ids = list(set(self.n_ids))

    def set_external(self, is_external: bool = True):
        """
        """
        self.is_external = is_external

    def set_measure(self, measure: float):
        """
        """
        self.measure = measure


class CellLowerDim(Cell):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.signed_incident_ids = []

    # def __str__(self):
    #     """
    #     """
    #     cell_str = "CellLowerDim(id=%d)" % self.id
        
    #     return cell_str

    @property
    def incident_ids(self):
        """
        unsigned ids
        """
        return list(map(abs, self.signed_incident_ids))
    
    def add_incident_cell(self, signed_incident_id: int):
        """
        """
        if abs(signed_incident_id) not in self.incident_ids:
            self.signed_incident_ids.append(signed_incident_id)
    
    # def add_incident_cells(self, incident_ids: Iterable):
    #     """
    #     FIXME: incident_ids may be signed?
    #     """
    #     self.incident_ids += incident_ids
    #     self.incident_ids = list(set(self.incident_ids))

    def get_degree(self) -> int:
        """
        """
        return len(self.signed_incident_ids)


class TripleJunction(CellLowerDim):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.junction_type = None

    # def __str__(self):
    #     """
    #     """
    #     cell_str = "TripleJunction(id=%d)" % self.id
        
    #     return cell_str

    def set_junction_type(self, junction_type: int):
        """
        junction_type equals number of special incident cells
        check consistency: external has type None
        """
        if not self.is_external:
            self.junction_type = junction_type
        elif self.is_external:
            raise ValueError(
                'External junction cannot have a type other than None'
            )


class GrainBoundary(CellLowerDim):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.is_special = False
        self.theta = None

    # def __str__(self):
    #     """
    #     """
    #     cell_str = "GrainBoundary(id=%d)" % self.id
    #     return cell_str

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
    
    def set_theta(
            self,
            theta: float,
            lower_thrd: float = None,
            upper_thrd: float = None):
        """
        disorientation angle (in degrees)
        theta must be >= 0 ?
        for external cells theta = -1
        """
        if theta < 0:
            self.set_external(True)
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

    def set_external(self, is_external: bool = True):
        """
        When set external, it is set not special with theta = -1
        Default for edge dim = 2
        """
        self.is_external = is_external
        if is_external:
            self.theta = -1
            self.set_special(False)
        elif self.theta == -1:
            self.theta = None

    
class Vertex(CellLowerDim):
    """
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        super().__init__(id)
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_tess_file(cls, file: io.TextIOBase) -> Dict:
        """
        Note that file must be oper prior to calling this method
        Be careful with reading the same file with different methods
        """      
        _vertices = {}
        for line in file:
            if '**vertex' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    v_id = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    _vertices[v_id] = cls(v_id, x, y, z)
                return _vertices

    def __str__(self) -> str:
        """
        """
        v_str = self.__name__ + "(id=%d, x=%.3f, y=%.3f, z=%.3f)" % (
            self.id, self.x, self.y, self.z
        )
        return v_str

    @property
    def coord(self) -> Tuple[float]:
        """
        """
        return (self.x, self.y, self.z)


class Vertex2D(Vertex, TripleJunction):
    """
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        super().__init__(id, x, y, z)
        self.junction_type = None

    def __str__(self) -> str:
        """
        """
        v_str = self.__name__ + "(id=%d, x=%.3f, y=%.3f)" % (
            self.id, self.x, self.y
        )
        return v_str

    @property
    def coord(self) -> Tuple[float]:
        """
        """
        return (self.x, self.y)


class Edge(CellLowerDim):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id)
        self.v_ids = v_ids

    # def __str__(self):
    #     """
    #     """
    #     e_str = "Edge(id=%d)" % self.id
    #     return e_str

    @classmethod
    def from_tess_file(
            cls,
            file: io.TextIOBase,
            _vertices: Dict = {}):
        """
        """
        _edges = {}
        for line in file:
            if '**edge' in line:
                n = int(file.readline().rstrip('\n'))
                for _ in range(n):
                    row = file.readline().split()
                    e_id = int(row[0])
                    v1_id = int(row[1])
                    v2_id = int(row[2])
                    v_ids = [v1_id, v2_id]
                    if _vertices:
                        _vertices[v1_id].add_incident_edge(e_id)
                        _vertices[v1_id].add_neighbor(v2_id)
                        _vertices[v2_id].add_incident_edge(-e_id)
                        _vertices[v2_id].add_neighbor(v1_id)
                    _edges[e_id] = cls(e_id, v_ids)
                return _edges

    @property
    def len(self) -> float:
        """
        """
        try:
            return self.measure
        except AttributeError:
            return None 


class Edge3D(Edge, TripleJunction):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id, v_ids)
        self.junction_type = None

    # def __str__(self):
    #     """
    #     """
    #     e_str = "Edge3D(id=%d)" % self.id
    #     return e_str


class Edge2D(Edge, GrainBoundary):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id, v_ids)
        self.is_special = False

    # def __str__(self):
    #     """
    #     """
    #     e_str = "Edge2D(id=%d)" % self.id
    #     return e_str


class Face(CellLowerDim):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id)
        self.v_ids = v_ids
        self.e_ids = []

    # def __str__(self):
    #     """
    #     """
    #     f_str = "Face(id=%d)" % self.id
        
    #     return f_str

    def add_edge(self, e_id: int):
        """
        """
        if e_id not in self.e_ids:
            self.e_ids.append(e_id)

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

    @property
    def area(self) -> float:
        """
        """
        try:
            return self.measure
        except AttributeError:
            return None 


class Face3D(Face, GrainBoundary):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id, v_ids)
        self.is_special = False

    # def __str__(self):
    #     """
    #     """
    #     f_str = "Face3D(id=%d)" % self.id
    #     return f_str


class Poly(Cell):
    """
    """
    def __init__(self, id: int, f_ids: Iterable):
        super().__init__(id)
        self.v_ids = []
        self.e_ids = []
        self.f_ids = f_ids

    # def __str__(self):
    #     """
    #     """
    #     p_str = "Poly(id=%d)" % self.id
        
    #     return p_str

    def add_vertex(self, v_id: int):
        """
        """
        if v_id not in self.v_ids:
            self.v_ids.append(v_id)

    def add_vertices(self, v_ids: Iterable):
        """
        """
        self.v_ids += v_ids
        self.v_ids = list(set(self.v_ids))

    def add_edge(self, e_id: int):
        """
        """
        if e_id not in self.e_ids:
            self.e_ids.append(e_id)

    def add_edges(self, e_ids: Iterable):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))


class TripleJunctionSet():
    """
    """

    def __init__(self, p, j_tuple) -> None:
        """
        """
        self.p = p
        self.q = 1 - p
        self.p_entropy = matutils.entropy(p)
        self.p_entropy_m = matutils.entropy_m(p)
        self.p_entropy_s = matutils.entropy_s(p)
        self.j0, self.j1, self.j2, self.j3 = j_tuple
        self.p_expected = (self.j1 + 2*self.j2 + 3*self.j3) / 3
        self.delta_p = abs(self.p_expected - self.p)
        self.S = matutils.entropy(*j_tuple)
        self.S_m = matutils.entropy_m(*j_tuple)
        self.S_s = matutils.entropy_s(*j_tuple)
        self.kappa = self.S_m / self.S_s if self.S_s != 0 else 0
        self.delta_S = self.p_entropy - self.S
        self.d1, self.d2, self.d3 = matutils.get_d_tuple(j_tuple)

    
def _add_neighbors(_cells, _incident_cells):
    """
    Add neighbors to incident_cells from common cells
    _vertices, _edges
    _edges, _faces
    _faces, _polyhedra
    """ 
    for cell in _cells.values():
        for inc_cell_id in cell.incident_ids:
            s = set(cell.incident_ids)
            s.difference_update([inc_cell_id])
            _incident_cells[inc_cell_id].add_neighbors(list(s))

#TODO: measure and theta are loaded from files and then set to edges and etc 
            # measure: bool = False,
            # theta: bool = False,
            # lower_thrd: float = None,
            # upper_thrd: float = None
        
        # if measure:
        #     filename_m = filename.rstrip('.tess') + '.stedge'
        #     with open(filename_m, 'r', encoding="utf-8") as file:
        #         for line in file:
        #             row = line.split()
        #             e_id = int(row[0])
        #             e_length = float(row[1])
        #             _edges[e_id].set_length(e_length)
        #             if theta:
        #                 e_theta = float(row[2])
        #                 _edges[e_id].set_theta(
        #                     e_theta, 
        #                     lower_thrd=lower_thrd,
        #                     upper_thrd=upper_thrd
        #                 )        

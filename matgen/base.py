"""Base classes for complex analysis.
"""
from typing import Iterable, List
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

    def __str__(self):
        """
        """
        cell_str = "Cell(id=%d)" % self.id
        
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


class CellLowerDim(Cell):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.signed_incident_ids = []

    def __str__(self):
        """
        """
        cell_str = "CellLowerDim(id=%d)" % self.id
        
        return cell_str

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


class Vertex(CellLowerDim):
    """
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        super().__init__(id)
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        """
        """
        v_str = "Vertex(id=%d, x=%.3f, y=%.3f, z=%.3f)" % (
            self.id, self.x, self.y, self.z
        )
        return v_str


class Edge(CellLowerDim):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id)
        self.v_ids = v_ids

    def __str__(self):
        """
        """
        e_str = "Edge(id=%d)" % self.id
        
        return e_str
    

class Face(CellLowerDim):
    """
    """
    def __init__(self, id: int, v_ids: Iterable):
        super().__init__(id)
        self.v_ids = v_ids
        self.e_ids = []

    def __str__(self):
        """
        """
        f_str = "Face(id=%d)" % self.id
        
        return f_str

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


class Poly(Cell):
    """
    """
    def __init__(self, id: int, f_ids: Iterable):
        super().__init__(id)
        self.v_ids = []
        self.e_ids = []
        self.f_ids = f_ids

    def __str__(self):
        """
        """
        p_str = "Poly(id=%d)" % self.id
        
        return p_str

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
"""Base classes for complex analysis.

"""
from __future__ import annotations # to avoid NameError in type hinting
import io
import time
from typing import Dict, Iterable, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matgen import matutils


class Cell:
    """
    Cell class.

    Parameters
    ----------
    id : int
        Identifier of the cell.
        
    Attributes
    ----------
    id : int
        Identifier of the cell.
    n_ids : List
        A list of neighbouring cells identifiers.
    is_external : bool
        True if the cell is external, False if the cell is internal.
    measure : float, optional
        A measure of the cell. It can be created by using `set_measure`
        method.

    Methods
    -------
    add_neighbor(n_id)
    add_neighbors(n_ids)
    set_external(is_external=True)
    set_measure(measure)
    """

    def __init__(self, id: int):
        self.id = id
        self.n_ids = [] # neighbour ids
        # self.nn_ids = [] # neighbour of neighbor ids
        self.is_external = False

    def __str__(self):
        cell_str = self.__class__.__name__ + "(id=%d)" % self.id
        return cell_str
    
    def __repr__(self) -> str:
        return self.__str__()

    def add_neighbor(self, n_id: int) -> None:
        """Add a neighbour identifier to neighbouring cells list.

        Parameters
        ----------
        n_id
            A neighbour identifier.

        Notes
        -----
        Neighbouring cells list doesn't contain duplicates and the cell's
        own identifier.
        """
        if n_id not in self.n_ids and n_id != self.id:
            self.n_ids.append(n_id)
    
    def add_neighbors(self, n_ids: List) -> None:
        """Add a list of identifiers to neighbouring cells list.

        Parameters
        ----------
        n_ids
            A list of neighbours identifiers.

        Notes
        -----
        Neighbouring cells list doesn't contain duplicates and the cell's
        own identifier.
        """
        self.n_ids += n_ids
        s = set(self.n_ids) # eliminate duplicates
        s.difference_update([self.id]) # eliminate self.id
        self.n_ids = list(s)

    # def add_neighbors_neighbors(self, nn_ids: Iterable):
    #     """
    #     """
    #     self.nn_ids += nn_ids
    #     s = set(self.nn_ids) # eliminate duplicates
    #     s.difference_update([self.id] + self.n_ids) # eliminate id and n_ids
    #     self.nn_ids = list(s)

    def set_external(self, is_external: bool = True) -> None:
        """Make the cell external (is_external=True) or internal
        (is_external=False).

        Parameters
        ----------
        is_external
            True if the cell is external, False if the cell is internal.
        """
        self.is_external = is_external

    def set_measure(self, measure: float) -> None:
        """Add measure attribute to the cell.
        """
        self.measure = measure


class Grain(Cell):
    """
    Grain class.

    Parameters
    ----------
    id : int
        Identifier of the grain.
        
    Attributes
    ----------
    id : int
        Identifier of the grain.
    seed : Tuple
        A grain seed (x, y, z) coordinates.
    crysym: str
        Grain crystal symmetry (see Crystal Symmetries section of Neper
        documentation).
    oridesc: str
        A descriptor of used to parameterize the crystal orientations. See
        Rotations and Orientations section of Neper documentation for the
        list of available descriptors.
    ori : Tuple
        Grain crystal orientation components. They depends on the descriptor
        (`oridesc` attribute).
    R : np.ndarray, optional

    Methods
    -------
    set_crystal_ori(crysym, oridesc, ori_components)
    set_seed(seed_coord)
    dis_angle(other)
    """

    def __init__(self, id: int):
        super().__init__(id)
        self.seed = None
        self.oridesc = None
        self.ori = None
        self.crysym = None

    def set_crystal_ori(
        self,
        crysym: str,
        oridesc: str,
        ori_components: Tuple
    ) -> None:
        """Set grain crystal orientation parameters.
        """
        self.crysym = crysym
        self.oridesc = oridesc
        self.ori = ori_components

    def set_seed(self, seed_coord: Tuple) -> None:
        """Set grain seed (x, y, z) coordinates.
        """
        self.seed = seed_coord

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix associated with grain crystal orientation.
        """
        return matutils.ori_mat(self.ori, self.oridesc)
    
    def dis_angle(self, other: Grain) -> float:
        """Calculate disorientation angle between the grain and another one.
        """
        return matutils.dis_angle(self, other)


class LowerOrderCell(Cell):
    """
    Cells of order k = 0, 1, ..., p - 1 of a p-complex.

    Parameters
    ----------
    id : int
        Identifier of the cell.
    
    Attributes
    ----------
    id : int
        Identifier of the cell.
    signed_incident_ids: List
        A list of incident cells identifiers. Identifiers are signed: they
        are negative if orientations of the cell and incident cell differ.
    incident_ids: List
        A list of unsigned incident cells identifiers.
    degree: int
        Number of incident cells of the given cell. 

    Methods
    -------
    add_incident_cell(signed_incident_id)
    
    Notes
    -----
    All k-cells except for p-cells have incident (k + 1)-cells that can be
    of same or different orientation. If orientations are different, then
    this incident cell with different orientations will denoted with negative
    id in the incident cells list.

    References
    ---------
    ..  [1] Grady, Leo J., and Jonathan R. Polimeni. Discrete calculus: Applied
        analysis on graphs for computational science. Vol. 3. London: Springer, 2010.
        https://doi.org/10.1007/978-1-84996-290-2
    """
    
    def __init__(self, id: int):
        super().__init__(id)
        self.signed_incident_ids = []

    @property
    def incident_ids(self):
        """A list of unsigned incident cells identifiers.
        """
        return list(map(abs, self.signed_incident_ids))
    
    def add_incident_cell(self, signed_incident_id: int):
        """Add a signed incident cell identifier to signed incident cells
        list.
        """
        if abs(signed_incident_id) not in self.incident_ids:
            self.signed_incident_ids.append(signed_incident_id)
    
    # def add_incident_cells(self, incident_ids: Iterable):
    #     """
    #     FIXME: incident_ids may be signed?
    #     """
    #     self.incident_ids += incident_ids
    #     self.incident_ids = list(set(self.incident_ids))

    @property
    def degree(self) -> int:
        """Number of incident cells of the given cell.
        """
        return len(self.signed_incident_ids)


class TripleJunction(LowerOrderCell):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.junction_type = None

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


class GrainBoundary(LowerOrderCell):
    """
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.is_special = False
        self.theta = None

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
    
    def _reset_theta_thrds(
        self,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        """
        if not self.is_external and self.theta:
            if lower_thrd and upper_thrd:
                if  self.theta >= lower_thrd and self.theta <= upper_thrd:
                    self.set_special(True)
                else:
                    self.set_special(False)
            elif lower_thrd:
                if  self.theta >= lower_thrd:
                    self.set_special(True)
                else:
                    self.set_special(False)
            elif upper_thrd:
                if  self.theta <= upper_thrd:
                    self.set_special(True)
                else:
                    self.set_special(False)
            else:
                self.set_special(False)
    
    def set_theta(
        self,
        theta: float,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        disorientation angle (in degrees)
        theta must be >= 0 ?
        for external cells theta = -1
        TODO: check if theta is changed
        """
        if theta < 0:
            self.set_external(True)
            return
        elif self.is_external:
            raise ValueError(
                f"External (id={self.id}) doesn't have theta other than -1")
        
        self.theta = theta
        self._reset_theta_thrds(lower_thrd, upper_thrd)

    def set_external(self, is_external: bool = True):
        """
        When set external, it is set not special with theta = -1
        """
        self.is_external = is_external
        if is_external:
            self.theta = -1
            self.set_special(False)
        elif self.theta == -1:
            self.theta = None

    
def _create_ax(dim: int = 2, figsize: Tuple = (8,8)) -> Axes:
    """
    """
    if dim == 2:
        projection = None
        # xlim = ylim = (-0.1, 1.1)
    elif dim == 3:
        projection = '3d'
        # xlim = ylim = zlim = (0, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    # if dim == 3:
    #     ax.set_zlim(*zlim)
    return ax


class Vertex(LowerOrderCell):
    """
    Vertex (0-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the vertex.
    x: float
        Vertex x-coordinate.
    y: float
        Vertex y-coordinate.
    z: float, optional
        Vertex z-coordinate.
    
    Attributes
    ----------
    id : int
        Identifier of the vertex.
    x: float
        Vertex x-coordinate.
    y: float
        Vertex y-coordinate.
    z: float, optional
        Vertex z-coordinate.
    coord: Tuple[float]
        A tuple of x-, y-, z-coordinates.

    Methods
    -------
    from_tess_file(file)
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        super().__init__(id)
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_tess_file(cls, file: io.TextIOBase | str) -> Dict:
        """Extracts vertices from Neper .tess file and returns a dictionary
        of Vertex examples.

        Parameters
        ----------
        file
            Filename or file object.

        Returns
        -------
        _vertices
            A dictionary of vertices. Keys are vertex identifiers, values are
            Vertex examples.

        Notes
        -----
        Expects valid .tess file format (as for Neper 4.5.0). See details
        https://neper.info/doc/fileformat.html
        """      
        if isinstance(file, str):
            file = open(file, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!') 
        
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
        v_str = self.__class__.__name__ + "(id=%d, x=%.3f, y=%.3f, z=%.3f)" % (
            self.id, self.x, self.y, self.z
        )
        return v_str

    @property
    def coord(self) -> Tuple[float]:
        """A tuple of x-, y-, z-coordinates.
        """
        return (self.x, self.y, self.z)


class Vertex2D(Vertex, TripleJunction):
    """
    """
    def __init__(self, id: int, x: float, y: float, z: float = 0):
        super().__init__(id, x, y, z)
        self.junction_type = None

    def __str__(self) -> str:
        v_str = self.__class__.__name__ + "(id=%d, x=%.3f, y=%.3f)" % (
            self.id, self.x, self.y
        )
        return v_str

    @property
    def coord(self) -> Tuple[float]:
        """
        """
        return (self.x, self.y)

    def plot(
            self,
            ax: Axes = None,
            figsize: Tuple = (8,8),
            **kwargs) -> Axes:
        """
        """
        if not ax:
            ax = _create_ax(dim=2, figsize=figsize)
        ax.scatter(self.x, self.y, **kwargs)
        return ax


class Vertex3D(Vertex):
    """
    """
    def plot(
            self,
            ax: Axes = None,
            figsize: Tuple = (8,8),
            **kwargs) -> Axes:
        """
        """
        if not ax:
            ax = _create_ax(dim=3, figsize=figsize)
        ax.scatter(self.x, self.y, self.z, **kwargs)
        return ax


class Edge(LowerOrderCell):
    """
    Edge (1-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the edge.
    v_ids: List
        A list of two vertex identifiers of the edge.
    
    Attributes
    ----------
    id : int
        Identifier of the edge.
    v_ids : List
        A list of two vertex identifiers of the edge.
    length : float, optional
        Length of the edge.

    Methods
    -------
    from_tess_file(file)
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id)
        self.v_ids = v_ids

    @classmethod
    def from_tess_file(
        cls,
        file: io.TextIOBase | str,
        _vertices: Dict = {}
    ):
        """Extracts edges from Neper .tess file and returns a dictionary
        of Edge examples.

        Parameters
        ----------
        file
            Filename or file object.

        Returns
        -------
        _edges
            A dictionary of edges. Keys are edge identifiers, values are
            Edge examples.

        Notes
        -----
        Expects valid .tess file format (as for Neper 4.5.0). See details
        https://neper.info/doc/fileformat.html
        """
        if isinstance(file, str):
            file = open(file, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!') 

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
                        _vertices[v1_id].add_incident_cell(e_id)
                        _vertices[v1_id].add_neighbor(v2_id)
                        _vertices[v2_id].add_incident_cell(-e_id)
                        _vertices[v2_id].add_neighbor(v1_id)
                    _edges[e_id] = cls(e_id, v_ids)
                return _edges

    @property
    def length(self) -> float:
        """
        """
        try:
            return self.measure
        except AttributeError:
            return None 


class Edge2D(Edge, GrainBoundary):
    """
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id, v_ids)
        self.is_special = False


class Edge3D(Edge, TripleJunction):
    """
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id, v_ids)
        self.junction_type = None


class Face(Cell):
    """
    Face (2-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the face.
    v_ids : List
        A list of vertex identifiers of the face.
    
    Attributes
    ----------
    id : int
        Identifier of the face.
    v_ids: List
        A list of vertex identifiers of the face.
    e_ids : List
        A list of edge identifiers of the face.
    d : float

    a : float

    b : float

    c : float

    normal : Tuple[float]

    area : float, optional
        Area of the face.

    Methods
    -------
    from_tess_file(file)
    add_edge(e_id)
    add_edges(e_ids)
    add_equation(d, a, b, c)
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id)
        self.v_ids = v_ids
        self.e_ids = []

    @classmethod
    def from_tess_file(
        cls,
        file: io.TextIOBase | str,
        _edges: Dict = {}
    ):
        """Extracts faces from Neper .tess file and returns a dictionary
        of Face examples.

        Parameters
        ----------
        file
            Filename or file object.

        Returns
        -------
        _faces
            A dictionary of faces. Keys are face identifiers, values are
            Face examples.

        Notes
        -----
        Expects valid .tess file format (as for Neper 4.5.0). See details
        https://neper.info/doc/fileformat.html
        """
        if isinstance(file, str):
            file = open(file, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!') 

        _faces = {}
        for line in file:
            if '**face' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    f_id = int(row[0])
                    v_ids = []
                    for k in range(2, int(row[1]) + 2):
                        v_ids.append(int(row[k]))
                    face = cls(f_id, v_ids)
                    
                    row = file.readline().split()
                    e_ids = []
                    for k in range(1, int(row[0]) + 1):
                        e_id = int(row[k])
                        e_ids.append(abs(e_id))
                        if _edges:
                            if e_id > 0:
                                _edges[abs(e_id)].add_incident_cell(f_id)
                            else:
                                _edges[abs(e_id)].add_incident_cell(-f_id)
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
                return _faces

    def add_edge(self, e_id: int):
        """
        """
        if e_id not in self.e_ids:
            self.e_ids.append(e_id)

    def add_edges(self, e_ids: List):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))

    def add_equation(self, d: float, a: float, b: float, c: float):
        """

        Notes
        -----
        Parameters d, a, b, c are the parameters of the equation of a face
        :math:`ax + by + cz = d` with :math:`a^2 + b^2 + c^2 = 1`. See details
        https://neper.info/doc/fileformat.html
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


class Face2D(Face, Grain):
    """
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id, v_ids)
        self.seed = None
        self.ori = None
        self.oridesc = None

    @classmethod
    def from_tess_file(
        cls,
        file: io.TextIOBase | str,
        _edges: Dict = {}
    ):
        """
        """
        if isinstance(file, str):
            file = open(file, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!') 

        seeds = {}
        ori = {}
        _faces = {}
        file.seek(0) # to ensure that the pointer is set to the beginning
        for line in file:
            if '**cell' in line:
                N = int(file.readline().rstrip('\n'))
            if '*crysym' in line:
                crysym = file.readline().strip() #.rstrip('\n')
            if '*seed' in line:
                for i in range(N):
                    row = file.readline().split()
                    seeds[int(row[0])] = tuple([*map(float, row[1:4])])
            if '*ori' in line:
                oridesc = file.readline().strip() #.rstrip('\n')
                for i in range(N):
                    row = file.readline().split()
                    ori[i + 1] = tuple([*map(float, row)])
            if '**face' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    f_id = int(row[0])
                    v_ids = []
                    for k in range(2, int(row[1]) + 2):
                        v_ids.append(int(row[k]))
                    face = cls(f_id, v_ids)
                    
                    row = file.readline().split()
                    e_ids = []
                    for k in range(1, int(row[0]) + 1):
                        e_id = int(row[k])
                        e_ids.append(abs(e_id))
                        if _edges:
                            if e_id > 0:
                                _edges[abs(e_id)].add_incident_cell(f_id)
                            else:
                                _edges[abs(e_id)].add_incident_cell(-f_id)
                    face.add_edges(e_ids)
                    
                    row = file.readline().split()
                    face.add_equation(
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3])
                    )
                    _ = file.readline()
                    
                    face.set_crystal_ori(crysym, oridesc, ori[f_id])
                    face.set_seed(seeds[f_id])
                    _faces[f_id] = face
                return _faces


class Face3D(Face, GrainBoundary):
    """
    """
    def __init__(self, id: int, v_ids: List):
        super().__init__(id, v_ids)
        self.is_special = False


class Poly(Grain):
    """
    Polyhedron (3-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the polyhedron.
    f_ids : List
        A list of face identifiers of the polyhedron.
    
    Attributes
    ----------
    id : int
        Identifier of the face.
    v_ids: List
        A list of vertex identifiers of the polyhedron.
    e_ids : List
        A list of edge identifiers of the polyhedron.
    f_ids : List
        A list of face identifiers of the polyhedron.
    vol : float, optional
        Volume of the polyhedron.

    Methods
    -------
    from_tess_file(file)
    add_vertex(v_id)
    add_vertices(v_ids):
    add_edge(e_id)
    add_edges(e_ids)
    """
    def __init__(self, id: int, f_ids: List):
        super().__init__(id)
        self.v_ids = []
        self.e_ids = []
        self.f_ids = f_ids

    @classmethod
    def from_tess_file(
        cls,
        file: io.TextIOBase | str,
        _faces: Dict
    ):
        """
        """
        if isinstance(file, str):
            file = open(file, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!')

        seeds = {}
        ori = {}
        _polyhedra = {}
        file.seek(0) # to ensure that the pointer is set to the beginning
        for line in file:
            if '**cell' in line:
                N = int(file.readline().rstrip('\n'))
            if '*crysym' in line:
                crysym = file.readline().strip() #.rstrip('\n')
            if '*seed' in line:
                for i in range(N):
                    row = file.readline().split()
                    seeds[int(row[0])] = tuple([*map(float, row[1:4])])
            if '*ori' in line:
                oridesc = file.readline().strip() #.rstrip('\n')
                for i in range(N):
                    row = file.readline().split()
                    ori[i + 1] = tuple([*map(float, row)])
            if '**polyhedron' in line:
                n = int(file.readline().rstrip('\n'))
                for i in range(n):
                    row = file.readline().split()
                    p_id = int(row[0])
                    f_ids = []
                    v_ids = []
                    e_ids = []
                    for k in range(2, int(row[1]) + 2):
                        f_id = int(row[k])
                        f_ids.append(abs(f_id))
                        v_ids += _faces[abs(f_id)].v_ids
                        e_ids += _faces[abs(f_id)].e_ids
                        if f_id > 0:
                            _faces[abs(f_id)].add_incident_cell(p_id)
                        else:
                            _faces[abs(f_id)].add_incident_cell(-p_id)
                    f_ids = list(set(f_ids))
                    poly = cls(p_id, f_ids)
                    poly.add_vertices(v_ids)
                    poly.add_edges(e_ids)
                    poly.set_crystal_ori(crysym, oridesc, ori[p_id])
                    poly.set_seed(seeds[p_id])
                    _polyhedra[p_id] = poly
                return _polyhedra

    def add_vertex(self, v_id: int):
        """
        """
        if v_id not in self.v_ids:
            self.v_ids.append(v_id)

    def add_vertices(self, v_ids: List):
        """
        """
        self.v_ids += v_ids
        self.v_ids = list(set(self.v_ids))

    def add_edge(self, e_id: int):
        """
        """
        if e_id not in self.e_ids:
            self.e_ids.append(e_id)

    def add_edges(self, e_ids: List):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))

    @property
    def vol(self) -> float:
        """
        """
        try:
            return self.measure
        except AttributeError:
            return None 


class TripleJunctionSet:
    """
    """

    def __init__(self, p, j_tuple) -> None:
        """
        """
        self.p = p
        self.j0, self.j1, self.j2, self.j3 = j_tuple

    @property
    def q(self):
        """
        """
        return 1 - self.p

    @property
    def j_tuple(self):
        """
        """
        return (self.j0, self.j1, self.j2, self.j3)

    @property
    def Sp(self):
        """
        """
        if self.p == 0 or self.p == 1:
            return 0
        else:
            return matutils.entropy(self.p)

    @property
    def Sp_m(self):
        """
        """
        if self.p == 0:
            return np.inf
        else:
            return matutils.entropy_m(self.p)

    @property
    def Sp_s(self):
        """
        """
        if self.p == 0:
            return - np.inf
        else:
            return matutils.entropy_s(self.p)

    @property
    def p_expected(self):
        """
        """
        return (self.j1 + 2*self.j2 + 3*self.j3) / 3
        
    @property
    def delta_p(self):
        """
        """
        return abs(self.p_expected - self.p)

    @property
    def S(self):
        """
        """
        return matutils.entropy(*self.j_tuple)

    @property
    def S_m(self):
        """
        """
        if np.any(np.array(self.j_tuple) == 0):
            return np.inf
        else:
            return matutils.entropy_m(*self.j_tuple)

    @property
    def S_s(self):
        """
        """
        if np.any(np.array(self.j_tuple) == 0):
            return - np.inf
        else:
            return matutils.entropy_s(*self.j_tuple)

    @property
    def kappa(self):
        """
        """
        if self.S_s == 0:
            return 0
        else:
            return self.S_m / self.S_s

    @property
    def delta_S(self):
        """
        """
        return self.Sp - self.S

    @property
    def d_tuple(self):
        """
        """
        return matutils.get_d_tuple(self.j_tuple)

    @property
    def d1(self):
        """
        """
        return self.d_tuple[0]

    @property
    def d2(self):
        """
        """
        return self.d_tuple[1]

    @property
    def d3(self):
        """
        """
        return self.d_tuple[2]

    def get_property(self, attr):
        """
        """
        return getattr(self, attr)
    
    def get_properties(self, attr_list: List = []) -> Dict:
        """
        """
        if not attr_list:
            attr_list = [
                'p',
                'q',
                'Sp',
                'Sp_m',
                'Sp_s',
                'S',
                'S_m',
                'S_s',
                'kappa',
                'delta_S',
                'd1',
                'd2',
                'd3'
            ]

        try:
            return {attr_name: getattr(self, attr_name) for attr_name in attr_list}
        except:
            logging.exception('Check properties!')

        # values = [getattr(self, attr_name) for attr_name in attr_list]
        # return pd.DataFrame([values], columns = attr_list)


def _add_neighbors(_cells: Dict, _incident_cells: Dict):
    """
    Add neighbors to incident_cells from common cells
    _vertices, _edges
    _edges, _faces
    _faces, _polyhedra
    """ 
    for cell in _cells.values():
        for inc_cell_id in cell.incident_ids:
            _incident_cells[inc_cell_id].add_neighbors(cell.incident_ids)


# def _add_neighbors_neighbors(_cells: Dict):
#     """
#     """
#     for cell in _cells.values():
#         nn_ids = []
#         for n_id in cell.n_ids:
#             nn_ids += _cells[n_id].n_ids
#         cell.add_neighbors_neighbors(nn_ids)


def _add_measures(_cells: Dict, measures: List):
    """
    """
    n = len(_cells.keys())
    if n != len(measures):
        raise ValueError(
            'Number of cells must be equal to number of measures'
        )
    
    for i in range(n):
        _cells[i + 1].set_measure(measures[i])


def _add_thetas(
    _cells: Dict,
    thetas: Iterable,
    lower_thrd: float = None,
    upper_thrd: float = None
):
    """
    """
    n = len(_cells.keys())
    if n != len(thetas):
        raise ValueError(
            'Number of cells must be equal to number of theta'
        )
    
    for i in range(n):
        _cells[i + 1].set_theta(
            thetas[i],
            lower_thrd=lower_thrd,
            upper_thrd=upper_thrd
        )


def _parse_stfile(file: io.TextIOBase | str):
    """
    Return columns of the stfile
    """
    data = np.loadtxt(file)
    if len(data.shape) == 1:
        return (data,)
    
    columns = []
    for i in range(data.shape[1]):
        columns.append(data[:, i])
    return tuple(columns)


class CellComplex:
    """
    Cell complex.

    Parameters
    ----------
    dim: int
    _vertices: Dict
    _edges: Dict
    _faces: Dict
    _polyhedra: Dict, optional
    
    Attributes
    ----------
    dim: int
    _vertices: Dict
    _edges: Dict
    _faces: Dict
    _polyhedra: Dict, optional
    crysym : str
    load_time: float


    Methods
    -------
    from_tess_file(file)
    add_vertex(v_id)
    add_vertices(v_ids):
    add_edge(e_id)
    add_edges(e_ids)
    """
    def __init__(
        self,
        dim: int,
        _vertices: Dict,
        _edges: Dict,
        _faces: Dict,
        _polyhedra: Dict = None
    ):
        """
        """
        self.dim = dim
        self._vertices = _vertices
        self._edges = _edges
        self._faces = _faces
        if _polyhedra:
            self._polyhedra = _polyhedra

    @classmethod
    def from_tess_file(
        cls,
        file: io.TextIOBase | str,
        with_measures: bool = False,
        with_theta: bool = False,
        theta_file: str = None,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        """
        start = time.time()
        filename = None

        if isinstance(file, str):
            filename = file
            file = open(filename, 'r', encoding='utf-8')
        elif not isinstance(file, io.TextIOBase):
            raise ValueError('Check file name or format!') 

        for line in file:
            if '**general' in line:
                dim = int(file.readline().split()[0])
                if dim not in [2, 3]:
                    raise ValueError(
                        f'Dimension must be 2 or 3, not {dim}'
                    )
                break
        if dim == 2:
            _vertices = Vertex2D.from_tess_file(file)
        elif dim == 3:
            _vertices = Vertex3D.from_tess_file(file)


        if dim == 2:
            _edges = Edge2D.from_tess_file(file, _vertices)
        elif dim == 3:
            _edges = Edge3D.from_tess_file(file, _vertices)
                    
        _add_neighbors(_vertices, _edges)
        # _add_neighbors_neighbors(_vertices)
        # _add_neighbors_neighbors(_edges)

        if dim == 2:
            _faces = Face2D.from_tess_file(file, _edges)
        elif dim == 3:
            _faces = Face3D.from_tess_file(file, _edges)
        
        _add_neighbors(_edges, _faces)
        # _add_neighbors_neighbors(_faces)

        if dim == 3:
            _polyhedra = Poly.from_tess_file(file, _faces)
            _add_neighbors(_faces, _polyhedra)
            # _add_neighbors_neighbors(_polyhedra)
        
        # Set external
        if dim == 2:
            for e in _edges.values():
                if e.degree == 1:
                    e.set_external(True)
                    for v_id in e.v_ids:
                        _vertices[v_id].set_external(True)
                    for f_id in e.incident_ids:
                        _faces[f_id].set_external(True)
            file.close()

        elif dim == 3:
            for f in _faces.values():
                if f.degree == 1:
                    f.set_external(True)
                    for v_id in f.v_ids:
                        _vertices[v_id].set_external(True)
                    for e_id in f.e_ids:
                        _edges[e_id].set_external(True)
                    for p_id in f.incident_ids:
                        _polyhedra[p_id].set_external(True)
            file.close()
 
        if with_measures and filename:
            filename_m = filename.rstrip('.tess') + '.stedge'
            try:
                columns = _parse_stfile(filename_m)
                _add_measures(_edges, columns[0])
            except:
                logging.error(f'Error reading file {filename_m}')
            if with_theta and dim == 2:
                try:
                    _add_thetas(_edges, columns[1], lower_thrd, upper_thrd)
                except:
                    logging.error(f'Error reading theta from file {filename_m}')

            filename_m = filename.rstrip('.tess') + '.stface'
            try:
                columns = _parse_stfile(filename_m)
                _add_measures(_faces, columns[0])
            except:
                logging.error(f'Error reading file {filename_m}')
            if with_theta and dim == 3:
                try:
                    _add_thetas(_faces, columns[1], lower_thrd, upper_thrd)
                except:
                    logging.error(f'Error reading theta from file {filename_m}')

            if dim == 3:
                filename_m = filename.rstrip('.tess') + '.stpoly'
                try:
                    columns = _parse_stfile(filename_m)
                    _add_measures(_polyhedra, columns[0])
                except:
                    logging.error(f'Error reading file {filename_m}')
        
        elif with_theta and filename or theta_file:
            if theta_file:
                filename_m = theta_file
            if dim == 2:
                if not theta_file:
                    filename_m = filename.rstrip('.tess') + '.stedge'
                try:
                    columns = _parse_stfile(filename_m)
                    _add_thetas(_edges, columns[0], lower_thrd, upper_thrd)
                except:
                    logging.error(f'Error reading theta from file {filename_m}')
            elif dim == 3:
                if not theta_file:
                    filename_m = filename.rstrip('.tess') + '.stface'
                try:
                    columns = _parse_stfile(filename_m)
                    _add_thetas(_faces, columns[0], lower_thrd, upper_thrd)
                except:
                    logging.error(f'Error reading theta from file {filename_m}')

        if dim ==2:
            cellcomplex = cls(dim, _vertices, _edges, _faces)
            cellcomplex.crysym = _faces[1].crysym
        elif dim ==3:
            cellcomplex = cls(dim, _vertices, _edges, _faces, _polyhedra)
            cellcomplex.crysym = _polyhedra[1].crysym


        # Set junction types from theta (if known from a file)
        # If lower or upper threshold are known or both
        if lower_thrd or upper_thrd:
            cellcomplex.set_junction_types()

        cellcomplex.load_time = round(time.time() - start, 1)
        print('Complex loaded:', cellcomplex.load_time, 's')
        return cellcomplex

    def __str__(self):
        """
        """
        cc_str = f"<class {self.__class__.__name__}> {self.dim}D" +\
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
    def vertices(self):
        """
        """
        return [v for v in self._vertices.values()]

    @property
    def edges(self):
        """
        """
        return [e for e in self._edges.values()]

    @property
    def faces(self):
        """
        """
        return [f for f in self._faces.values()]

    @property
    def polyhedra(self):
        """
        """
        return [p for p in self._polyhedra.values()]

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

    def _choose_cell_type(self, cell_type: str | int):
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

    @property
    def _GBs(self):
        """
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('edge')
        elif self.dim == 3:
            _cells = self._choose_cell_type('face')
        return _cells

    @property
    def _TJs(self):
        """
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('vertex')
        elif self.dim == 3:
            _cells = self._choose_cell_type('edge')
        return _cells

    @property
    def _grains(self):
        """
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('face')
        elif self.dim == 3:
            _cells = self._choose_cell_type('poly')
        return _cells

    def get_one(self, cell_type: str | int, cell_id: int):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        return _cells[cell_id]
    
    def get_many(self, cell_type: str | int, cell_ids: List):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        return [_cells[cell_id] for cell_id in cell_ids]

    def get_external_ids(self, cell_type: str | int):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        return [cell.id for cell in _cells.values() if cell.is_external]

    def get_internal_ids(self, cell_type: str | int):
        """
        """
        _cells = self._choose_cell_type(cell_type)
        return [cell.id for cell in _cells.values() if not cell.is_external]

    def get_special_ids(self):
        """
        2D - edges can be special
        3D - faces can be special
        """
        return [cell.id for cell in self._GBs.values() if cell.is_special]

    def plot_vertices(
        self,
        v_ids: List = [],
        ax: Axes = None,
        figsize: Tuple = (8,8),
        labels: bool = False,
        **kwargs
    ) -> Axes:
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
                ax = v.plot(ax=ax, label=v.id, **kwargs)
            else:
                ax = v.plot(ax=ax, **kwargs)
        if labels:
            ax.legend(loc='best')
        return ax

    def plot_edges(
        self,
        e_ids: List = [],
        ax: Axes = None,
        figsize: Tuple = (8,8),
        labels: bool = False,
        **kwargs
    ) -> Axes:
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

    def plot_faces(
        self,
        f_ids: List = [],
        ax: Axes = None,
        figsize: Tuple = (8,8),
        labels: bool = False,
        **kwargs
    ) -> Axes:
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

    def plot_polyhedra(
        self,
        p_ids: List = [],
        ax: Axes = None,
        figsize: Tuple = (8,8),
        **kwargs
    ) -> Axes:
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

    def plot_seeds(
        self,
        cell_ids: List = [],
        ax: Axes = None,
        figsize: Tuple = (8,8),
        **kwargs
    ) -> Axes:
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
            x, y, z = cell.seed
            if self.dim == 3: 
                zs.append(z)
            xs.append(x)
            ys.append(y)
        if self.dim == 2:
            ax.scatter(xs, ys, **kwargs)
        elif self.dim == 3:
            ax.scatter(xs, ys, zs, **kwargs)
        return ax

    def get_junction_ids_of_type(self, junction_type: int) -> List:
        """
        2D - vertices are junctions
        3D - edges are junctions
        """
        junction_ids = []
        for cell in self._TJs.values():
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
        if n_int == 0:
            return 0
        else:
            p = n_spec / n_int
            return p

    def get_ext_fraction(self, cell_type: str | int):
        """
        number_of_external / number_of_cells
        """
        n_ext = len(self.get_external_ids(cell_type))
        n_cells = len(self._choose_cell_type(cell_type))
        frac = n_ext / n_cells
        return frac

    def get_j_fraction(self, junction_type: int):
        """
        number_of_junction_of_type / number_of_internal
        """
        n_junc = len(self.get_junction_ids_of_type(junction_type))
        if self.dim == 2:
            n_int = len(self.get_internal_ids('v'))
        elif self.dim == 3:
            n_int = len(self.get_internal_ids('e'))
        if n_int == 0:
            return 0
        else:
            frac = n_junc / n_int
            return frac

    @property
    def p(self):
        return self.get_spec_fraction()

    @property
    def j_tuple(self):
        j0 = self.get_j_fraction(0)
        j1 = self.get_j_fraction(1)
        j2 = self.get_j_fraction(2)
        j3 = self.get_j_fraction(3)
        return (j0, j1, j2, j3)

    def to_TJset(self):
        """
        """
        return TripleJunctionSet(self.p, self.j_tuple)

    def set_junction_types(self) -> None:
        """
        external has None junction type
        """
        if self.dim == 2:
            for v in self._vertices.values():
                v.n_spec_edges = 0
                if not v.is_external:
                    for e_id in v.incident_ids:
                        if self._edges[e_id].is_special:
                            v.n_spec_edges += 1
                    if v.n_spec_edges > 3:
                        logging.warning(
                            f'{v} is incident to ' +
                            f'{v.n_spec_edges} special edges'
                        )
                    v.set_junction_type(v.n_spec_edges)
        elif self.dim == 3:
            for e in self._edges.values():
                e.n_spec_faces = 0
                if not e.is_external:
                    for f_id in e.incident_ids:
                        if self._faces[f_id].is_special:
                            e.n_spec_faces += 1
                    if e.n_spec_faces > 3:
                        logging.warning(
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
        for cell in self._GBs.values():
            cell._reset_theta_thrds(lower_thrd, upper_thrd)
        self.set_junction_types()

    def describe(self, attr_list: List = []):
        """
        """
        state = self.to_TJset()
        return state.get_properties(attr_list)

    def set_theta_from_ori(
        self,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        """
        for cell in self._GBs.values():
            if not cell.is_external:
                g_ids = cell.incident_ids
                if len(g_ids) != 2:
                    raise ValueError('GB has more than 2 incident grains')
                g1, g2 = self._grains[g_ids[0]], self._grains[g_ids[1]]
                theta = matutils.dis_angle(g1, g2)
                cell.set_theta(theta, lower_thrd, upper_thrd)

    def set_thetas(
        self,
        thetas: Iterable,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        """
        _add_thetas(self._GBs, thetas, lower_thrd, upper_thrd)

    def set_theta_from_file(
        self,
        file: io.TextIOBase | str,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        columns = _parse_stfile(file)
        _add_thetas(self._GBs, columns[0], lower_thrd, upper_thrd)

    def find_neighbors_of_order(self, max_order: int):
        """
        0 order - grain id
        1 order - grain neighbor ids
        2 order - grain neighbor of neighbor ids
        etc.

        access to neighbors  nX_ids attributes of grains
        """
        # n0_ids is [self.id]
        for g in self._grains.values():
            setattr(g, f'n0_ids', [g.id])
        # n0_ids is [self.id]
        for k in range(1, max_order + 1):
            for g in self._grains.values():
                nn_ids = []
                for n_id in getattr(g, f'n{k - 1}_ids'):
                    nn_ids += self._faces[n_id].n_ids
                s = set(nn_ids)
                for j in range(k):
                    s.difference_update(getattr(g, f'n{j}_ids'))
                setattr(g, f'n{k}_ids', list(s))
        self.n_max_order = max_order

    def get_neighbor_dis_angles(self, order: int):
        """
        0 order - grain id
        1 order - grain neighbor ids
        2 order - grain neighbor of neighbor ids
        etc.
        """     
        angles = []
        for g in self._grains.values():
            g.rot_mtx = g.R
        for g1 in tqdm(self._grains.values()):
            n_ids = getattr(g1, f'n{order}_ids')
            for n_id in n_ids:
                if g1.id < n_id:
                    angle = matutils.calculate_disorient(
                        g1.rot_mtx,
                        self._grains[n_id].rot_mtx,
                        g1.crysym)
                    angles.append(angle)
        return angles

    def get_neighbor_counts_of_order(self, order: int):
        """
        """
        return np.array(
            [len(getattr(g, f'n{order}_ids')) for g in self._grains.values()]
        )
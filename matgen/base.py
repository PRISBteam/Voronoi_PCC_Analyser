"""Base classes.

"""
from __future__ import annotations
import io
import time
from typing import Dict, Iterable, List, Tuple
import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO
)
from tqdm import tqdm
import numpy as np
import random
import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matgen import matutils
from matgen.entropic import TripleJunctionSet


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
    n_ids : list
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
    
    def add_neighbors(self, n_ids: list) -> None:
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

    @property
    def size(self) -> float:
        """
        """
        try:
            return self.measure
        except AttributeError:
            return None 


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
    signed_incident_ids: list
        A list of incident cells identifiers. Identifiers are signed: they
        are negative if orientations of the cell and incident cell differ.
    incident_ids: list
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

    @property
    def degree(self) -> int:
        """Number of incident cells of the given cell.
        """
        return len(self.signed_incident_ids)


class GrainBoundary(LowerOrderCell):
    """
    Grain Boundary (GB). 
    2D: 1-cells (edges) are considered to be grain biundaries.
    3D: 2-cells (faces) are considered to be grain boundaries.

    Parameters
    ----------
    id : int
        Identifier of the grain boundary.
    
    Attributes
    ----------
    id : int
        Identifier of the grain boundary.
    is_special: bool
        Is the grain boundary special?
    theta: float
        Disorientation angle between incident grains (in degrees).
        Theta = -1 for external grain boundaries.
    gb_index: int
        Grain boundary index.

    Methods
    -------
    set_special(is_special)
    set_theta(theta, lower_thrd, upper_thrd)
    set_external(is_external)
    set_gb_index(gb_index)
    get_new_seed_prob(critical_size)

    References
    ---------
    Discrete model for discontinuous dynamic recrystallisation in its
    application to post-dynamic recrystallisation in adiabatic shear
    bands
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.is_special = False
        self.theta = None
        self.gb_index = None

    def set_special(self, is_special: bool = True):
        """
        Some grain boundaries can be set special.
        External cells cannot be set special.
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
        Reset is_special with respect to new thresholds for theta.
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
        When set external, it is set non-special and theta = -1
        """
        self.is_external = is_external
        if is_external:
            self.theta = -1
            self.set_special(False)
        elif self.theta == -1:
            self.theta = None

    def set_gb_index(self, gb_index: int):
        """
        gb_index equals ???
        check consistency: external has type None
        """
        if not self.is_external:
            self.gb_index = gb_index
        elif self.is_external:
            raise ValueError(
                'External GB cannot have a GB index other than None'
            )

    def get_new_seed_prob(self, critical_size=0):
        """
        The probability for nucleation of a new grain on the GB.
        Without the coefficient of initial probability etha0.
        """
        try:
            if self.eq_diam > critical_size:
                coeff = (1 - (critical_size / self.eq_diam)**3)
            else:
                coeff = 0
        except:
            coeff = 1

        if self.is_external:
            return 0
        else:
            return (2 * self.gb_index) / (3 * len(self.n_ids)) * coeff
        

class TripleJunction(LowerOrderCell):
    """
    Triple junction (TJ). 
    2D: 0-cells (vertices) are considered to be triple junctions.
    3D: 1-cells (edges) are considered to be triple junctions.

    Parameters
    ----------
    id : int
        Identifier of the triple junction.
    
    Attributes
    ----------
    id : int
        Identifier of the triple junction.
    junction type: int
        Junction_type equals the number of special incident grain boundaries.

    Methods
    -------
    set_junction_type(junction_type)
    
    Notes
    -----
    Any binary classification of grain boundaries induces four distinct
    TJs types: J0, J1, J2 and J3, where Jk is the index enumerating the number
    of special GBs joint at a specific triple line. The corresponding TJ fractions of
    different types j0, j1, j2 and j3 are the probabilities that a randomly chosen
    TJ will possess with the corresponding type.

    References
    ---------
    """
    def __init__(self, id: int):
        super().__init__(id)
        self.junction_type = None

    def set_junction_type(self, junction_type: int):
        """
        Junction type equals number of special incident cells.
        External has type None.
        """
        if not self.is_external:
            self.junction_type = junction_type
        elif self.is_external:
            raise ValueError(
                'External junction cannot have a type other than None'
            )

    
def _create_ax(dim: int = 2, figsize: Tuple = (8,8)) -> Axes:
    """
    Create axis.
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
    Vertex (0-cell) of a 2D cell complex.

    Parameters
    ----------
    id : int
        Identifier of the vertex.
    x: float
        Vertex x-coordinate.
    y: float
        Vertex y-coordinate.
    z: float, optional
        Vertex z-coordinate. Default is 0.
    
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
    plot(ax, figsize, **kwargs)
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
    Vertex (0-cell) of a 3D cell complex.

    Methods
    -------
    plot(ax, figsize, **kwargs)
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
    v_ids: list
        A list of two vertex identifiers of the edge.
    
    Attributes
    ----------
    id : int
        Identifier of the edge.
    v_ids : list
        A list of two vertex identifiers of the edge.
    length : float, optional
        Length of the edge.

    Methods
    -------
    from_tess_file(file, _vertices)
    """
    def __init__(self, id: int, v_ids: list):
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
    Edge (1-cell) of a 2D cell complex is a grain boundary.

    Parameters
    ----------
    id : int
        Identifier of the edge.
    v_ids: list
        A list of two vertex identifiers of the edge.
    
    Additional Attributes
    ----------
    tj_ids : list
        Alias for the list of two vertex identifiers of the edge.
    eq_diam: float
        Alias for the edge's length.


    """
    def __init__(self, id: int, v_ids: list):
        super().__init__(id, v_ids)
        self.is_special = False

    @property
    def tj_ids(self):
        return self.v_ids
    
    @property
    def eq_diam(self):
        return self.length


class Edge3D(Edge, TripleJunction):
    """
    Edge (1-cell) of a 3D cell complex is a triple junction.
    """
    def __init__(self, id: int, v_ids: list):
        super().__init__(id, v_ids)
        self.junction_type = None


class Face(Cell):
    """
    Face (2-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the face.
    v_ids : list
        A list of vertex identifiers of the face.
    
    Attributes
    ----------
    id : int
        Identifier of the face.
    v_ids: list
        A list of vertex identifiers of the face.
    e_ids : list
        A list of edge identifiers of the face.
    d : float
        Parameter of the equation of the face (see Notes).
    a : float
        Parameter of the equation of the face (see Notes).
    b : float
        Parameter of the equation of the face (see Notes).
    c : float
        Parameter of the equation of the face (see Notes).
    normal : Tuple[float]
        Normal vector of the face (a, b, c).
    area : float, optional
        Area of the face.

    Methods
    -------
    from_tess_file(file, _edges)
    add_edge(e_id)
    add_edges(e_ids)
    add_equation(d, a, b, c)

    Notes
    -----
    Parameters d, a, b, c are the parameters of the equation of a face
    :math:`ax + by + cz = d` with :math:`a^2 + b^2 + c^2 = 1`. See details
    https://neper.info/doc/fileformat.html
    """
    def __init__(self, id: int, v_ids: list):
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

    def add_edges(self, e_ids: list):
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
    Face (2-cell) of a 2D cell complex is a grain.
    """
    def __init__(self, id: int, v_ids: list):
        super().__init__(id, v_ids)
        self.seed = None
        self.ori = None
        self.oridesc = None

    @property
    def gb_ids(self):
        return self.e_ids

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
                    seeds[int(row[0])] = tuple([*map(float, row[1:3])])
                    # seeds[int(row[0])] = tuple([*map(float, row[1:4])])
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
    Face (2-cell) of a 3D cell complex is a grain boundary.
    """
    def __init__(self, id: int, v_ids: list):
        super().__init__(id, v_ids)
        self.is_special = False

    @property
    def tj_ids(self):
        return self.e_ids
    
    @property
    def eq_diam(self):
        if self.area is not None:
            return math.sqrt(4 * self.area / math.pi)


class Poly(Grain):
    """
    Polyhedron (3-cell) of a cell complex.

    Parameters
    ----------
    id : int
        Identifier of the polyhedron.
    f_ids : list
        A list of face identifiers of the polyhedron.
    
    Attributes
    ----------
    id : int
        Identifier of the face.
    v_ids: list
        A list of vertex identifiers of the polyhedron.
    e_ids : list
        A list of edge identifiers of the polyhedron.
    f_ids : list
        A list of face identifiers of the polyhedron.
    vol : float, optional
        Volume of the polyhedron.

    Methods
    -------
    from_tess_file(file, _faces)
    add_vertex(v_id)
    add_vertices(v_ids):
    add_edge(e_id)
    add_edges(e_ids)
    """
    def __init__(self, id: int, f_ids: list):
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

    def add_vertices(self, v_ids: list):
        """
        """
        self.v_ids += v_ids
        self.v_ids = list(set(self.v_ids))

    def add_edge(self, e_id: int):
        """
        """
        if e_id not in self.e_ids:
            self.e_ids.append(e_id)

    def add_edges(self, e_ids: list):
        """
        """
        self.e_ids += e_ids
        self.e_ids = list(set(self.e_ids))

    @property
    def vol(self) -> float:
        """
        Polyhedron volume.
        """
        try:
            return self.measure
        except AttributeError:
            return None
    
    @property
    def gb_ids(self):
        return self.f_ids


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


def _add_measures(_cells: Dict, measures: list):
    """
    Add measures to the dict of cells from the list of measures.
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
    Add thetas to the dict of cells from the list of thetas.
    Parameters `lower_thrd` and `upper_thrd` can be used to
    set special GBs from theta values.
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
    dim : int
        Cell complex dimension (2 or 3).
    _vertices : dict
    vertices : list
    _edges : dict
    edges : list
    _faces : dict
    faces : list
    _polyhedra : dict
    polyhedra : list
    crysym : str
    load_time : float
    vernb : int
        Number of vertices.
    edgenb : int
        Number of edges.
    facenb : int
        Number of faces.
    polynb : int
        Number of polyhedra.
    grainnb : int
        Number of grains.
    _GBs : dict
        Grain boundaries. Keys are grain ids, values are the cells of
        the corresponding class (edges for 2D, faces for 3D).
    _TJs : dict
        Triple junctions. Keys are grain ids, values are the cells of
        the corresponding class (vertices for 2D, edges for 3D).
    _grains : dict
        Grains. Keys are grain ids, values are the cells of
        the corresponding class (faces for 2D, polyhedra for 3D).
    _three_sided_grains : dict
        Grains with 3 sides in 2D cell complex.
    p : float
        Special GB fraction.
    j_tuple : tuple
        TJ fractions (j0, j1, j2, j3).
    n_max_order : int
    
    Methods
    -------
    from_tess_file(file, with_cell_size, with_measures, 
                    with_theta, theta_file, lower_thrd, upper_thrd)
    get_one(cell_type, cell_id)
    get_many(cell_type, cell_ids)
    get_external_ids(cell_type)
    get_internal_ids(cell_type)
    get_special_ids()
    get_nonspecial_internal_ids()
    plot_vertices(v_ids, ax, figsize, labels, **kwargs)
    plot_edges(e_ids, ax, figsize, labels, **kwargs)
    plot_faces(f_ids, ax, figsize, labels, **kwargs)
    plot_polyhedra(p_ids, ax, figsize, **kwargs)
    plot_seeds(cell_ids, ax, figsize, **kwargs)
    get_junction_ids_of_type(junction_type)
    get_spec_fraction()
    get_ext_fraction(cell_type)
    get_j_fraction(junction_type)
    get_three_sided_distribution()
    set_measures_from_coo()
    set_junction_types()
    set_gb_indexes()
    set_three_sided_types()
    reset_special(lower_thrd, upper_thrd, special_ids, warn_external)
    to_TJset()
    describe(attr_list)
    set_theta_from_ori(lower_thrd, upper_thrd)
    set_thetas(thetas, lower_thrd, upper_thrd)
    set_theta_from_file(file, lower_thrd, upper_thrd)
    find_neighbors_of_order(max_order)
    get_neighbor_dis_angles(order)
    get_neighbor_counts_of_order(order)
    get_new_random_seeds(k, critical_size, spec_prob, exclusion_list, replace)
    """
    def __init__(
        self,
        dim: int,
        _vertices: Dict,
        _edges: Dict,
        _faces: Dict,
        _polyhedra: Dict = None
    ):
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
        with_cell_size: bool = False,
        with_measures: bool = False,
        with_theta: bool = False,
        theta_file: str = None,
        lower_thrd: float = None,
        upper_thrd: float = None
    ):
        """
        with_measures: all measures desired (.stedge, .stface, .stpoly)
        with_cell_size: only size for cells desired (.stcell)
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

        if dim == 2:
            _faces = Face2D.from_tess_file(file, _edges)
        elif dim == 3:
            _faces = Face3D.from_tess_file(file, _edges)
        
        _add_neighbors(_edges, _faces)

        if dim == 3:
            _polyhedra = Poly.from_tess_file(file, _faces)
            _add_neighbors(_faces, _polyhedra)
        
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
 
        if with_cell_size and filename:
            filename_m = filename.rstrip('.tess') + '.stcell'
            try:
                columns = _parse_stfile(filename_m)
            except:
                logging.warning(f'Error reading file {filename_m}')
            if dim == 2:
                _add_measures(_faces, columns[0])
            elif dim == 3:
                _add_measures(_polyhedra, columns[0])

        if with_measures and filename:
            filename_m = filename.rstrip('.tess') + '.stedge'
            try:
                columns = _parse_stfile(filename_m)
                _add_measures(_edges, columns[0])
            except:
                logging.warning(f'Error reading file {filename_m}')
            if with_theta and dim == 2:
                try:
                    _add_thetas(_edges, columns[1], lower_thrd, upper_thrd)
                except:
                    logging.warning(f'Error reading theta from file {filename_m}')

            filename_m = filename.rstrip('.tess') + '.stface'
            try:
                columns = _parse_stfile(filename_m)
                _add_measures(_faces, columns[0])
            except:
                logging.warning(f'Error reading file {filename_m}')
            if with_theta and dim == 3:
                try:
                    _add_thetas(_faces, columns[1], lower_thrd, upper_thrd)
                except:
                    logging.warning(f'Error reading theta from file {filename_m}')

            if dim == 3:
                filename_m = filename.rstrip('.tess') + '.stpoly'
                try:
                    columns = _parse_stfile(filename_m)
                    _add_measures(_polyhedra, columns[0])
                except:
                    logging.warning(f'Error reading file {filename_m}')
        
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
                    logging.warning(f'Error reading theta from file {filename_m}')
            elif dim == 3:
                if not theta_file:
                    filename_m = filename.rstrip('.tess') + '.stface'
                try:
                    columns = _parse_stfile(filename_m)
                    _add_thetas(_faces, columns[0], lower_thrd, upper_thrd)
                except:
                    logging.warning(f'Error reading theta from file {filename_m}')

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
            if cellcomplex.dim == 2:
                cellcomplex.set_three_sided_types()

        cellcomplex.load_time = round(time.time() - start, 1)
        # print('Complex loaded:', cellcomplex.load_time, 's')
        return cellcomplex

    def __str__(self):
        cc_str = f"<class {self.__class__.__name__}> {self.dim}D" +\
        f"\n{self.vernb} vertices" + f"\n{self.edgenb} edges" +\
        f"\n{self.facenb} faces" 
        
        if self.dim == 3:
            cc_str += f"\n{self.polynb} polyhedra"
        return cc_str

    def __repr__(self):
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
        if self.dim == 3:
            return [p for p in self._polyhedra.values()]

    @property
    def vernb(self):
        """
        Number of vertices.
        """
        return len(self._vertices)

    @property
    def edgenb(self):
        """
        Number of edges.
        """
        return len(self._edges)

    @property
    def facenb(self):
        """
        Number of faces.
        """
        return len(self._faces)

    @property
    def polynb(self):
        """
        Number of polyhedra.
        """
        if self.dim == 3:
            return len(self._polyhedra)

    @property
    def grainnb(self):
        """
        Number of grains.
        """
        return len(self._grains)

    def _choose_cell_type(self, cell_type: str | int):
        """
        Returns a dict with cells corresponding to the cell type.
        cell_type may be of str or int data type.
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
        Grain boundaries.
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('edge')
        elif self.dim == 3:
            _cells = self._choose_cell_type('face')
        return _cells

    @property
    def _TJs(self):
        """
        Triple junctions.
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('vertex')
        elif self.dim == 3:
            _cells = self._choose_cell_type('edge')
        return _cells

    @property
    def _grains(self):
        """
        Grains.
        """
        if self.dim == 2:
            _cells = self._choose_cell_type('face')
        elif self.dim == 3:
            _cells = self._choose_cell_type('poly')
        return _cells

    @property
    def _three_sided_grains(self):
        """
        Grains with 3 sides in 2D cell complex.
        """
        if self.dim == 2:
            three_sided = {}
            for f_id, face in self._faces.items():
                if len(face.e_ids) == 3 and not face.is_external:
                    three_sided[f_id] = face
        return three_sided

    def get_one(self, cell_type: str | int, cell_id: int):
        """
        Get one cell of chosen type with chosen id.
        """
        _cells = self._choose_cell_type(cell_type)
        return _cells[cell_id]
    
    def get_many(self, cell_type: str | int, cell_ids: list):
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
        only internal GBs may be special
        """
        return [cell.id for cell in self._GBs.values() if cell.is_special]

    def get_nonspecial_internal_ids(self):
        """
        internal and external can be nonspecial
        """
        cell_ids = []
        for cell in self._GBs.values():
            if not cell.is_special and not cell.is_external:
                cell_ids.append(cell.id)
        return cell_ids

    def plot_vertices(
        self,
        v_ids: list = [],
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
        e_ids: list = [],
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
        f_ids: list = [],
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
        p_ids: list = [],
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
        cell_ids: list = [],
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

    def get_junction_ids_of_type(self, junction_type: int) -> list:
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

    def get_three_sided_distribution(self):
        """
        dictionary {type: fraction}
        fraction = number_of_grains_of_type / number_of_three_sided_grains
        11 possible types for i + j
        """
        distribution = {three_sided_type: 0 for three_sided_type in range(11)}
        for face in self._three_sided_grains.values():
            distribution[face.three_sided_type] += 1
        return distribution

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
    
    def set_measures_from_coo(self):
        """
        2D case
        """
        if self.dim == 3:
            pass
        elif self.dim == 2:
            for face in self._faces.values():
                points = []
                for v_id in face.v_ids:
                    points.append(self._vertices[v_id].coord)
                face.set_measure(matutils.polygon_area_2D(points))
            for edge in self._edges.values():
                points = []
                for v_id in edge.v_ids:
                    points.append(self._vertices[v_id].coord)
                edge.set_measure(matutils.edge_length_2D(points))

    def set_junction_types(self):
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

    def set_gb_indexes(self):
        """
        """
        for gb in self._GBs.values():
            if not gb.is_external:
                counter = {1: 0, 2: 0, 3: 0}
                for tj_id in gb.tj_ids:
                    j_type = self._TJs[tj_id].junction_type
                    # j_type may be None, 0, 1, 2, 3
                    if j_type:
                        counter[j_type] += 1
                gb_index = counter[1] + 2*counter[2] + 3*counter[3]
                gb.set_gb_index(gb_index=gb_index)

    def set_three_sided_types(self):
        """
        T_{i + j}
        i - the number of special GBs (0...6)
        j - the number of triple junctions with two or more special
        boundaries (0...4) 
        i + j = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        """
        if self.dim != 2:
            logging.warning('Three sided types are for 2D complexes')
            return
        for face in self._three_sided_grains.values():
            e_ids_set = set()
            i = 0
            j = 0
            for vertex in self.get_many('v', face.v_ids):
                e_ids_set.update(vertex.incident_ids)
                if vertex.junction_type and vertex.junction_type >= 2:
                    j += 1
            
            # fourth junction
            fourth_junction = 0
            for edge in self.get_many('e', face.e_ids):
                if edge.is_special:
                    fourth_junction += 1
            if fourth_junction >= 2:
                j += 1

            if len(e_ids_set) != 6:
                logging.warning(
                    f'{face.id} three-sided grain has ' +
                    f'{len(e_ids_set)} boundaries'
                )
            for edge in self.get_many('e', list(e_ids_set)):
                if edge.is_special:
                    i += 1
            face.three_sided_type = i + j
    
    def reset_special(
            self,
            lower_thrd: float = None,
            upper_thrd: float = None,
            special_ids: list | set = None,
            warn_external: bool = True):
        """
        two options for reset:
        1. Specify new thresholds if theta is known
        2. Specify explicitly the list of special GBs
        Options cannot be combined together.
        If special_ids specified, then thresholds are ignored.
        GBs that are in special_ids becomes special, that aren't
        becomes not special. 
        """
        external_ids = []
        if special_ids is not None:
            for cell in self._GBs.values():
                if cell.id in special_ids:
                    if cell.is_external:
                        external_ids.append(cell.id)
                    else:
                        cell.set_special(True)
                else:
                    cell.set_special(False)
            if external_ids and warn_external:
                logging.warning(
                    f'GBs with id {external_ids}' +
                    ' are external and cannot be special'
                )
        else:        
            for cell in self._GBs.values():
                if not cell.is_external and not cell.theta:
                    raise ValueError('Set theta first!')
                cell._reset_theta_thrds(lower_thrd, upper_thrd)
        self.set_junction_types()
        self.set_gb_indexes()
        if self.dim == 2:
            self.set_three_sided_types()

    def to_TJset(self):
        """
        """
        return TripleJunctionSet(self.p, self.j_tuple)
    
    def describe(self, attr_list: list = []):
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

    def get_new_random_seeds(
        self,
        k: int,
        critical_size: float = 0.0,
        spec_prob: float = 1.0,
        exclusion_list: list = [],
        replace: bool = True
    ) -> list:
        """
        k - the number of new seeds
        k = k_spec + k_nonspec

        k_spec is defined by spec_prob probability for new seed to be on
        special GB

        exclusion_list - list of GB ids that cannot produce a new seed 
        """
        gb_ids = []
        probs = []
        for gb_id, gb in self._GBs.items():
            if not gb.is_external:
                gb_ids.append(gb_id)
                probs.append(spec_prob * gb.get_new_seed_prob(critical_size))
        
        sample_ids = random.choices(population=gb_ids, weights=probs, k=k)

        # # ----------------------------------
        # # Lagacy version
        # # Get special and non-special GB ids
        # special_ids = self.get_special_ids()
        # nonspecial_ids = self.get_nonspecial_internal_ids()

        # # Eliminate GB ids from exclusion list
        # special_ids = list(set(special_ids).difference(exclusion_list))
        # nonspecial_ids = list(set(nonspecial_ids).difference(exclusion_list))

        # # Choose randomly k_spec and k_nonspec
        # rng = np.random.default_rng()
        # k_spec = rng.binomial(k, p=spec_prob)
        # k_nonspec = k - k_spec

        # # Choose k random grain boundaries from special and non-special
        # if len(special_ids) >= k_spec and len(nonspecial_ids) >= k_nonspec:
        #     sample_ids = rng.choice(
        #         special_ids, size=k_spec, replace=replace
        #     ).tolist()
        #     sample_ids += rng.choice(
        #         nonspecial_ids, size=k_nonspec, replace=replace
        #     ).tolist()
        # elif len(special_ids) < k_spec:
        #     logging.exception(
        #         f'Not enough special GBs to produce new {k_spec} seeds'
        #     )
        #     return []
        # elif len(nonspecial_ids) < k_nonspec:
        #     logging.exception(
        #         f'Not enough non-special GBs to produce new {k_nonspec} seeds'
        #     )
        #     return []


#         # Choose k random grain boundaries from non-special
# #        nonspecial_ids = list(set(internal_ids) - set(special_ids))
#         if len(nonspecial_ids) >= k:
#             sample_ids = random.sample(nonspecial_ids, k=k)
#         else:
#             logging.warning('All GBs are special')
#             return []
        
        # Choose a point for each chosen grain boundary 
        new_seeds = []
        for sample_id in sample_ids:
            grain_boundary = self._GBs[sample_id]
            vs = self.get_many('v', grain_boundary.v_ids)
            xp = [v.x for v in vs]
            yp = [v.y for v in vs]
            if self.dim == 3:
                zp = [v.z for v in vs]
                new_seed_coord = matutils.get_random_point_on_face(
                    a=grain_boundary.a,
                    b=grain_boundary.b,
                    c=grain_boundary.c,
                    d=grain_boundary.d,
                    xp=xp, yp=yp, zp=zp
                )
                new_seeds.append(new_seed_coord)
            else:
                new_seed_coord = matutils.get_random_point_on_edge(xp, yp)
                new_seeds.append(new_seed_coord)
        return new_seeds

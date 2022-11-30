"""
class CellComplex
"""

# import os
# from typing import Dict, Iterable, List, Tuple, Union
# import time

# import numpy as np

# #import representation
# from matgen import matutils

# class Vertex():
#     """
#     Class Vertex (0-cell).
    
#     Attributes
#     ----------
#     id
#         Identifier of a vertex.
#     x
#         x-coordinate.
#     y
#         y-coordinate.
#     z
#         z-coordinate.
#     coord
#         A tuple of x-, y-, z-coordinates.
#     neighbors
#         List of neighbouring vertices.
#     e_ids
#         List of edges which the vetrex incident to.
        
#     Methods
#     -------
#     add_incident_edge
        
#     add_incident_edges
        
#     get_degree


# class Edge():
#     """
#     Class Edge (1-cell).
    
#     Attributes
#     ----------
#     id
#         Identifier of an edge.
#     v_ids
#         A list of two vertices (their ids) of the edge.
#     neighbors
#         List of neighbouring edges.
#     incident_faces
#         List of faces which the edge belongs to.
        
#     Methods
#     -------
#     add_neighbor
#         Add a neighboring edge (by id).
#     add_neighbors
#         Add a list of neighboring edges (by id).
#     add_incident_face
    
#     add_incident_faces
    
#     set_orientation
    
#     get_degree
    
#     """


#     # def plot(
#     #         self,
#     #         dim: int = 2,
#     #         ax: Axes = None,
#     #         figsize: Tuple = (8,8),
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         ax = _create_ax(dim, figsize)
        
#     #     v1, v2 = self.v_ids # won't work because ids not equal to objects
#     #     x_space = np.linspace(v1.x, v2.x, 50)
#     #     y_space = np.linspace(v1.y, v2.y, 50)
#     #     if dim == 2:
#     #         ax.plot(x_space, y_space, **kwargs)
#     #     elif dim == 3:
#     #         z_space = np.linspace(v1.z, v2.z, 50)
#     #         ax.plot(x_space, y_space, z_space, **kwargs)
        
#     #     return ax

#     # def plot3D(self, ax: Axes = None):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_3D_axis()
#     #     x1, x2 = [v.x for v in self.vertices]
#     #     y1, y2 = [v.y for v in self.vertices]        
#     #     z1, z2 = [v.z for v in self.vertices]

#     #     x_space = np.linspace(x1, x2, 50)
#     #     y_space = np.linspace(y1, y2, 50)
#     #     z_space = np.linspace(z1, z2, 50)
        
#     #     ax.plot(x_space, y_space, z_space)



# class Face():
#     """
#     Class Face (2-cell).
    
#     Attributes
#     ----------
#     id
#         Identifier of a face.
#     vertices
#         A list of vertices of the face.
#     neighbors
#         List of neighbouring faces.
#     incident_poly
#         List of polyhedra which the face belongs to.
#     is_special
        
#     Methods
#     -------
#     add_neighbor
#         Add a neighboring face (by id).
#     add_neighbors
#         Add a list of neighboring faces (by id).
#     add_incident_poly
    
#     add_incident_polys
    
#     set_orientation
        
#     get_degree
    
#     get_type
    
#     set_special
    
#     """

        
  

# class Poly():
#     """
#     Class Poly (3-cell).
    
#     Attributes
#     ----------
#     id
#         Identifier of a polyhedron.
#     vertices
#         A list of vertices of the polyhedron.
#     neighbors
#         List of neighbouring polyhedra.
#     faces
#         List of faces that are on the boundary of the polyhedron.
#     seed
        
        
#     Methods
#     -------
#     add_neighbor
#         Add a neighboring polyhedron (by id).
#     add_neighbors
#         Add a list of neighboring polyhedra (by id).
#     set_orientation
        
#     get_degree
#     """
#     def __init__(self, id: int, f_ids: Iterable):
#         """
#         """



# class CellComplex():
#     """
#     Class CellComplex.
#     A class for cell complex.

#     Attributes
#     ----------

#     Methods
#     -------

#     """
#     # Initializes a cell complex from Neper .tess file.
#     def __init__(
#             self,
#             filename: str = 'complex.tess',
#             measures: bool = False,
#             theta: bool = False,
#             lower_thrd: float = None,
#             upper_thrd: float = None):
#         """
#         """
#         start = time.time()
#         self.source_file = filename
#         self.measures = measures # has measures or not

#         # Define the cell complex dimension (can be 2 or 3)
#         with open(filename, 'r', encoding="utf-8") as f:
#             for line in f:
#                 if '**general' in line:
#                     self.dim = int(f.readline().split()[0])
#                     if self.dim not in [2, 3]:
#                         raise ValueError(
#                             f'Dimension must be 2 or 3, not {self.dim}'
#                         )
#                     break
        
#         # In 2D case edges have misorintation angles (theta)
#         # In 3D case faces have misorintation angles (theta)
#         if self.dim == 2:
#             theta2D = theta
#             theta3D = False
#         elif self.dim ==3:
#             theta2D = False
#             theta3D = theta

#         # A dictionary
#         self._vertices = Vertex.from_tess_file(filename)
#         # A list
#         self.vertices = [v for v in self._vertices.values()]
        
#         # print(len(self._vertices.keys()), 'vertices loaded:',
#         #     time.time() - start, 's')
#         # A dictionary
#         self._edges = Edge.from_tess_file(
#             filename = filename, 
#             _vertices = self._vertices,
#             measure=measures,
#             theta=theta2D,
#             lower_thrd=lower_thrd,
#             upper_thrd=upper_thrd
#         )
#         # A list
#         self.edges = [e for e in self._edges.values()]
        
#         # print(len(self._edges.keys()),'edges loaded:',
#         #     time.time() - start, 's')
        
#         # Add neighbors to edges from common vertices
#         # for v in self.vertices:
#         #     for e_id in v.e_ids:
#         #         s = set(v.e_ids)
#         #         s.difference_update([e_id])
#         #         self._edges[e_id].add_neighbors(list(s))

#         # print('neighbor edges found',
#         #     time.time() - start, 's')        
#         # A dictionary
#         self._faces = Face.from_tess_file(
#             filename = filename,
#             _edges=self._edges,
#             measure=measures,
#             theta=theta3D,
#             lower_thrd=lower_thrd,
#             upper_thrd=upper_thrd
#         )
#         # A list
#         self.faces = [f for f in self._faces.values()]
        
#         # print(len(self._faces.keys()), 'faces loaded:',
#         #     time.time() - start, 's')
#         # Add neighbors to faces from common edges
#         # for e in self.edges:
#         #     for f_id in e.f_ids:
#         #         s = set(e.f_ids)
#         #         s.difference_update([f_id])
#         #         self._faces[f_id].add_neighbors(list(s))
        
#         # print('neighbor faces found',
#         #     time.time() - start, 's') 
        
#         # In 2D case faces have seeds and orientations
#         if self.dim == 2:
#             with open(filename, 'r', encoding="utf-8") as file:
#                 for line in file:
#                     if '**cell' in line:
#                         N = int(file.readline().rstrip('\n'))
#                     if '*seed' in line:
#                         for i in range(N):
#                             row = file.readline().split()
#                             f_id = int(row[0])
#                             seed_coord = tuple([*map(float, row[1:3])])
#                             self._faces[f_id].set_seed(seed_coord)
#                     if '*ori' in line:
#                         ori_format = file.readline().strip() #.rstrip('\n')
#                         for i in range(N):
#                             row = file.readline().split()
#                             ori_c = tuple([*map(float, row)])
#                             self.faces[i].set_crystal_ori(ori_format, ori_c)
#                         break
            
#             # Set external edges and vertices
#             for e in self.edges:
#                 if len(e.f_ids) == 1:
#                     e.set_external(dim=self.dim)
#                     for v_id in e.v_ids:
#                         self._vertices[v_id].set_external()
#                     for f_id in e.f_ids:
#                         self._faces[f_id].set_external(dim=self.dim)
        
#         # In 3D there are polyhedra, that have seeds and orientations
#         elif self.dim == 3:
#             # A dictionary
#             self._polyhedra = Poly.from_tess_file(
#                 filename,
#                 self._faces,
#                 measure=measures
#             )
#             # A list
#             self.polyhedra = [p for p in self._polyhedra.values()]
            
#             # Add neighbors to polyhedra from common faces
#             # for f in self.faces:
#             #     for p_id in f.p_ids:
#             #         s = set(f.p_ids)
#             #         s.difference_update([p_id])
#             #         self._polyhedra[p_id].add_neighbors(list(s))
            
#             # print('neighbor polyhedra found',
#             #     time.time() - start, 's')

#             # Set external faces, edges and vertices
#             for f in self.faces:
#                 if len(f.p_ids) == 1:
#                     f.set_external(dim=self.dim)
#                     for v_id in f.v_ids:
#                         self._vertices[v_id].set_external()
#                     for e_id in f.e_ids:
#                         self._edges[e_id].set_external(dim=self.dim)
#                     for p_id in f.p_ids:
#                         self._polyhedra[p_id].set_external()
        
#         # Set junction types from theta (if known from a file)
#         # If lower or upper threshold are known or both
#         if lower_thrd or upper_thrd:
#             self.set_junction_types()

#         self.load_time = round(time.time() - start, 1)
#         print('Complex loaded:', self.load_time, 's')


#     def plot_seeds(
#             self,
#             cell_ids: List = [],
#             ax: Axes = None,
#             figsize: Tuple = (8,8),
#             **kwargs):
#         """
#         """
#         if not ax:
#             ax = _create_ax(self.dim, figsize)
#         if self.dim == 2 and cell_ids:
#             cell_list = self.get_many('f', cell_ids)
#         elif self.dim == 2 and not cell_ids:
#             cell_list = self.faces
#         elif self.dim == 3 and cell_ids:
#             cell_list = self.get_many('p', cell_ids)
#         elif self.dim == 3 and not cell_ids:
#             cell_list = self.polyhedra
        
#         xs = []
#         ys = []
#         if self.dim == 3:
#             zs = []
#         for cell in cell_list:
#             if self.dim == 2:
#                 x, y = cell.seed
#             elif self.dim == 3:
#                 x, y, z = cell.seed
#                 zs.append(z)
#             xs.append(x)
#             ys.append(y)
#         if self.dim == 2:
#             ax.scatter(xs, ys, **kwargs)
#         elif self.dim == 3:
#             ax.scatter(xs, ys, zs, **kwargs)
#         return ax

  
#     def get_sparse_A(self, cell_type: Union[str, int]):
#         """
#         """
#         _cells = self._choose_cell_type(cell_type)
   
#         return matutils.get_A_from_cells(_cells)

#     def get_sparse_B(self, cell_type: Union[str, int]):
#         """
#         """
#         _cells = self._choose_cell_type(cell_type)
   
#         return matutils.get_B_from_cells(_cells)

#     def get_graph_from_A(self, cell_type: Union[str, int]):
#         """
#         """
#         _cells = self._choose_cell_type(cell_type)
        
#         return matutils.get_G_from_cells(_cells)

#     def save_to_files(self, work_dir: str = '.', representation=None):
#         """
#         Can be saved as a set of srapse matrices or cell lists.
#         representation: operator form, cells list, neper format?
#         A, B
#         Change dependancies !!!
#         """
#         matutils.save_A(self, work_dir)
#         matutils.save_B(self, work_dir)

#         nc_filename = os.path.join(work_dir, 'number_of_cells.txt')
#         with open(nc_filename, 'w') as file:
#             file.write(f'{self.vernb}\n{self.edgenb}\n{self.facenb}')
#             if self.dim == 3:
#                 file.write(f'\n{self.polynb}')
        
#         normals_filename = os.path.join(work_dir, 'normals.txt')
#         with open(normals_filename, 'w') as file:
#             for face in self.faces:
#                 file.write(f'{face.id} {face.a} {face.b} {face.c}\n')

#         seeds_filename = os.path.join(work_dir, 'seeds.txt')
#         with open(seeds_filename, 'w') as file:
#             if self.dim == 2:
#                 for face in self.faces:
#                     file.write('%.12f %.12f\n' % face.seed) 
#             elif self.dim == 3:
#                 for poly in self.polyhedra:
#                     file.write('%.12f %.12f %.12f\n' % poly.seed) 


# representation

# """
# Get cells from Neper .tess file in different formats.
# 1. Cells list representation stores a p-complex using p dictionaries.
# The key of each dictionary corresponds to a k-cell and each value contains
# the vertex list (0-cells) of the corresponding k-cell. 

# Plotting https://likegeeks.com/3d-plotting-in-python/ 

# """

# import numpy as np
# from typing import Dict, List, Tuple

# from matplotlib.axes import Axes
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# def get_v_coordinates(filename: str) -> Dict:
#     """
#     Get coordinates of vertices from Neper .tess file in the form
#     of a dictionary:
#     {vertex_id: {'x': x_coord, 'y': y_coord, 'z': z_coord}}

#     Parameters
#     ----------
#     filename
#         Name of Neper file.

#     Returns
#     -------
#     vertices
#         Dictionary of vertices and their coordinates.
#     """
#     vertices = {}
#     with open(filename, 'r', encoding="utf-8") as f:
#         for line in f:
#             if '**vertex' in line:
#                 n = int(f.readline().rstrip('\n'))
#                 for i in range(n):
#                     row = f.readline().split()
#                     v_id = int(row[0])
#                     vertices[v_id] = {}
#                     vertices[v_id]['x'] = float(row[1])
#                     vertices[v_id]['y'] = float(row[2])
#                     vertices[v_id]['z'] = float(row[3])
#                 return vertices


# def get_edges_from_tess(filename: str) -> Dict:
#     """
#     Get edges of a complex from Neper .tess file in the form
#     of a dictionary:
#     {edge_id: [ver_1, ver_2]}

#     Parameters
#     ----------
#     filename
#         Name of Neper file.

#     Returns
#     -------
#     edges
#         Dictionary of edges and their vertices.
#     """
#     edges = {}
#     with open(filename, 'r', encoding="utf-8") as f:
#             for line in f:
#                 if '**edge' in line:
#                     n = int(f.readline().rstrip('\n'))
#                     for i in range(n):
#                         row = f.readline().split()
#                         e_id = int(row[0])
#                         edges[e_id] = [int(row[1]), int(row[2])]
#                     return edges


# def get_faces_from_tess(filename: str) -> Dict:
#     """
#     Get faces of a complex from Neper .tess file in the form
#     of a dictionary:
#     {face_id: [ver_1, ver_2 ...]}

#     Parameters
#     ----------
#     filename
#         Name of Neper file.

#     Returns
#     -------
#     faces
#         Dictionary of faces and their vertices.
#     """
#     faces = {}
#     with open(filename, 'r', encoding="utf-8") as f:
#         for line in f:
#             if '**face' in line:
#                 n = int(f.readline().rstrip('\n'))
#                 for i in range(n):
#                     row = f.readline().split()
#                     f_id = int(row[0])
#                     faces[f_id] = []
#                     for k in range(2, int(row[1]) + 2):
#                         faces[f_id].append(int(row[k]))
#                     _ = f.readline()
#                     _ = f.readline()
#                     _ = f.readline()
#                 return faces


# def get_poly_from_tess(filename: str) -> Dict:
#     """
#     Get polyhedra of a complex from Neper .tess file in the form
#     of a dictionary:
#     {poly_id: [ver_1, ver_2 ...]}

#     Parameters
#     ----------
#     filename
#         Name of Neper file.

#     Returns
#     -------
#     polyhedra
#         Dictionary of polyhedra and their vertices.
#     """
#     polyhedra = {}
#     faces = get_faces_from_tess(filename)
#     with open(filename, 'r', encoding="utf-8") as f:
#             for line in f:
#                 if '**polyhedron' in line:
#                     n = int(f.readline().rstrip('\n'))
#                     for i in range(n):
#                         row = f.readline().split()
#                         p_id = int(row[0])
#                         polyhedra[p_id] = []
#                         for k in range(2, int(row[1]) + 2):
#                             polyhedra[p_id] += faces[abs(int(row[k]))]
#                         polyhedra[p_id] = list(set(polyhedra[p_id]))
#                     return polyhedra


# def _from_dict_to_coord_tuples(
#         vertices: Dict, 
#         subset=[]) -> List[Tuple]:
#     """
#     """
#     v_list = []
#     keys_list = vertices.keys() if not subset else subset
#     for v in keys_list:
#         v_list.append(
#             (vertices[v]['x'], 
#              vertices[v]['y'], 
#              vertices[v]['z'])
#         )
#     return v_list


# def _transform_coord(
#         vertices: Dict, 
#         vertices_subset: List = []) -> Tuple[List]:
#     """
#     Transform coordinates of vertices in the form (xs, ys, zs),
#     where xs, ys and zs are the lists of x-, y- and z-coordinates
#     respectively.

#     Parameters
#     ----------
#     vertices
#         Dictionary of vertices and their coordinates.
#     vertices_subset, optional
#         List of vertices to transfrom. If not present,
#         output will contain all the vertices from the dictionary.

#     Returns
#     -------
#     xs
#         List of x-coordinates of vertices.
#     ys
#         List of y-coordinates of vertices.
#     zs
#         List of z-coordinates of vertices.
#     """
#     xs = []
#     ys = []
#     zs = []
#     if not vertices_subset:
#         vertices_subset = vertices.keys()
#     for v in vertices_subset:
#         xs.append(vertices[v]['x'])
#         ys.append(vertices[v]['y'])
#         zs.append(vertices[v]['z'])
#     return (xs, ys, zs)


# def create_3D_axis(
#         figsize: Tuple = (8, 8),
#         xlim: Tuple = (0, 1),
#         ylim: Tuple = (0, 1),
#         zlim: Tuple = (0, 1)
#         ):
#     """
#     """
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(*xlim)
#     ax.set_ylim(*ylim)
#     ax.set_zlim(*zlim)
#     return fig, ax


# def create_2D_axis(
#         figsize: Tuple = (8, 8),
#         xlim: Tuple = (0, 1),
#         ylim: Tuple = (0, 1)
#         ):
#     """
#     """
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111)
#     # ax.set_xlim(*xlim)
#     # ax.set_ylim(*ylim)
#     # plt.grid()
#     return fig, ax

# def plot_points(
#         ax: Axes,
#         vertices: Dict,
#         vert_id_list: List,
#         color=None,
#         label: List = None) -> None:
#     """
#     """
#     xs, ys, zs = _transform_coord(vertices, vert_id_list)
#     ax.scatter(xs, ys, zs, label=label, c=color)
#     if label:
#         ax.legend(loc="best")
#     plt.show()

# def plot_edge(
#         ax: Axes,
#         vertices: Dict,
#         edge: List,
#         color: str = None,
#         label: List = None) -> None:
#     """
#     """
#     x = np.linspace(vertices[edge[0]]['x'], vertices[edge[1]]['x'], 50)
#     y = np.linspace(vertices[edge[0]]['y'], vertices[edge[1]]['y'], 50)
#     z = np.linspace(vertices[edge[0]]['z'], vertices[edge[1]]['z'], 50)
#     ax.plot(x, y, z, color=color, label=label)
#     if label:
#         ax.legend(loc="best")
#     plt.show()

# def plot_face(
#         ax: Axes,
#         vertices: Dict,
#         face: List,
#         color: str =None,
#         alpha: float = 0.2) -> None:
#     """
#     """
#     vl = _from_dict_to_coord_tuples(vertices, face)
#     poly = Poly3DCollection([vl], alpha=alpha, color=color)
#     ax.add_collection3d(poly)

"""
class CellComplex
"""

# import os
# from typing import Dict, Iterable, List, Tuple, Union
# import time

# import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    


#     def plot_vertices(
#             self,
#             v_ids: List = [],
#             ax: Axes = None,
#             figsize: Tuple = (8,8),
#             labels: bool = False,
#             **kwargs):
#         """
#         """
#         if not ax:
#             ax = _create_ax(self.dim, figsize)
#         if v_ids:
#             v_list = self.get_many('v', v_ids)
#         else:
#             v_list = self.vertices
#         for v in v_list:
#             if labels:
#                 ax = v.plot(dim=self.dim, ax=ax, label=v.id, **kwargs)
#             else:
#                 ax = v.plot(dim=self.dim, ax=ax, **kwargs)
#         if labels:
#             ax.legend(loc='best')
#         return ax

    
#     def plot_edges(
#             self,
#             e_ids: List = [],
#             ax: Axes = None,
#             figsize: Tuple = (8,8),
#             labels: bool = False,
#             **kwargs):
#         """
#         """
#         if not ax:
#             ax = _create_ax(self.dim, figsize)
#         if e_ids:
#             e_list = self.get_many('e', e_ids)
#         else:
#             e_list = self.edges
#         for e in e_list:
#             v1, v2 = self.get_many('v', e.v_ids)
#             x_space = np.linspace(v1.x, v2.x, 50)
#             y_space = np.linspace(v1.y, v2.y, 50)
#             if self.dim == 2:
#                 if labels:
#                     ax.plot(x_space, y_space, label=e.id, **kwargs)
#                 else:
#                     ax.plot(x_space, y_space, **kwargs)
#             elif self.dim == 3:
#                 z_space = np.linspace(v1.z, v2.z, 50)
#                 if labels:
#                     ax.plot(x_space, y_space, z_space, label=e.id, **kwargs)
#                 else:
#                     ax.plot(x_space, y_space, z_space, **kwargs)
#         if labels:
#             ax.legend(loc='best')
#         return ax

#     # def plot_vertices_3D(
#     #         self,
#     #         v_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_3D_axis()
#     #     if v_ids:
#     #         v_list = self.get_many('v', v_ids)
#     #     else:
#     #         v_list = self.vertices
#     #     for v in v_list:
#     #         ax = v.plot3D(ax, **kwargs)
        
#     #     return ax

#     # def plot_edges_3D(
#     #         self,
#     #         e_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     if not ax:
#     #         fig, ax = representation.create_3D_axis()
#     #     if e_ids:
#     #         e_list = self.get_many('e', e_ids)
#     #     else:
#     #         e_list = self.edges
#     #     for e in e_list:
#     #         v1, v2 = self.get_many('v', e.v_ids)

#     #         x_space = np.linspace(v1.x, v2.x, 50)
#     #         y_space = np.linspace(v1.y, v2.y, 50)
#     #         z_space = np.linspace(v1.z, v2.z, 50)
            
#     #         ax.plot(x_space, y_space, z_space, **kwargs)
            
#     #     return ax
    
    
#     def plot_faces(
#             self,
#             f_ids: List = [],
#             ax: Axes = None,
#             figsize: Tuple = (8,8),
#             labels: bool = False,
#             **kwargs):
#         """
#         labels doesn't work with 3d faces
#         """
#         if not ax:
#             ax = _create_ax(self.dim, figsize)
#         if f_ids:
#             f_list = self.get_many('f', f_ids)
#         else:
#             f_list = self.faces
#         f_coord_list = []
#         for f in f_list:
#             v_list = self.get_many('v', f.v_ids)

#             if self.dim == 2:
#                 xs = [v.x for v in v_list]
#                 ys = [v.y for v in v_list]
#                 if labels:
#                     ax.fill(xs, ys, label=f.id, **kwargs)
#                 else:
#                     ax.fill(xs, ys, **kwargs)
#             elif self.dim == 3:
#                 coord_list = [v.coord for v in v_list]
#                 f_coord_list.append(coord_list)

#         if self.dim == 3:
#             poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
#             ax.add_collection3d(poly)
#         elif labels:
#             ax.legend(loc='best')
#         return ax

#     # def plot_faces_3D(
#     #         self,
#     #         f_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_3D_axis()
#     #     if f_ids:
#     #         f_list = self.get_many('f', f_ids)
#     #     else:
#     #         f_list = self.faces
#     #     f_coord_list = []
#     #     for f in f_list:
#     #         v_list = self.get_many('v', f.v_ids)
#     #         coord_list = [v.coord for v in v_list]
#     #         f_coord_list.append(coord_list)
#     #     poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
#     #     ax.add_collection3d(poly)

#     #     return ax

#     def plot_polyhedra(
#             self,
#             p_ids: List = [],
#             ax: Axes = None,
#             figsize: Tuple = (8,8),
#             **kwargs):
#         """
#         """
#         if not ax:
#             ax = _create_ax(self.dim, figsize)
#         if p_ids:
#             p_list = self.get_many('p', p_ids)
#         else:
#             p_list = self.polyhedra
        
#         for p in p_list:
#             ax = self.plot_faces(f_ids=p.f_ids, ax=ax, **kwargs)

#         return ax

#     # def plot_vertices_2D(
#     #         self,
#     #         v_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_2D_axis()
#     #     if v_ids:
#     #         v_list = self.get_many('v', v_ids)
#     #     else:
#     #         v_list = self.vertices
#     #     for v in v_list:
#     #         ax = v.plot2D(ax, **kwargs)
        
#     #     return ax

#     # def plot_edges_2D(
#     #         self,
#     #         e_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     if not ax:
#     #         fig, ax = representation.create_2D_axis()
#     #     if e_ids:
#     #         e_list = self.get_many('e', e_ids)
#     #     else:
#     #         e_list = self.edges
#     #     for e in e_list:
#     #         v1, v2 = self.get_many('v', e.v_ids)

#     #         x_space = np.linspace(v1.x, v2.x, 50)
#     #         y_space = np.linspace(v1.y, v2.y, 50)
            
#     #         ax.plot(x_space, y_space, **kwargs)
            
#     #     return ax


#     # def plot_faces_2D(
#     #         self,
#     #         f_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_2D_axis()
#     #     if f_ids:
#     #         f_list = self.get_many('f', f_ids)
#     #     else:
#     #         f_list = self.faces
#     #     for f in f_list:
#     #         # _ = self.plot_edges_2D(f.e_ids, ax, **kwargs)
#     #         v_list = self.get_many('v', f.v_ids)
#     #         xs = [v.x for v in v_list]
#     #         ys = [v.y for v in v_list]
#     #         ax.fill(xs, ys, **kwargs)

#     #     return ax

#     #     
#     #     for f in f_list:
#     #         v_list = self.get_many('v', f.v_ids)
#     #         coord_list = [v.coord for v in v_list]
#     #         f_coord_list.append(coord_list)
#     #     poly = Poly3DCollection(f_coord_list, alpha = 0.2, **kwargs)
#     #     ax.add_collection3d(poly)

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


#     # def plot_seeds_2D(
#     #         self,
#     #         f_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         ax = plt.subplot(111)
#     #         # ax.set_xlim(0, 1)
#     #         # ax.set_ylim(0, 1)
#     #     if f_ids:
#     #         f_list = self.get_many('f', f_ids)
#     #     else:
#     #         f_list = self.faces
        
#     #     xs = []
#     #     ys = []
#     #     for f in f_list:
#     #         x, y = f.seed
#     #         xs.append(x)
#     #         ys.append(y)
#     #     ax.scatter(xs, ys, **kwargs)
#     #     return ax

#     # def plot_seeds_3D(
#     #         self,
#     #         p_ids: List = [],
#     #         ax: Axes = None,
#     #         **kwargs):
#     #     """
#     #     """
#     #     if not ax:
#     #         fig, ax = representation.create_2D_axis()
#     #     if p_ids:
#     #         p_list = self.get_many('p', p_ids)
#     #     else:
#     #         p_list = self.polyhedra
        
#     #     xs = []
#     #     ys = []
#     #     zs = []
#     #     for p in p_list:
#     #         x, y, z = p.seed
#     #         xs.append(x)
#     #         ys.append(y)
#     #         zs.append(z)
#     #     ax.scatter(xs, ys, zs, **kwargs)
#     #     return ax
  
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
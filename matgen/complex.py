
# class CellComplex():
#     """

  
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



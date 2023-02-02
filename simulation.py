# """
# """
# import os
# import argparse
# from math import log2, isclose
# from re import S
# import random
# import numpy as np

# from matgen.core import CellComplex






# def get_new_seed_3D(c: CellComplex, f_id: int):
#     """
#     """
#     face = c.get_one('f', f_id)
#     if face.a == 0 and face.b == 0 and face.c == 0:
#         raise ValueError('Invalid face equation')
#     else:
#         vs = c.get_many('v', face.v_ids)
#         xp = np.array([v.x for v in vs])
#         yp = np.array([v.y for v in vs])
#         zp = np.array([v.z for v in vs])
    
#         xmin = np.min(xp)
#         xmax = np.max(xp)
#         ymin = np.min(yp)
#         ymax = np.max(yp)    
#         zmin = np.min(zp)
#         zmax = np.max(zp)    
    
    
#     if not face.c and not face.b: # c == 0, b == 0
#         while True:
#             y = _get_random(ymin, ymax)
#             z = _get_random(zmin, zmax)
#             if _in_polygon(y, z, yp, zp):
#                 break
#         x = (face.d - face.b * y - face.c * z) / face.a
#     elif not face.c: # c == 0, b != 0
#         while True:
#             x = _get_random(xmin, xmax)
#             z = _get_random(zmin, zmax)
#             if _in_polygon(x, z, xp, zp):
#                 break
#         y = (face.d - face.a * x - face.c * z) / face.b
#     else: # c != 0
#         while True:
#             x = _get_random(xmin, xmax)
#             y = _get_random(ymin, ymax)
#             if _in_polygon(x, y, xp, yp):
#                 break
#         z = (face.d - face.a * x - face.b * y) / face.c

#     # if face.a and face.b and face.c: # a != 0, b != 0, c != 0
#     #     while True:
#     #         x = _get_random(xmin, xmax)
#     #         y = _get_random(ymin, ymax)
#     #         if _in_polygon(x, y, xp, yp):
#     #             break
#     #     z = (face.d - face.a * x - face.b * y) / face.c
#     # elif not face.a: # a == 0
#     #     if not face.c: # a == 0, c == 0
#     #         while True:
#     #             x = _get_random(xmin, xmax)
#     #             z = _get_random(zmin, zmax)
#     #             if _in_polygon(x, z, xp, zp):
#     #                 break
#     #         y = face.d / face.b
#     #     elif not face.b: # a == 0, b == 0
#     #         while True:
#     #             x = _get_random(xmin, xmax)
#     #             y = _get_random(ymin, ymax)
#     #             if _in_polygon(x, y, xp, yp):
#     #                 break
#     #         z = (face.d - face.b * y) / face.c
#     # elif not face.c: # a != 0, c == 0
#     #     y = _get_random(ymin, ymax)
#     #     x = (face.d - face.b * y) / face.a
#     #     z = _get_random(zmin, zmax)
#     # else: # a != 0, b == 0, c != 0
#     #     x = _get_random(xmin, xmax)
#     #     y = _get_random(ymin, ymax)
#     #     z = (face.d - face.a * x) / face.c

#     return (x, y, z)


# def get_new_seed_2D(c: CellComplex, e_id: int):
#     """
#     """
#     edge = c.get_one('e', e_id)
#     vs = c.get_many('v', edge.v_ids)
#     xp = np.array([v.x for v in vs])
#     yp = np.array([v.y for v in vs])

#     if isclose(xp[0], xp[1]):
#         x = xp[0]
#         y = _get_random(yp[0], yp[1])
#         # if isclose(y, yp[0]):
#         #     y = _get_random(yp[0], yp[1])
#     else:
#         x = _get_random(xp[0], xp[1])
#         # if isclose(x, xp[0]):
#         #     x = _get_random(xp[0], xp[1])    
#         y = (yp[1] - yp[0]) / (xp[1] - xp[0]) * (x - xp[0]) + yp[0]

#     return (x, y)
    

# def main():
#     """
#     """
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         '-id',
#         nargs=1,
#         type=int,
#         default=1
#     )

#     parser.add_argument(
#         '-n',
#         nargs=1,
#         type=int,
#         default=8
#     )
    
#     parser.add_argument(
#         '--filename',
#         default='complex'
#     )

#     args = parser.parse_args()
#     os.system(
#         'neper -T -n %d -id %d -o %s -reg 1' % (
#             args.n, args.id, args.filename) +\
#         ' -statpoly id,vol -statface id,area -statedge id,length'
#     )
#     # extract_seeds(args.filename + '.tess', '.')
#     # write_matrices(args.filename + '.tess', '.', True)

#     # -periodicity all - check
#     # -morphooptiini "coo:file(seeds)"

# if __name__ == '__main__':
#     main()
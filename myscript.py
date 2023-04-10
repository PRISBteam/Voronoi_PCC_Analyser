import numpy as np
from matgen.base import CellComplex

rng = np.random.default_rng()

# for id_ in [1, 2, 42]:
#     for n, ks in zip([100, 1000], [[14, 27, 81], [145, 290, 870]]):
#         filename = f'sim_examples/n{n}-id{id_}.tess'
#         c = CellComplex.from_tess_file(filename)
#         for k in ks:
#             for i in [1, 2, 3]:
#                 ids = rng.choice(c.get_internal_ids('e'), k, replace=False)
#                 np.savetxt(f'sim_examples/n{n}-id{id_}-spec-{k}-{i}.txt', ids, fmt='%d')

id_ = 1
n = 10000
ks = [1483, 2965, 8895]
filename = f'sim_examples/n{n}-id{id_}.tess'
c = CellComplex.from_tess_file(filename)
for k in ks:
    for i in [1, 2, 3]:
        ids = rng.choice(c.get_internal_ids('e'), k, replace=False)
        np.savetxt(f'sim_examples/n{n}-id{id_}-spec-{k}-{i}.txt', ids, fmt='%d')
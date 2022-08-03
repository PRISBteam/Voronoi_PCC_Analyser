from matgen import core, simulation
import subprocess
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import shutil


n0 = 8
neper_id = 1
dim = 2
wdir = 'simul2D_2'
seeds_ini = wdir + '/' + 'seeds_initial.txt'
seeds_filename = wdir + '/' + 'seeds.txt'

shutil.copy(seeds_ini, seeds_filename) 

st = time.time()

for m in tqdm(range(10)):
    n = n0 + m
    output_file = wdir + '/' + f'n{n}-id{neper_id}-{dim}D.tess'
    com_line_list = ['neper', '-T', '-n', str(n), '-id', str(neper_id), '-dim', str(dim), '-morphooptiini', f'coo:file({seeds_filename})', '-o', output_file.rstrip('.tess')]
    run = subprocess.run(com_line_list, capture_output=True)
    c = core.CellComplex(output_file)
    e_int = c.get_internal_ids('e')
    e_ext = c.get_external_ids('e')

    for f_id in range(n0 + 1, n + 1):
        e_ids = c.get_one('f', f_id).e_ids
        for e_id in e_ids:
            if e_id not in e_ext:
                c.get_one('e', e_id).set_special()
    
    e_spec = c.get_special_ids()
    
    e_sel = list(set(e_int) - set(e_spec))
    e_id_sampled = random.sample(e_sel, 1)[0]
    x, y = simulation.get_new_seed_2D(c, e_id_sampled)
    ax = c.plot_edges(e_sel, color='b')
    ax = c.plot_edges(e_spec, color='g', ax=ax)
    ax = c.plot_edges(e_ext, color='k', ax=ax)
    ax.scatter(x, y, color='r')
    plt.savefig(output_file.rstrip('.tess') + '.png')
    plt.close()
    with open(seeds_filename, 'a') as file:
        file.write('%.12f %.12f\n' % (x, y))
    print('Total grains:', n, 'New grains:', m)
    print('New grain fraction:', m / n)
    print('Special GB fraction:', len(e_spec) / len(e_int))

print('\nTotal time:', time.time() - st)
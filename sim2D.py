from matgen import core, simulation
import subprocess
import random
import matplotlib.pyplot as plt

n0 = 8
neper_id = 1
dim = 2
seeds_filename = 'simul2D/seeds.txt'
for i in range(38):
    n = n0 + i
    output_file = f'simul2D/n{n}-id{neper_id}-{dim}D.tess'
    com_line_list = ['neper', '-T', '-n', str(n), '-id', str(neper_id), '-dim', str(dim), '-morphooptiini', f'coo:file({seeds_filename})', '-o', output_file.rstrip('.tess')]
    run = subprocess.run(com_line_list, capture_output=True)
    c = core.CellComplex(output_file)
    e_int = c.get_internal_ids('e')
    e_ext = c.get_external_ids('e')
    e_id_sampled = random.sample(e_int, 1)[0]
    x, y = simulation.get_new_seed_2D(c, e_id_sampled)
    ax = c.plot_edges(e_int, color='b')
    ax = c.plot_edges(e_ext, color='k', ax=ax)
    ax.scatter(x, y, color='r')
    plt.savefig(output_file.rstrip('.tess') + '.png')
    plt.close()
    with open(seeds_filename, 'a') as file:
        file.write('%.12f %.12f\n' % (x, y))
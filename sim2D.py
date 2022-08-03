from matgen import core, simulation
import subprocess
import random
import matplotlib.pyplot as plt
import time
# from tqdm import tqdm
import shutil
import argparse


def main():
    n0 = 100
    neper_id = 1
    dim = 2
    wdir = 'simul2D_3'
    seeds_ini = wdir + '/' + 'seeds_initial.txt'
    seeds_filename = wdir + '/' + 'seeds.txt'

    shutil.copy(seeds_ini, seeds_filename) 

    st = time.time()


    m = 0

    # for m in tqdm(range(15)):
    while True:
        m += 1
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
        if e_sel:
            e_id_sampled = random.sample(e_sel, 1)[0]
        else:
            print('All GB are special')
            ax = c.plot_edges(e_spec, color='g')
            ax = c.plot_edges(e_ext, color='k', ax=ax)
            ax.set_title(f'New G: {m} | G frac: {G_frac} | GB frac: {GB_frac}')
            plt.savefig(output_file.rstrip('.tess') + '.png')
            plt.close()
            break
        
        G_frac = round(m / n, 3)
        GB_frac = round(len(e_spec) / len(e_int), 3)

        x, y = simulation.get_new_seed_2D(c, e_id_sampled)
        ax = c.plot_edges(e_sel, color='b')
        if e_spec:
            ax = c.plot_edges(e_spec, color='g', ax=ax)
        ax = c.plot_edges(e_ext, color='k', ax=ax)
        ax.scatter(x, y, color='r')
        ax.set_title(f'New G: {m} | G frac: {G_frac} | GB frac: {GB_frac}')
        plt.savefig(output_file.rstrip('.tess') + '.png')
        plt.close()
        with open(seeds_filename, 'a') as file:
            file.write('%.12f %.12f\n' % (x, y))
        print('Total grains:', n, 'New grains:', m)
        print('New grain fraction:', G_frac)
        print('Special GB fraction:', GB_frac)

    print('\nTotal time:', time.time() - st, 's')

if __name__ == '__main__':
    main()
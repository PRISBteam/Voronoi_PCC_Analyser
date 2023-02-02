from matgen import base
import subprocess
import random
import matplotlib.pyplot as plt
import time
# from tqdm import tqdm
import shutil
import argparse
import numpy as np


def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-id',
        #nargs=1,
        type=int,
        default=1
    )

    parser.add_argument(
        '-n',
        #nargs=1,
        type=int,
        default=8
    )
    
    parser.add_argument(
        '--dir',
        default='.'
    )

    parser.add_argument(
        '-dim',
        #nargs=1,
        type=int,
        default=2
    )

    args = parser.parse_args()

    print(args.n, args.id, args.dim, args.dir)

    n0 = args.n
    neper_id = args.id
    dim = args.dim
    wdir = args.dir
    seeds_ini = wdir + '/' + 'seeds_initial.txt'
    seeds_filename = wdir + '/' + 'seeds.txt'

    shutil.copy(seeds_ini, seeds_filename) 

    st = time.time()



    m = 0

    # for m in tqdm(range(15)):
    while True:
        n = n0 + m
        output_file = wdir + '/' + f'n{n}-id{neper_id}-{dim}D.tess'
        com_line_list = ['neper', '-T', '-n', str(n), '-id', str(neper_id), '-dim', str(dim), '-morphooptiini', f'coo:file({seeds_filename})', '-o', output_file.rstrip('.tess')]
        run = subprocess.run(com_line_list, capture_output=True)
        c = base.CellComplex.from_tess_file(output_file)
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
            ax.set_title(f'm = {m} | p = {G_frac} | P_HAGBs = {GB_frac}')
            plt.savefig(output_file.rstrip('.tess') + '.png')
            plt.close()
            break
        
        D0 = 0
        D1 = 0
        D2 = 0
        w = 0

        for e in c.edges:
            if not e.is_external:
                f_ids = np.array(e.incident_ids)
                d = (f_ids > n0).sum()
                if d == 2:
                    D2 += 1
                elif d == 1:
                    D1 += 1
                elif d == 0:
                    D0 += 1
                else:
                    raise ValueError("Number of new grains must be 0, 1 or 2")

        D2 = D2 / len(e_int)
        D1 = D1 / len(e_int)    
        D0 = D0 / len(e_int)

        G_frac = round(m / n, 3)
        GB_frac = round(len(e_spec) / len(e_int), 3)

        p = m / n
        D0r = (1 - p) * (1 - p)
        D1r = 2 * (1 - p) * p
        D2r = p * p
        
        if D1 <= D1r and p != 0:
            w = 1 - D1 / D1r
        elif D1 > D1r and p !=0:
            w = D0 * D2 / D0r / D2r - 1
        else:
            w = -9999

        x, y = c.get_new_random_seeds(k=1)[0]
        ax = c.plot_edges(e_sel, color='b')
        if e_spec:
            ax = c.plot_edges(e_spec, color='g', ax=ax)
        ax = c.plot_edges(e_ext, color='k', ax=ax)
        ax.scatter(x, y, color='r')
        ax.set_title(f'm = {m} | p = {G_frac} | P_HAGBs = {GB_frac}')
        plt.savefig(output_file.rstrip('.tess') + '.png')
        plt.close()
        with open(seeds_filename, 'a') as file:
            file.write('%.12f %.12f\n' % (x, y))
        print('Total grains:', n, 'New grains:', m)
        print('New grain fraction:', G_frac)
        print('Special GB fraction:', GB_frac)
        print(f'D0 = {D0}, D1 = {D1}, D2 = {D2}, w = {w}\n')

        m += 1

    print('\nTotal time:', time.time() - st, 's')

if __name__ == '__main__':
    main()
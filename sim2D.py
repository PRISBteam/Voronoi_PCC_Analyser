from matgen import base
from sparsemat import extract_seeds

import os
import subprocess
import random
import matplotlib.pyplot as plt
import time
# from tqdm import tqdm
import shutil
import argparse
import numpy as np

def extract_seeds(complex: base.CellComplex, wdir: str):
    """
    """
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    N = complex.grainnb
    seeds = [complex._grains[g_id].seed for g_id in range(1, N + 1)]
    np.savetxt(seeds_filename, seeds, fmt='%.12f')

def create_new_complex(
    wdir: str,
    n: int,
    neper_id: int,
    dim: int
):
    """
    """
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    output_file = wdir + '/' + f'n{n}-id{neper_id}-{dim}D.tess'

    com_line_list = [
        'neper', '-T', '-n', str(n),
        '-id', str(neper_id), '-dim', str(dim),
        '-morphooptiini', f'coo:file({seeds_filename})',
        '-statcell', 'size',
        '-o', output_file.rstrip('.tess')
    ]
      
    run = subprocess.run(com_line_list, capture_output=True)
    # print(run.stdout)
    cell_complex = base.CellComplex.from_tess_file(output_file, with_cell_size=True)
    return cell_complex


NUMBER_OF_STEPS = 10
NUMBER_OF_NEW_SEEDS = [1 for _ in range(NUMBER_OF_STEPS)]
SPEC_PROB = 0

def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-id', type=int, default=1)
    # parser.add_argument('-n', type=int)
    # parser.add_argument('-dim', type=int, default=2)

    parser.add_argument('--tessfile', type=str)
    parser.add_argument('--specfile', type=str)
    parser.add_argument('--dir', default='.')

    args = parser.parse_args()

    # print(args.n, args.id, args.dim, args.dir)
    wdir = args.dir
    
    initial_complex = base.CellComplex.from_tess_file(args.tessfile, with_cell_size=True)
    # extract_seeds(initial_complex, wdir)
    n0 = initial_complex.grainnb
    dim = initial_complex.dim

    special_ids = np.loadtxt(args.specfile, dtype=int).tolist()
    initial_complex.reset_special(special_ids=special_ids)

    pairs_with_special_GB = []
    for special_id in initial_complex.get_special_ids():
        pair = set(initial_complex._GBs[special_id].incident_ids)
        pairs_with_special_GB.append(pair)
    # TODO: find grain boundary id given ids of its incident cells

    neper_id = args.id
    output_file = wdir + '/' + f'n{n0}-id{neper_id}-{dim}D.tess'
    initial_complex.plot_edges()
    plt.savefig(output_file.rstrip('.tess') + '.png')
    #n0 = args.n
    #dim = args.dim
    
    # seeds_ini = wdir + '/' + 'seeds_initial.txt'
    
    
    seeds_filename = wdir + '/' + 'seeds.txt'

    # shutil.copy(seeds_ini, seeds_filename) 

    st = time.time()


    n = n0
    m = 0

    # for m in tqdm(range(15)):
    # while True:

    for step_idx in range(NUMBER_OF_STEPS):
        #TODO Change initial step 
        n += NUMBER_OF_NEW_SEEDS[step_idx]
        output_file = wdir + '/' + f'n{n}-id{neper_id}-{dim}D.tess'
        # com_line_list = ['neper', '-T', '-n', str(n), '-id', str(neper_id), '-dim', str(dim), '-morphooptiini', f'coo:file({seeds_filename})', '-o', output_file.rstrip('.tess')]
        # run = subprocess.run(com_line_list, capture_output=True)
        # c = base.CellComplex.from_tess_file(output_file)
        
        c = create_new_complex(wdir, n, neper_id, dim)

        # Set special GBs from initial complex
        for cell in c._GBs.values():
            if set(cell.incident_ids) in pairs_with_special_GB:
                    if not cell.is_external:
                        cell.set_special(True)

        # New cells volume fraction
        total_size = sum([grain.size for grain in c._grains.values()])
        if n > n0:
            new_cells_size = sum(
                [c._grains[g_id].size for g_id in range(n0 + 1, n + 1)]
            )
            new_cells_volume_fraction = new_cells_size / total_size



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

        k = 1
        new_seeds = np.array(c.get_new_random_seeds(k=k, spec_prob=0.9))
        ax = c.plot_edges(e_sel, color='b')
        if e_spec:
            ax = c.plot_edges(e_spec, color='g', ax=ax)
        ax = c.plot_edges(e_ext, color='k', ax=ax)
        if len(new_seeds) > 0:
            ax.scatter(new_seeds[:,0], new_seeds[:,1], color='r')
        ax.set_title(f'm = {m} | p = {G_frac} | P_HAGBs = {GB_frac}')
        plt.savefig(output_file.rstrip('.tess') + '.png')
        plt.close()
        with open(seeds_filename, 'a') as file:
            np.savetxt(file, new_seeds, fmt='%.12f')
            # for new_seed_coord in new_seeds:
            #     file.write('%.12f %.12f\n' % (new_seed_coord[0], new_seed_coord[1]))
        print('Total grains:', n, 'New grains:', n - n0)
        print('New grain fraction:', G_frac)
        print('Special GB fraction:', GB_frac)
        print(new_cells_volume_fraction)
        print(f'D0 = {D0}, D1 = {D1}, D2 = {D2}, w = {w}\n')

        #m += k

    print('\nTotal time:', time.time() - st, 's')

if __name__ == '__main__':
    main()
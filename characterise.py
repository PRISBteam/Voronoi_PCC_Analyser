"""
"""

import argparse
import os
import time
import glob
import logging
import pandas as pd

from matgen import base

def main() -> None:
    """
    """
    start = time.time()
    
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--file",
    #     default='complex.tess',
    #     help="Filepath to .tess file with a tesselation (Neper output)"
    # )

    parser.add_argument(
        "--dir",
        default='.',
        help="Path to folder which contain complexes to characterise"
    )

    args = parser.parse_args()
    filenames = glob.glob(args.dir + '/*')

    TJsets = {}
    complex_id = 0

    for filename in filenames:
        file = open(filename, 'r')
        is_tess = 'tess' in file.read()
        if is_tess:
            pass
        else:
            try:
                file.seek(0)
                for line in file:
                    complex_id += 1
                    row = [*map(float, line.split())]
                    if len(row) != 5:
                        raise ValueError()
                    p = row[0]
                    j_tuple = tuple(row[1:])
                    TJsets[complex_id] = base.TripleJunctionSet(p, j_tuple)
            except ValueError:
                logging.exception(f'Wrong file: {filename}')
            except:
                logging.exception('Some error')

    ids = [*range(1, len(TJsets.keys()) + 1)]
    df = pd.DataFrame(
        {
            'p' : [TJsets[i].p for i in ids],
            'q' : [TJsets[i].q for i in ids],
            'Sp' : [TJsets[i].p_entropy for i in ids],
            'Sp_m' : [TJsets[i].p_entropy_m for i in ids],
            'Sp_s' : [TJsets[i].p_entropy_s for i in ids],
            'S' : [TJsets[i].S for i in ids],
            'S_m' : [TJsets[i].S_m for i in ids],
            'S_s' : [TJsets[i].S_s for i in ids],
            'kappa' : [TJsets[i].kappa for i in ids],
            'delta_S': [TJsets[i].delta_S for i in ids],
            'd1': [TJsets[i].d1 for i in ids],
            'd2': [TJsets[i].d2 for i in ids],
            'd3': [TJsets[i].d3 for i in ids]
        }
    )
    
    df.to_csv('characteristics.txt', index=False, sep=' ')

    print('Time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()

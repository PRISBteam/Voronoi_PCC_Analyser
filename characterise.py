"""
"""

import argparse
import os
import time
import glob
import logging

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

    print('Time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()

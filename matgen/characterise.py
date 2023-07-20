"""
Module for cell complex characterisation.
"""

import argparse
import time
import glob
import logging
import pandas as pd

from matgen.entropic import TripleJunctionSet

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
        default=None,
        help="Path to folder which contain complexes to characterise"
    )

    args = parser.parse_args()
    if args.dir:
        filenames = glob.glob(args.dir + '/*')
    else:
        filenames = []

    TJsets = []

    for filename in filenames:
        file = open(filename, 'r')
        is_tess = 'tess' in file.read()
        if is_tess:
            pass
        else:
            try:
                file.seek(0)
                for line in file:
                    row = [*map(float, line.split())]
                    if len(row) != 5:
                        raise ValueError()
                    p = row[0]
                    j_tuple = tuple(row[1:])
                    TJsets.append(TripleJunctionSet(p, j_tuple).__dict__)
            except ValueError:
                logging.exception(f'Wrong file: {filename}')
            except:
                logging.exception('Some error')

    df = pd.DataFrame(TJsets)
    df = df.sort_values(by='p')
    df.to_csv('characteristics.txt', index=False, sep=' ')

    print('Time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()

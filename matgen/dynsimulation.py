"""
"""
import os
import argparse

import sparsemat

def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-id',
        nargs=1,
        type=int,
        default=1
    )

    parser.add_argument(
        '-n',
        nargs=1,
        type=int,
        default=8
    )
    
    parser.add_argument(
        '--filename',
        default='complex'
    )

    args = parser.parse_args()
    os.system(
        'neper -T -n %d -id %d -o %s -reg 1' % (
            args.n, args.id, args.filename) +\
        ' -statpoly id,vol -statface id,area -statedge id,length'
    )
    sparsemat.extract_seeds(args.filename + '.tess', '.')
    sparsemat.write_matrices(args.filename + '.tess', '.', True)

    # -periodicity all
    # -morphooptiini "coo:file(seeds)"

if __name__ == '__main__':
    main()
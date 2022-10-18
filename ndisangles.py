"""
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from matgen import base, matutils


def main() -> None:
    """
    """
    start = time.time()
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        help="Filepath to .tess file with a tesselation (Neper output)"
    )

    parser.add_argument(
        "--dir",
        default='.',
        help="Path to folder which will contain generated matrices"
    )

    parser.add_argument(
        "--max-order",
        required=True,
        type=int,
        dest='order',
        help="Maximum order of neighbors"
    )

    parser.add_argument(
        "--min-order",
        default=1,
        type=int,
        dest='min_order',
        help="Maximum order of neighbors"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    cellcomplex = base.CellComplex.from_tess_file(args.file)
    cellcomplex.find_neighbors_of_order(args.order)
    asum = 0
    for i in range(args.min_order, args.order + 1):
        print(f'Calculating disangles of order {i}...')
        angles = cellcomplex.get_neighbor_dis_angles(i)
        print(f'Finished: {len(angles)} disangles of order {i}')
        filename = os.path.join(args.dir, f'disangles{i}.txt')
        asum += len(angles)
        np.savetxt(filename, angles, fmt='%.2f')
    
    print('Total disangles calculated:', asum)
    print('Total time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()
"""
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import math
from tqdm import tqdm
import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO
)



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
        help="Path to folder which will contain ouput files"
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
        help="Minimum order of neighbors"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    # Loading Cell Complex
    cellcomplex = base.CellComplex.from_tess_file(args.file)
    cellcomplex.find_neighbors_of_order(args.order)
    for g in cellcomplex._grains.values():
        g.rot_mtx = g.R
    logging.info('Cell Complex loaded')

    # Grain orientations in quaternions
    quat = []
    for g in cellcomplex._grains.values():
        A = g.rot_mtx
        q0 = math.sqrt(1 + A[0, 0] + A[1, 1] + A[2, 2]) / 2
        q1 = (A[2, 1] - A[1, 2]) / (4 * q0)
        q2 = (A[0, 2] - A[2, 0]) / (4 * q0)
        q3 = (A[1, 0] - A[0, 1]) / (4 * q0)
        quat.append((q0, q1, q2, q3))
    filename = os.path.join(args.dir, f'quaternions.txt')
    np.savetxt(filename, quat, fmt='%.5f')
    logging.info('Crystal orientaions saved')

    asum = 0
    for i in range(args.min_order, args.order + 1):
        logging.info(f'Start: calculation of order {i}')
        quaternions = []
        thetas = []
        for g1 in tqdm(cellcomplex._grains.values()):
            n_ids = getattr(g1, f'n{i}_ids')
            for n_id in n_ids:
                if g1.id < n_id:
                    q_tuple, theta = matutils.calculate_disorient_quatern(
                        g1.rot_mtx,
                        cellcomplex._grains[n_id].rot_mtx,
                        g1.crysym,
                        angle=True)
                    quaternions.append(q_tuple)
                    thetas.append(theta)
        filename = os.path.join(args.dir, f'quaternions{i}.txt')
        np.savetxt(filename, quaternions, fmt='%.5f')
        filename = os.path.join(args.dir, f'disangles{i}.txt')
        np.savetxt(filename, thetas, fmt='%.3f')
        logging.info(
            f'Finished: {len(quaternions)} disorientations of order {i}'
        )
        
        asum += len(quaternions)
    
    logging.info(f'Total disorientations calculated: {asum}')
    logging.info(f'Total time elapsed: {round(time.time() - start, 2)} s')


if __name__ == "__main__":
    main()
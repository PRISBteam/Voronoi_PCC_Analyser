"""
"""
import os
import argparse
from math import log2
from re import S

from matgen.core import CellComplex

def choose_boundaries(faces, p):
    """
    """
    pass


def get_entropy(c: CellComplex):
    """
    добавить проверку != 0
    """
    S = 0
    for jtype in range(4):
        j = c.get_j_fraction(jtype)
        S -= j * log2(j)
    
    return S

def get_m_entropy(c: CellComplex):
    """
    добавить проверку != 0
    """
    Sm = 1
    for jtype in range(4):
        j = c.get_j_fraction(jtype)
        Sm *= j
    Sm = log2(Sm) / 4

    return Sm

def get_s_entropy(c: CellComplex):
    """
    добавить проверку != 0
    """
    Ss = 0
    for k in range(4):
        for l in range(k, 4):
            jk = c.get_j_fraction(k)
            jl = c.get_j_fraction(l)
            Ss += (jk - jl) * log2(jk / jl)
    Ss = Ss / 4

    return Ss


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
    # extract_seeds(args.filename + '.tess', '.')
    # write_matrices(args.filename + '.tess', '.', True)

    # -periodicity all - check
    # -morphooptiini "coo:file(seeds)"

if __name__ == '__main__':
    main()
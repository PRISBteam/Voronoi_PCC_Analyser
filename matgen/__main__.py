import os
import argparse

from matgen.sparsemat import extract_seeds
from matgen.core import CellComplex


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    default='complex.tess',
    help="Filepath to .tess file with a tesselation (Neper output)"
)
parser.add_argument(
    "--dir",
    default='.',
    help="Path to folder which will contain generated matrices"
)
parser.add_argument(
    "-o", dest='is_signed',
    action='store_true',
    help="Orientation"
)

args = parser.parse_args()

if not os.path.exists(args.dir):
    os.mkdir(args.dir)

extract_seeds(args.file, args.dir)
# write_matrices(args.file, args.dir, args.is_signed) - wrong
c = CellComplex(filename=args.file)
c.save_into_files(work_dir=args.dir)


"""
"""

import argparse
import os
import time
import glob

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

    for filename in filenames:
        pass


    print('Time elapsed:', round(time.time() - start, 2), 's')


if __name__ == "__main__":
    main()

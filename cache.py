import os
import sys

import colorama

from data.ucr import cache_UCR


def main():
    colorama.init(autoreset=True)
    ucr_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', 'UCRArchive_2018'))
    num_workers = None
    l = len(sys.argv)
    if l >= 2:
        ucr_path = sys.argv[1]
    if l >= 3 and sys.argv[2] != 'None':
        num_workers = int(sys.argv[2])
    
    cache_UCR(ucr_path, num_workers)
    
    
if __name__ == '__main__':
    main()

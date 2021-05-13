import os
import sys

import colorama

from data.ucr import cache_UCR


def main():
    colorama.init(autoreset=True)
    if len(sys.argv) == 1:
        ucr_path = 'C:\\Users\\16333\\Desktop\\PyCharm\\dda\\UCRArchive_2018'
        fold_idx = 1
        num_workers = None
    else:
        ucr_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', 'UCRArchive_2018'))
        num_workers = None
        fold_idx = int(sys.argv[1])
        l = len(sys.argv)
        if l >= 3:
            ucr_path = sys.argv[2]
        if l >= 4 and sys.argv[3] != 'None':
            num_workers = int(sys.argv[3])
    
    cache_UCR(ucr_path, fold_idx, num_workers)
    
    
if __name__ == '__main__':
    main()

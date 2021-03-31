import os

import numpy as np
import pandas as pd


def UCRfy_EMI_data(data_path: str, train: bool, train_ratio: float):
    num_classes = 5
    whole_set = pd.read_csv(data_path, sep='\s+', header=None).values.astype(dtype=np.float32)
    data = {i: [] for i in range(num_classes)}
    [data[l[-1] - 1].append(l) for l in whole_set]
    
    picked = []
    for i in range(num_classes):
        train_n = round(train_ratio * len(data[i]))
        test_n = len(data[i]) - train_n
        picked.extend(data[i][:train_n] if train else data[i][-test_n:])
    picked = np.vstack(picked)
    
    final = np.concatenate((picked[:, -1:], picked[:, :-1]), axis=1)
    pfx = 'TRAIN' if train else 'TEST'
    ds_dir = f'C:\\Users\\16333\\Desktop\\PyCharm\\dda\\UCRArchive_2018\\_EMI_ratio{train_ratio:g}'
    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)
    fname = f'{ds_dir}\\EMI_ratio{train_ratio:g}_{pfx}.tsv'
    if not os.path.exists(fname):
        np.savetxt(fname, final, fmt='%.15g', delimiter='\t')


if __name__ == '__main__':
    fname = r'C:\Users\16333\Desktop\PyCharm\dda\data_256_500.txt'
    for ratio in [0.2, 0.8, 0.5]:
        UCRfy_EMI_data(data_path=fname, train=True, train_ratio=ratio)
        UCRfy_EMI_data(data_path=fname, train=False, train_ratio=ratio)

import os

import numpy as np
import torch


def UCRfy_ISOLET_data(root_dir: str, train: bool):
    i_path = root_dir
    i_path = os.path.join(i_path, 'isolet5.data' if train else 'isolet1+2+3+4.data')
    raw = np.loadtxt(i_path, delimiter=',', dtype=np.float32)
    
    final = np.concatenate((raw[:, -1:].astype(np.int)-1, raw[:, :-1]), axis=1)
    pfx = 'TRAIN' if train else 'TEST'
    o_path = f'C:\\Users\\16333\\Desktop\\PyCharm\\dda\\UCRArchive_2018\\_ISOLET'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    fname = f'{o_path}\\ISOLET_{pfx}.tsv'
    if not os.path.exists(fname):
        print(f'saved at {fname}, shape={final.shape}, num_cls={final[:, 0].max() + 1}')
        np.savetxt(fname, final, fmt='%.15g', delimiter='\t')


def get_isolet_dataset(batch_size=128):
    isolet_train = np.loadtxt("C:\\Users\\16333\\Desktop\\PyCharm\\dda\\_ISOLET\\uci-isolet\\original\\isolet5.data", delimiter=',', dtype=np.float32)
    x_data_train = torch.from_numpy(isolet_train[:, :-1]).float()
    y_data_train = torch.from_numpy(isolet_train[:, -1] - 1).long()
    train_dataset = torch.utils.data.TensorDataset(x_data_train, y_data_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    isolet_test = np.loadtxt("C:\\Users\\16333\\Desktop\\PyCharm\\dda\\_ISOLET\\uci-isolet\\original\\isolet1+2+3+4.data", delimiter=',', dtype=np.float32)
    x_data_test = torch.from_numpy(isolet_test[:, :-1]).float()
    y_data_test = torch.from_numpy(isolet_test[:, -1] - 1).long()
    train_dataset = torch.utils.data.TensorDataset(x_data_test, y_data_test)
    test_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    for tt in [False, True]:
        UCRfy_ISOLET_data('C:\\Users\\16333\\Desktop\\PyCharm\\dda\\_ISOLET\\uci-isolet\\original', tt)


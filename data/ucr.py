import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from colorama import Fore

from data.aug import emd
from utils.misc import time_str


def __read_data(root_path, dname, normalize=True, eemd=True, eemd_name='EEMD', num_workers=None):
    tr_path = os.path.join(root_path, dname, f'{dname}_TRAIN.tsv')
    te_path = os.path.join(root_path, dname, f'{dname}_TEST.tsv')

    def get_dl(path):
        items = np.loadtxt(path, delimiter='\t', dtype=np.float32) # [:8]  # todo: rm[:4]
        items = items[np.lexsort(items[:, ::-1].T)]
        data, labels = items[:, 1:], items[:, 0].astype(np.long)
        if labels.min().item() == 1:
            labels -= 1
        return data, labels

    tr_data, tr_labels = get_dl(tr_path)
    te_data, te_labels = get_dl(te_path)
    num_classes = max(tr_labels.max().item(), te_labels.max().item()) + 1
    
    if normalize:
        m, s = tr_data.mean(), tr_data.std(axis=1).mean()
        tr_data = (tr_data - m) / s
        te_data = (te_data - m) / s
    
    start_t = time.time()
    avg_ratio = 0
    tr_emd = []
    bar = tqdm(tr_data)
    for d in bar:
        n, o = emd(d, eemd=eemd, num_workers=num_workers)
        ratio = 100 * np.abs(np.where(n > 0.1, n, np.zeros_like(n))).mean() / np.abs(d).mean()
        avg_ratio += ratio
        tr_emd.append(np.stack((n, o)))
        # bar.set_description(f'[{dname}]: emd, large_ratio={100 * (n > 0.2).sum() / n.shape[0]:.2f}%')
        bar.set_description(f'[{dname}.{eemd_name}], ratio={ratio:.2f}%')
    avg_ratio /= tr_data.shape[0]
    tr_emd = np.stack(tr_emd)
    print(f'{time_str()}[{dname}.{eemd_name}] cost={time.time() - start_t:.2f}s, avg_ratio={avg_ratio:.2f}%')   # 4 worker: 23s; no parallel: 12s ?
    
    t = torch.from_numpy
    return {
        'train_data': t(tr_data).float(),
        'train_emd': t(tr_emd).float(),
        'train_label': t(tr_labels).long(),
        'test_data': t(te_data),
        'test_label': t(te_labels),
        'num_classes': num_classes,
    }, avg_ratio


def cache_UCR(root_path: str, num_workers=None):
    cached_root = os.path.join(os.path.split(root_path)[0], 'UCRTensorData')
    emd_names = ['EEMD', 'CEEM']
    for name in emd_names:
        if not os.path.exists(f'{cached_root}_{name}'):
            os.makedirs(f'{cached_root}_{name}')
    
    ratios = defaultdict(int)
    for it, dname in enumerate(os.listdir(root_path)):
        # if len(dname) < 6:
        #     continue
        for eemd, name in zip([True, False], emd_names):
            cache_fname = os.path.join(f'{cached_root}_{name}', f'{dname}.pth')
            if os.path.exists(cache_fname):
                print(f'{time_str()}[{dname}.{name}] already cached')
            else:
                desc, ratio = __read_data(root_path, dname, normalize=True, eemd=eemd, eemd_name=name, num_workers=num_workers)
                ratios[name] += ratio
                torch.save(desc, cache_fname)
                print(f'{time_str()}[{dname}.{name}] '
                      f'train_data.shape={tuple(desc["train_data"].shape)}, '
                      f'train_emd_data.shape={tuple(desc["train_emd"].shape)}, '
                      f'test_data.shape={tuple(desc["test_data"].shape)}, '
                      f'num_classes={desc["num_classes"]}, '
                      f'cached @ {cache_fname}')
        avg_ratios = {name: f'{ratios[name] / (it+1):5.2f}%' for name in emd_names}
        print(Fore.LIGHTCYAN_EX + f'{time_str()}[CACHE] num_workers={num_workers}, ratios={avg_ratios}')
        

class UCRTensorDataSet(TensorDataset):
    def __init__(self, tensor_data_root_path: str, set_name: str, train: bool, emd: bool):
        desc = torch.load(os.path.join(tensor_data_root_path, f'{set_name}.pth'), map_location='cpu')
        if train:
            time_series_k = 'train_emd' if emd else 'train_data'
            labels_k = 'train_label'
        else:
            time_series_k, labels_k = 'test_data', 'test_label'
        self.num_classes = desc['num_classes']
        super().__init__(desc[time_series_k], desc[labels_k])


if __name__ == '__main__':
    tr_emd = UCRTensorDataSet(r'C:\Users\16333\Desktop\PyCharm\dda\UCRTensorData', 'Adiac', train=True, emd=True)
    tr = UCRTensorDataSet(r'C:\Users\16333\Desktop\PyCharm\dda\UCRTensorData', 'Adiac', train=True, emd=False)

# class UCRDataset():
#     def __init__(self, data_path: str, train_ratio: float, normalize: bool, num_of_dataset=127, data_name_list=[]):
#         self.dict = {}
#         cur_num_of_dataset = 0
#         for dname in os.listdir(data_path):
#             if os.path.isdir(os.path.join(data_path, dname)) and (len(data_name_list) == 0 or dname in data_name_list):
#                 train_and_valid_data = self.__sortByCategory(
#                     np.loadtxt(os.path.join(data_path, dname, dname + '_TRAIN.tsv'), delimiter='\t', dtype=np.float32))
#                 test_data = self.__sortByCategory(
#                     np.loadtxt(os.path.join(data_path, dname, dname + '_TEST.tsv'), delimiter='\t', dtype=np.float32))
#
#                 train_and_valid_signals = torch.from_numpy(train_and_valid_data[:, 1:]).float()
#                 train_and_valid_labels = torch.from_numpy(train_and_valid_data[:, 0]).long()
#
#                 test_signals = torch.from_numpy(test_data[:, 1:]).float()
#                 test_labels = torch.from_numpy(test_data[:, 0]).long()
#
#                 if train_and_valid_data[0][0] == 1:
#                     train_and_valid_labels = train_and_valid_labels - 1
#                     test_labels = test_labels - 1
#
#                 if normalize:
#                     train_and_valid_signals = (train_and_valid_signals - torch.mean(
#                         train_and_valid_signals)) / torch.std(train_and_valid_signals)
#                     test_signals = (test_signals - torch.mean(test_signals)) / torch.std(test_signals)
#
#                 # print(train_and_valid_signals)
#                 num_classes = train_and_valid_labels[-1].item() - train_and_valid_labels[0].item() + 1
#                 cur_category = 0
#                 former_pos = 0
#                 for cur_pos in range(len(train_and_valid_signals)):
#                     if train_and_valid_labels[cur_pos].item() > cur_category:
#                         num = cur_pos - former_pos
#                         train_num = round(num * train_ratio)
#                         # valid_num = num - train_num
#                         if cur_category == 0:
#                             train_signals = train_and_valid_signals[former_pos: former_pos + train_num]
#                             train_labels = train_and_valid_labels[former_pos: former_pos + train_num]
#
#                             valid_signals = train_and_valid_signals[former_pos + train_num: cur_pos]
#                             valid_labels = train_and_valid_labels[former_pos + train_num: cur_pos]
#                         else:
#                             train_signals = torch.cat(
#                                 (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
#                             train_labels = torch.cat(
#                                 (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)
#
#                             valid_signals = torch.cat(
#                                 (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos]), 0)
#                             valid_labels = torch.cat(
#                                 (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos]), 0)
#                         cur_category = train_and_valid_labels[cur_pos].item()
#                         former_pos = cur_pos
#
#                 num = len(train_and_valid_signals) - former_pos
#                 train_num = round(num * train_ratio)
#                 # valid_num = num - train_num
#                 train_signals = torch.cat(
#                     (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
#                 train_labels = torch.cat(
#                     (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)
#
#                 valid_signals = torch.cat(
#                     (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos + 1]), 0)
#                 valid_labels = torch.cat(
#                     (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos + 1]), 0)
#             # print(train_signals)
#                 self.dict[dname] = {
#                     'num_classes': num_classes,
#                     'train': torch.utils.data.TensorDataset(train_signals, train_labels),
#                     'valid': torch.utils.data.TensorDataset(valid_signals, valid_labels),
#                     'test': torch.utils.data.TensorDataset(test_signals, test_labels),
#                 }
#                 cur_num_of_dataset = cur_num_of_dataset + 1
#                 if cur_num_of_dataset == num_of_dataset:
#                     break
#
#     def getDatasetByName(self, name):
#         return self.dict[name]
#
#     def getNameList(self):
#         return self.dict.keys()
#
#     def getAllDataset(self):
#         return self.dict
#
#     def __sortByCategory(self, data):
#         # print(data)
#         reverse_transpose = data[:, ::-1].T
#         index = np.lexsort(reverse_transpose)
#         # print(data[index])
#         return data[index]


# if __name__ == '__main__':
#     ucrDataset = UCRDataset(
#         data_path='UCRArchive_2018',
#         normalize=True,
#         train_ratio=0.5,
#     )
#     print(ucrDataset.getDatasetByName('DistalPhalanxTW')['num_classes'])
#     print(ucrDataset.getDatasetByName('DistalPhalanxTW')['train'][:])
#     print(ucrDataset.getDatasetByName('DistalPhalanxTW')['valid'][:])
#     print(ucrDataset.getDatasetByName('DistalPhalanxTW')['test'][:])

    # for key, value in ucrDataset.getAllDataset().items():
    #     print(key)
    #     print(value['num_classes'])
    #     print(value['train'][:])
    #     print(value['valid'][:])
    #     print(value['test'][:])

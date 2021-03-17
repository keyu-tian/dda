import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from utils.misc import ints_ceil


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, shuffle=True, drop_last=False, fill_last=False, seed=None):
        assert not (drop_last and fill_last)
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else ints_ceil(dataset_len, batch_size)
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = 1
        self.seed = seed
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        if self.shuffle:
            if self.seed is None:
                indices = torch.randperm(self.dataset_len)
            else:
                g = torch.Generator()
                g.manual_seed(self.seed * 100 + self.epoch)
                indices = torch.randperm(self.dataset_len, generator=g)
        else:
            indices = torch.arange(self.dataset_len)
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            indices = torch.cat((indices, tails), dim=0)
            if self.shuffle:
                if self.seed is None:
                    indices = indices[torch.randperm(indices.shape[0])]
                else:
                    g = torch.Generator()
                    g.manual_seed(self.seed * 1000 + self.epoch)
                    indices = indices[torch.randperm(indices.shape[0], generator=g)]
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.numpy().tolist())
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


if __name__ == '__main__':
    sp = InfiniteBatchSampler(50000, 512, shuffle=True, drop_last=False, fill_last=True, seed=0)
    n = 0
    it = iter(sp)
    for i in range(len(sp)):
        idx = next(it)
        n += len(idx)
        if i == 0:
            print(idx[:5])
    print(len(sp), n)

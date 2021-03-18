import math
import torch
from abc import ABCMeta, abstractmethod


class _WarmupScheduler(metaclass=ABCMeta):
    def __init__(self, op, max_lr: float, max_steps: int):
        self.op, self.max_lr, self.max_steps = op, max_lr, max_steps
        self.warmup_steps = max(round(self.max_steps / 200), 1)
        self.initial_lr = self.max_lr / 10
        self.cur_step = 0
        self.last_lr = -1
    
    @abstractmethod
    def get_lr(self, *args): ...
    
    @abstractmethod
    def state_dict(self): ...
    
    @abstractmethod
    def load_state_dict(self, state_dict): ...
    
    def get_warmup_lr(self):
        # assert self.cur_step <= self.warmup_steps
        ratio = self.cur_step / self.warmup_steps
        return self.initial_lr + ratio * (self.max_lr - self.initial_lr)
    
    def set_lr(self, *args):
        self.last_lr = self.get_warmup_lr() if self.cur_step <= self.warmup_steps else self.get_lr(*args)
        for param_group in self.op.param_groups:
            param_group['lr'] = self.last_lr
    
    def step(self, *args):
        self.set_lr(*args)
        self.cur_step += 1


class ConstantScheduler(_WarmupScheduler):
    def get_lr(self):
        return self.max_lr
    
    def state_dict(self):
        return {
            'cur_step': self.cur_step
        }
    
    def load_state_dict(self, state_dict):
        self.cur_step = state_dict['cur_step']


class CosineScheduler(_WarmupScheduler):
    def get_lr(self):
        ratio = (self.cur_step - self.warmup_steps) / (self.max_steps - 1 - self.warmup_steps)
        return self.max_lr * 0.5 * (1. + math.cos(math.pi * ratio))

    def state_dict(self):
        return {
            'cur_step': self.cur_step
        }

    def load_state_dict(self, state_dict):
        self.cur_step = state_dict['cur_step']


class ReduceOnPlateau(_WarmupScheduler):
    def __init__(self, op, max_lr, max_steps, **kwargs):
        super().__init__(op, max_lr, max_steps)
        self.sc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.op, factor=1 / pow(2, 1 / 3),
            **kwargs,
            # mode='min',
            # patience=patience,
            # verbose=True,
            # threshold=0.0001,
            # threshold_mode='rel',
            # cooldown=0,
            # min_lr=0.0001
        )
    
    def get_lr(self, metrics):
        self.sc.step(metrics)
        return self.sc._last_lr[0]

    def state_dict(self):
        return {
            'sc': self.sc.state_dict(),
            'cur_step': self.cur_step
        }

    def load_state_dict(self, state_dict):
        self.sc.load_state_dict(state_dict['sc'])
        self.cur_step = state_dict['cur_step']


if __name__ == '__main__':
    import torch.nn as nn
    from torch.optim import SGD
    op = SGD(nn.Linear(3, 2).parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=True)
    sc = CosineScheduler(op, 1., 10)
    
    for i in range(10):
        sc.step()
        print(f'after {i}: {sc.last_lr:.3f}')
        


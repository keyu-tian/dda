import datetime
import logging
import os
import random
import datetime
import shutil

import numpy as np
import pytz
import torch


def time_str():
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('[%m-%d %H:%M:%S]')


def ints_ceil(x: int, y: int) -> int:
    return (x + y - 1) // y  # or (x - 1) // y + 1


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('==> Dir created : {}.'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def create_logger(name, log_file, level=logging.INFO, stream=True):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)6s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self, length=0):
        self.length = int(length)
        self.history = []
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
    
    def reset(self):
        if self.length > 0:
            self.history.clear()
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
    
    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            # assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            
            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
    
    def get_trimmed_mean(self):
        if len(self.history) >= 5:
            trimmed = max(int(self.length * 0.1), 1)
            return np.mean(sorted(self.history)[trimmed:-trimmed])
        else:
            return self.avg
    
    def state_dict(self):
        return vars(self)
    
    def load_state(self, state_dict):
        self.__dict__.update(state_dict)


def init_params(model: torch.nn.Module, output=None):
    if output is not None:
        output('===================== param initialization =====================')
    tot_num_inited = 0
    for i, m in enumerate(model.modules()):
        clz = m.__class__.__name__
        is_conv = clz.find('Conv') != -1
        is_bn = clz.find('BatchNorm') != -1
        is_fc = clz.find('Linear') != -1
        
        cur_num_inited = []
        if is_conv:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_bn:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_fc:
            # torch.nn.init.normal_(m.weight, std=0.001)
            torch.nn.init.normal_(m.weight, std=1 / m.weight.size(-1))
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        tot_num_inited += sum(cur_num_inited)
        
        if output is not None:
            builtin = any((is_conv, is_bn, is_fc))
            cur_num_inited = f' ({" + ".join([str(x) for x in cur_num_inited])})'
            output(f'clz{i:3d}: {"  => " if builtin else ""}{clz}{cur_num_inited if builtin else "*"}')
    
    if output is not None:
        output('----------------------------------------------------------------')
        output(f'tot_num_inited: {tot_num_inited} ({tot_num_inited / 1e6:.3f} M)')
        output('===================== param initialization =====================\n')
    return tot_num_inited

import argparse
import datetime
import os
import time
from copy import deepcopy
from pprint import pformat

import colorama
import torch
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from data.sampler import InfiniteBatchSampler
from data.ucr import UCRTensorDataSet, ALL_NAMES
from model import model_entry
from model.augmenter import Augmenter
from utils.scheduler import CosineLRScheduler, ConstScheduler
from utils.misc import create_exp_dir, create_logger, set_seed, init_params, AverageMeter

using_gpu = torch.cuda.is_available()

try:
    import linklink as link
    link_dist = using_gpu
except ImportError:
    link = None
    link_dist = False


def prepare():
    colorama.init(autoreset=True)
    
    parser = argparse.ArgumentParser(description='Dynamic Data Augmentation for Time-Series')
    parser.add_argument('--cfg_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--seed', default='0', type=str)
    parser.add_argument('--input_size', required=True, type=int)
    parser.add_argument('--num_classes', required=True, type=int)
    parser.add_argument('--only_val', action='store_true', default=False)
    
    # Distribution
    if link_dist:
        torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
        link.initialize()
        world_size, rank = link.get_world_size(), link.get_rank()
    else:
        num_gpus, world_size, rank = 1, 1, 0
    
    # Parsing args
    args = parser.parse_args()
    args.save_dir = os.path.join(args.log_path, f'ckpts')
    if rank == 0:
        print(f'==> raw args:\n{pformat(vars(args))}')
    
    with open(args.cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) if hasattr(yaml, 'FullLoader') else yaml.load(f)
        cfg = EasyDict(cfg)
    
    if rank == 0:
        print(f'==> raw config:\n{pformat(cfg)}')
    
    cfg.model.kwargs.input_size = args.input_size
    cfg.model.kwargs.num_classes = args.num_classes
    
    cfg.optm.name = cfg.optm.name.strip().lower()
    cfg.optm.kwargs.weight_decay = float(cfg.optm.kwargs.weight_decay)
    
    cfg.sche.name = cfg.sche.name.strip().lower()
    cfg.sche.kwargs.lr = float(cfg.sche.kwargs.lr)
    if 'min_lr' in cfg.sche.kwargs:
        cfg.sche.kwargs.min_lr = float(cfg.sche.kwargs.min_lr)
    cfg.sche.kwargs.base_lr = cfg.sche.kwargs.lr / cfg.sche.kwargs.base_lr_division
    cfg.sche.kwargs.pop('base_lr_division')
    if 'warm_up_division' not in cfg.sche.kwargs:
        cfg.sche.kwargs.warm_up_division = 100
    
    if rank == 0:
        print('==> Creating dirs ...')
        create_exp_dir(args.log_path, scripts_to_save=None)
        create_exp_dir(args.save_dir, scripts_to_save=None)
        print('==> Creating dirs complete.\n')
        if link_dist:
            link.barrier()
    
    # Logger
    if rank == 0:
        print('==> Creating logger ...')
        lg = create_logger('global', os.path.join(args.log_path, 'log.txt'))
        print('==> Creating logger complete.\n')
        lg.info(f'==> args:\n{pformat(vars(args))}\n')
        lg.info(f'==> cfg:\n{pformat(cfg)}\n')
        tb_lg = SummaryWriter(os.path.join(args.log_path, 'events'))
        tb_lg.add_text('exp_time', time.strftime("%Y%m%d-%H%M%S"))
        tb_lg.add_text('exp_dir', f'~/{os.path.relpath(os.getcwd(), os.path.expanduser("~"))}')
        if link_dist:
            link.barrier()
    else:
        lg = None
        tb_lg = None
    
    if not using_gpu and rank == 0:
        lg.info('==> No available GPU device!\n')
    
    # Seed
    if args.seed is None:
        args.seed = 0
    elif args.seed.strip().lower() in ['rand', 'random', 'rnd']:
        args.seed = round(time.time() * 1e9) % round(1e9+7)
    else:
        args.seed = round(eval(args.seed))
    args.seed += rank
    
    if rank == 0:
        lg.info('==> Preparing data..')

    ds_clz: Dataset.__class__
    if cfg.data.name in ALL_NAMES:
        ds_clz = UCRTensorDataSet
    else:
        ds_clz = {
            ''
        }[cfg.data.name]

    train_set = ds_clz(train=True, emd=False, **cfg.data.kwargs)
    emd_set = ds_clz(train=True, emd=True, **cfg.data.kwargs)
    test_set = ds_clz(train=False, emd=False, **cfg.data.kwargs)

    cfg.model.kwargs.num_classes = train_set.num_classes
    
    if rank == 0:
        lg.info(f'==> Building dataloaders of {cfg.data.name} ...')

    train_loader = DataLoader(
        dataset=train_set, num_workers=4, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_set), batch_size=cfg.data.batch_size, shuffle=True, drop_last=False, fill_last=True, seed=0,
        )
    )
    emd_loader = DataLoader(
        dataset=emd_set, num_workers=4, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_set), batch_size=cfg.data.batch_size, shuffle=True, drop_last=False, fill_last=True, seed=0,
        )
    )
    test_loader = DataLoader(
        dataset=test_set, num_workers=4, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(test_set), batch_size=cfg.data.batch_size * (1 if cfg.model.name == 'LSTM' else 2), shuffle=False, drop_last=False,
        )
    )
    
    # Load checkpoints.
    loaded_ckpt = None
    if args.ckpt_path is not None:
        ckpt_path = os.path.abspath(args.ckpt_path)
        if rank == 0:
            lg.info(f'==> Getting ckpt for resuming at {ckpt_path} ...')
        assert os.path.isfile(ckpt_path), '==> Error: no checkpoint file found!'
        loaded_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if rank == 0:
            lg.info(f'==> Getting ckpt for resuming complete.\n')
    
    return args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader


def test(net, tot_it, test_iterator):
    net.eval()
    with torch.no_grad():
        tot_loss, tot_pred, tot_correct = 0., 0, 0
        for it in range(tot_it):
            inp, tar = next(test_iterator)
            if using_gpu:
                inp, tar = inp.cuda(), tar.cuda()
            outputs = net(inp)
            loss = F.cross_entropy(outputs, tar)
            
            tot_loss += loss.item()
            _, predicted = outputs.max(1)
            tot_pred += tar.size(0)
            tot_correct += predicted.eq(tar).sum().item()
    
    return tot_loss / tot_it, 100. * tot_correct / tot_pred


def build_model(cfg, lg, rank, loaded_ckpt):
    if rank == 0:
        lg.info('==> Building model..')
    if cfg.model.name == 'LSTM':
        cfg.model.kwargs.batch_size = cfg.batch_size
    net = model_entry(cfg.model)
    init_params(net, output=None if rank != 0 else lg.info)
    
    if loaded_ckpt is not None:
        net.load_state_dict(loaded_ckpt['model'])
    num_para = sum(p.numel() for p in net.parameters()) / 1e6
    if rank == 0:
        lg.info(f'==> Building model complete, type: {type(net)}, param: {num_para:.3f} * 10^6.\n')
    return net.cuda() if using_gpu else net


def build_op(op_cfg, sc_cfg, loaded_ckpt, net):
    op_cfg.kwargs.params = net.parameters()
    op_cfg.kwargs.lr = sc_cfg.kwargs.base_lr
    op = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }[op_cfg.name](**op_cfg.kwargs)
    op_cfg.kwargs.pop('params')
    if loaded_ckpt is not None:
        op.load_state_dict(loaded_ckpt['op'])
    return op


def build_sche(cfg, optm, start_iter, max_iter):
    if cfg.sche.name == 'cos':
        return CosineLRScheduler(
            optimizer=optm,
            T_max=max_iter,
            eta_min=cfg.sche.kwargs.min_lr,
            base_lr=cfg.sche.kwargs.base_lr,
            warmup_lr=cfg.sche.kwargs.lr,
            warmup_steps=max(round(max_iter / cfg.sche.kwargs.warm_up_division), 1),
            last_iter=start_iter-1
        )
    elif cfg.sche.name == 'con':
        return ConstScheduler(
            lr=cfg.optm.lr
        )
    else:
        raise AttributeError(f'unknown scheduler type: {cfg.sche.name}')


def train_from_scratch(args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader):
    # Initialize.
    # todo: set_seed
    
    model: torch.nn.Module = build_model(cfg, lg, rank, loaded_ckpt)
    auger = Augmenter(model.feature_dim)
    
    model_op = build_op(cfg.model_op, cfg.model_sc, loaded_ckpt, model)
    auger_op = build_op(cfg.auger_op, cfg.auger_sc, loaded_ckpt, auger)
    
    if loaded_ckpt is not None:
        start_iter = loaded_ckpt['start_iter']
    else:
        start_iter = 0
    tot_it = len(train_loader)
    tot_test_it = len(test_loader)
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)
    global_tot_it = tot_it * cfg.epochs
    sche = build_sche(cfg, model_op, start_iter=start_iter, max_iter=global_tot_it)
    
    lg.info(f'==> final args:\n{pformat(vars(args))}\n')
    lg.info(f'==> final cfg:\n{pformat(cfg)}\n')
    best_acc = 0
    tb_lg_freq = max(round(tot_it / 10), 1)
    val_freqs = [tot_it * 3, round(tot_it / 4)]
    
    train_loss_avg = AverageMeter(round(tot_it / tb_lg_freq))
    train_acc_avg = AverageMeter(round(tot_it / tb_lg_freq))
    speed_avg = AverageMeter(0)
    start_train_t = time.time()
    for epoch in range(cfg.epochs):
        is_late = int(epoch > 0.75 * cfg.epochs)
        
        # train a epoch
        last_t = time.time()
        for it in range(tot_it):
            inp, tar = next(train_iterator)
            global_it = tot_it * epoch + it
            data_t = time.time()
            sche.step()
            if using_gpu:
                inp, tar = inp.cuda(), tar.cuda()
            cuda_t = time.time()
            
            inp = aug(inp)
            aug_t = time.time()
            
            model_op.zero_grad()
            outputs = model(inp)
            # outputs = outputs[0]
            loss = F.cross_entropy(outputs, tar)
            loss.backward()
            loss_val = loss.item()
            train_loss_avg.update(loss_val)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            model_op.step()
            train_t = time.time()
            
            predicted = outputs.argmax(dim=1)
            pred, correct = tar.shape[0], predicted.eq(tar).sum().item() # tar.size(0) i.e. batch_size(or tail batch size)
            train_acc_avg.update(val=100 * correct / pred, num=pred / cfg.batch_size)
            
            if (global_it % tb_lg_freq == 0 or global_it == global_tot_it - 1) and rank == 0:
                tb_lg.add_scalar('train_loss', train_loss_avg.avg, global_it)
                tb_lg.add_scalar('train_acc', train_acc_avg.avg, global_it)
                tb_lg.add_scalar('lr', sche.get_lr()[0], global_it)
            
            if global_it % val_freqs[is_late] == 0 or global_it == global_tot_it - 1:
                test_loss, test_acc = test(model, tot_test_it, test_iterator)
                test_t = time.time()
                test_err = 100 - test_acc
                model.train()
                if rank == 0:
                    remain_secs = (global_tot_it - global_it - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    ep_str = f'%{len(str(cfg.epochs))}d' % (epoch+1)
                    it_str = f'%{len(str(tot_it))}d' % (it+1)
                    lg.info(
                        f'ep[{ep_str}/{cfg.epochs}], it[{it_str}/{tot_it}]:'
                        f' t-err[{100-train_acc_avg.val:5.2f}] ({100-train_acc_avg.avg:5.2f}),'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' v-err[{test_err:5.2f}],'
                        f' v-loss[{test_loss:.4f}],'
                        f' data[{data_t-last_t:.3f}],'
                        f' cuda[{cuda_t-data_t:.3f}],'
                        f' au[{aug_t-cuda_t:.3f}],'
                        f' bp[{train_t-aug_t:.3f}],'
                        f' te[{test_t-train_t:.3f}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                        f' lr[{sche.get_lr()[0]:.3g}]'
                    )
                    tb_lg.add_scalar('test_loss', test_loss, global_it)
                    tb_lg.add_scalar('test_acc', test_acc, global_it)
                    tb_lg.add_scalar('test_err', test_err, global_it)
                
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
                if rank == 0 and is_best:
                    model_ckpt_path = os.path.join(args.save_dir, f'best.pth.tar')
                    lg.info(f'==> Saving best model ckpt (epoch[{epoch}], err{100-test_acc:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                    torch.save({
                        'model': model.state_dict(),
                        'op': model_op.state_dict(),
                        'start_iter': global_it,
                    }, model_ckpt_path)
            
            speed_avg.update(time.time() - last_t)
            last_t = time.time()
    
    if rank == 0:
        if link_dist:
            best_accs = torch.zeros(world_size)
            best_accs[rank] = best_acc
            link.allreduce(best_accs)
            best_errs = 100 - best_accs
            lg.info(
                f'==> End training,'
                f' total time cost: {time.time()-start_train_t:.3f},'
                f' best test err @1: {best_errs.mean().item():.3f} ({best_errs.min().item():.3f})'
            )
        else:
            lg.info(
                f'==> End training,'
                f' total time cost: {time.time()-start_train_t:.3f},'
                f' best test err @1: {100-best_acc:.3f}'
            )
    
    tb_lg.close()
    
    if link_dist:
        link.finalize()


def main():
    args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader = prepare()
    if args.only_val:
        pass
    else:
        train_from_scratch(args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader)


if __name__ == '__main__':
    main()

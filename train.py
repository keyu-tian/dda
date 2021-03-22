import argparse
import datetime
import os
import time
from pprint import pformat

import colorama
import pytz
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from data.aug import augment_and_aggregate_batch
from data.sampler import InfiniteBatchSampler
from data.ucr import UCRTensorDataSet, ALL_NAMES
from model import model_entry
from model.augmenter import Augmenter
from utils.misc import create_exp_dir, create_logger, init_params, AverageMeter, inverse_grad, TopKHeap, time_str, flatten_grads
from utils.scheduler import ConstantScheduler, CosineScheduler, ReduceOnPlateau
from utils.seatable import SeatableLogger

try:
    import linklink as link
    
    link_dist = True
except ImportError:
    link = None
    link_dist = False


def prepare():
    colorama.init(autoreset=True)
    
    parser = argparse.ArgumentParser(description='Dynamic Data Augmentation for Time-Series')
    parser.add_argument('--sh_name', type=str, required=True)
    parser.add_argument('--cfg_name', type=str, required=True)
    parser.add_argument('--exp_dir_name', type=str, required=True)
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
    args.exp_path = os.path.join(os.getcwd(), args.exp_dir_name)
    args.job_name = os.path.split(os.getcwd())[-1]
    args.event_path = os.path.join(args.exp_path, f'events')
    args.save_path = os.path.join(args.exp_path, f'ckpts')
    if rank == 0:
        print(f'==> raw args:\n{pformat(vars(args))}')
    
    with open(args.cfg_name) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) if hasattr(yaml, 'FullLoader') else yaml.load(f)
        cfg = EasyDict(cfg)
    
    if rank == 0:
        print(f'==> raw config:\n{pformat(cfg)}')
    
    if rank == 0:
        print('==> Creating dirs ...')
        create_exp_dir(args.exp_path, scripts_to_save=None)
        create_exp_dir(args.event_path, scripts_to_save=None)
        create_exp_dir(args.save_path, scripts_to_save=None)
        print('==> Creating dirs complete.\n')
        if link_dist:
            link.barrier()
    
    # Logger
    if rank == 0:
        print('==> Creating loggers ...')
        lg = create_logger('global', os.path.join(args.exp_path, 'log.txt'))
        lg.info(f'==> args:\n{pformat(vars(args))}\n')
        lg.info(f'==> cfg:\n{pformat(cfg)}\n')
        tb_lg = SummaryWriter(args.event_path)
        tb_lg.add_text('exp_time', time_str())
        tb_lg.add_text('exp_path', args.exp_path)
        print('==> Creating loggers complete.\n')
        if link_dist:
            link.barrier()
    else:
        lg = None
        tb_lg = None
    
    if not torch.cuda.is_available():
        raise AttributeError('==> No available GPU device!')
    
    # Seed
    if cfg.seed == 'None':
        cfg.seed = None
    
    if cfg.seed is not None:
        if cfg.seed.strip().lower() in ['rand', 'random', 'rnd']:
            cfg.seed = round(time.time() * 1e9) % round(1e9 + 7)
        else:
            cfg.seed = round(eval(cfg.seed))
        cfg.seed += rank
    
    if rank == 0:
        lg.info('==> Preparing data..')
    
    ds_clz: Dataset.__class__
    if cfg.data.name in ALL_NAMES:
        ds_clz = UCRTensorDataSet
    else:
        raise NotImplementedError
        ds_clz = {
            ''
        }[cfg.data.name]
    
    if cfg.data.kwargs is None:
        cfg.data.kwargs = {}
    train_set = ds_clz(train=True, emd=False, **cfg.data.kwargs)
    emd_set = ds_clz(train=True, emd=True, **cfg.data.kwargs)
    test_set = ds_clz(train=False, emd=False, **cfg.data.kwargs)
    
    cfg.model.kwargs.input_size = train_set.input_size
    cfg.model.kwargs.num_classes = train_set.num_classes
    
    if rank == 0:
        lg.info(f'==> Building dataloaders of {cfg.data.name} ...')
    
    train_loader = DataLoader(
        dataset=train_set, num_workers=2, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_set), batch_size=cfg.data.batch_size, shuffle=True, drop_last=False, fill_last=True, seed=0,
        )
    )
    emd_loader = DataLoader(
        dataset=emd_set, num_workers=2, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_set), batch_size=cfg.data.batch_size, shuffle=True, drop_last=False, fill_last=True, seed=0,
        )
    )
    test_loader = DataLoader(
        dataset=test_set, num_workers=2, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(test_set), batch_size=cfg.data.batch_size * (1 if cfg.model.name == 'LSTM' else 2), shuffle=False, drop_last=False,
        )
    )
    
    # Load checkpoints.
    loaded_ckpt = None
    if cfg.ckpt_path is not None:
        ckpt_path = os.path.abspath(cfg.ckpt_path)
        if rank == 0:
            lg.info(f'==> Getting ckpt for resuming at {ckpt_path} ...')
        assert os.path.isfile(ckpt_path), '==> Error: no checkpoint file found!'
        loaded_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if rank == 0:
            lg.info(f'==> Getting ckpt for resuming complete.\n')
    
    return args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader


def test(model, tot_it, test_iterator):
    model.eval()
    with torch.no_grad():
        tot_loss, tot_pred, tot_correct = 0., 0, 0
        for it in range(tot_it):
            inp, tar = next(test_iterator)
            inp, tar = inp.cuda(), tar.cuda()
            logits = model(inp)
            loss = F.cross_entropy(logits, tar)
            
            tot_pred += tar.shape[0]
            tot_loss += loss.item() * tar.shape[0]
            tot_correct += logits.argmax(dim=1).eq(tar).sum().item()
    model.train()
    return tot_loss / tot_pred, 100. * tot_correct / tot_pred


def build_model_and_auger(cfg, lg, rank, loaded_ckpt):
    if rank == 0:
        lg.info('==> Building model..')
    if cfg.model.kwargs is None:
        cfg.model.kwargs = {}
    if cfg.auger.kwargs is None:
        cfg.auger.kwargs = {}
    
    if cfg.model.name == 'LSTM':
        cfg.model.kwargs.batch_size = cfg.data.batch_size
    model = model_entry(cfg.model)
    if cfg.init_model:
        init_params(model, output=None if rank != 0 else lg.info)
    
    if loaded_ckpt is not None:
        model.load_state_dict(loaded_ckpt['model'])
    if rank == 0:
        lg.info(f'==> Building model complete, type: {type(model)}, params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} * 10^6.\n')
    
    auger = Augmenter(model.feature_dim)
    if loaded_ckpt is not None:
        auger.load_state_dict(loaded_ckpt['auger'])
    if rank == 0:
        lg.info(f'==> Building augmenter complete, type: {type(auger)}, params: {sum(p.numel() for p in auger.parameters()) / 1e6:.3f} * 10^6.\n')
    
    return model.cuda(), auger.cuda()


def build_op_and_sc(op_cfg, sc_cfg, iters_per_epoch, network, loaded_ckpt, load_prefix):
    sc_cfg.name = sc_cfg.name.strip().lower()
    sc_cfg.kwargs.max_lr = float(sc_cfg.kwargs.max_lr)
    sc_cfg.kwargs.max_step = round(sc_cfg.kwargs.epochs * iters_per_epoch)
    sc_cfg.kwargs.pop('epochs')
    
    if 'min_lr' in sc_cfg.kwargs:
        sc_cfg.kwargs.min_lr = float(sc_cfg.kwargs.min_lr)
    if 'threshold' in sc_cfg.kwargs:
        sc_cfg.kwargs.threshold = float(sc_cfg.kwargs.threshold)
    if 'patience_ratio' in sc_cfg.kwargs:
        sc_cfg.kwargs.patience = round(float(sc_cfg.kwargs.patience_ratio) * sc_cfg.kwargs.max_step)
        sc_cfg.kwargs.pop('patience_ratio')
    
    op_cfg.name = op_cfg.name.strip().lower()
    op_cfg.kwargs.weight_decay = float(op_cfg.kwargs.weight_decay)
    op_cfg.kwargs.lr = sc_cfg.kwargs.max_lr / 10
    
    op, op_tag = {
        'sgd': (torch.optim.SGD, 'sgd'),
        'adam': (torch.optim.Adam, 'adm'),
        'adamw': (torch.optim.AdamW, 'amw'),
    }[op_cfg.name]
    op = op(network.parameters(), **op_cfg.kwargs)
    
    sc, sc_tag = {
        'const': (ConstantScheduler, 'con'),
        'cos': (CosineScheduler, 'cos'),
        'plateau': (ReduceOnPlateau, 'pla'),
    }[sc_cfg.name]
    sc = sc(op, **sc_cfg.kwargs)
    
    if loaded_ckpt is not None:
        op.load_state_dict(loaded_ckpt[f'{load_prefix}_op'])
        sc.load_state_dict(loaded_ckpt[f'{load_prefix}_sc'])
    return op, op_tag, sc, sc_tag


def train_from_scratch(args, cfg, lg, tb_lg, world_size, rank, loaded_ckpt, train_loader, emd_loader, test_loader):
    # Initialize.
    # todo: set_seed
    ablation = cfg.get('ablation', '')
    no_aug = ablation == 'no_aug'
    random_aug = ablation == 'random_aug'
    random_feature = ablation == 'random_feature'
    
    sea_lg = SeatableLogger(args.exp_path) if rank == 0 else None
    
    model: torch.nn.Module
    auger: torch.nn.Module
    model, auger = build_model_and_auger(cfg, lg, rank, loaded_ckpt)
    
    model_op, m_op_tag, model_sc, m_sc_tag = build_op_and_sc(cfg.model_op, cfg.model_sc, len(train_loader), model, loaded_ckpt, 'model')
    auger_op, a_op_tag, auger_sc, a_sc_tag = build_op_and_sc(cfg.auger_op, cfg.auger_sc, len(train_loader), auger, loaded_ckpt, 'auger')

    m_op_tag, m_sc_tag = 'M' + m_op_tag, 'M' + m_sc_tag
    a_op_tag, a_sc_tag = 'A' + a_op_tag, 'A' + a_sc_tag
    
    if rank == 0:
        lambda_kw = {chr(955): cfg.penalty_lambda}
        op_sc_kw = {m_op_tag: True, m_sc_tag: True, a_op_tag: True, a_sc_tag: True}
        ablation_kw = {'Naug': no_aug, 'Raug': random_aug, 'Rfea': random_feature}
        print(colorama.Fore.CYAN + f'op_sc_kw=\n {pformat(op_sc_kw)}')
        print(colorama.Fore.CYAN + f'ablation_kw=\n {pformat(ablation_kw)}')
        sea_lg.create_or_upd_row(
            cfg.data.name, vital=True,
            m=cfg.model.name, ep=cfg.epochs, bs=cfg.data.batch_size,
            p=cfg.aug_prob, mlr=f'{cfg.model_sc.kwargs.max_lr:.1g}', alr=f'{cfg.auger_sc.kwargs.max_lr:.1g}',
            pr=0, rem=0, beg_t=datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
            **lambda_kw, **op_sc_kw, **ablation_kw,
        )
    
    if loaded_ckpt is not None:
        start_ep = loaded_ckpt['start_ep'] + 1
    else:
        start_ep = 0
    
    iters_per_train_ep = len(train_loader)
    iters_per_test_ep = len(test_loader)
    emd_iterator = iter(emd_loader)
    test_iterator = iter(test_loader)
    max_iter = iters_per_train_ep * cfg.epochs

    
    
    if rank == 0:
        lg.info(f'==> ablation={ablation}, iters_per_train_ep={iters_per_train_ep}')
        lg.info(f'==> final args:\n{pformat(vars(args))}\n')
        lg.info(f'==> final cfg:\n{pformat(cfg)}\n')
    
    best_acc = 0
    topk_acc1s = TopKHeap(maxsize=max(1, round(cfg.epochs * 0.05)))
    tb_lg_freq = max(round(iters_per_train_ep / 2), 1)
    lg_freq = max(round(cfg.epochs * 0.02), 1)

    epoch_speed = AverageMeter(3)
    train_loss_avg = AverageMeter(iters_per_train_ep)
    penalty_avg = AverageMeter(iters_per_train_ep)
    alpha_A_avg = AverageMeter(iters_per_train_ep)
    alpha_B_avg = AverageMeter(iters_per_train_ep)
    train_acc_avg = AverageMeter(iters_per_train_ep)
    start_train_t = time.time()
    clipping_iters = round(0.05 * iters_per_train_ep * cfg.epochs)
    for epoch in range(start_ep, cfg.epochs):
        epoch_start_t = time.time()
        last_t = time.time()
        for it in range(iters_per_train_ep):
            cur_it = iters_per_train_ep * epoch + it
            noi_inp, oth_inp, tar = next(emd_iterator)
            data_t = time.time()
            noi_inp, oth_inp, tar = noi_inp.cuda(), oth_inp.cuda(), tar.cuda()
            cuda_t = time.time()
            
            org_inp = noi_inp + oth_inp
            if random_feature:
                feature = torch.randn(tar.shape[0], model.feature_dim).cuda()
            else:
                model.eval()
                with torch.no_grad():
                    feature = model(org_inp, returns_feature=True)
                model.train()
            feat_t = time.time()
            
            if no_aug:
                augmented = noi_inp + oth_inp
            elif random_aug:
                augmented = augment_and_aggregate_batch(noi_inp, oth_inp, None, cfg.aug_prob)
            else:
                alpha = auger(feature)
                ma, mb = alpha.mean(dim=0)
                ma, mb = ma.item(), mb.item()
                alpha_A_avg.update(ma)
                alpha_B_avg.update(mb)
                augmented = augment_and_aggregate_batch(noi_inp, oth_inp, alpha, cfg.aug_prob)
            
            aug__t = time.time()
            logits = model(augmented)
            forw_t = time.time()

            model_op.zero_grad()
            auger_op.zero_grad()
            
            loss = F.cross_entropy(logits, tar)
            backward_to_auger = not no_aug and not random_aug
            loss.backward(retain_graph=backward_to_auger)    # todo: 好像逃不掉retain... 否则两次aug的随机性不一致会出问题，loss和penalty就解耦了，这样不行。
            train_loss_avg.update(loss.item())
            inverse_grad(auger)
            back_t = time.time()

            if backward_to_auger:
                loss_grads = flatten_grads(auger)
                loss_grads_norm = loss_grads.norm().item()
                penalty = cfg.penalty_lambda * (augmented - org_inp).norm() / org_inp.norm()
                penalty.backward()
                pena_grads_norm = flatten_grads(auger).sub_(loss_grads).norm().item()
            else:
                loss_grads_norm = pena_grads_norm = -1
                penalty = torch.tensor(-1)
            penalty_avg.update(penalty.item())
            pena_t = time.time()

            sche_mlr = model_sc.step(loss.item())
            sche_alr = auger_sc.step(penalty.item() - loss.item())
            
            # if cur_it < clipping_iters:   # todo: clipping all the time
            orig_m_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model_grad_clip)
            actual_mlr = sche_mlr * min(1, cfg.model_grad_clip / orig_m_norm)
            if backward_to_auger:
                orig_a_norm = torch.nn.utils.clip_grad_norm_(auger.parameters(), cfg.auger_grad_clip)
                actual_alr = sche_alr * min(1, cfg.auger_grad_clip / orig_a_norm)
            else:
                orig_a_norm = -1
                actual_alr = sche_alr
            # else:
            #     orig_m_norm = cfg.model_grad_clip
            #     orig_a_norm = cfg.auger_grad_clip
            #     actual_mlr = sche_mlr
            #     actual_alr = sche_alr
            clip_t = time.time()

            model_op.step()
            if backward_to_auger:
                auger_op.step()
            optm_t = time.time()
            
            preds = logits.argmax(dim=1)
            tot, correct = tar.shape[0], preds.eq(tar).sum().item()
            train_acc_avg.update(100 * correct / tot)
            
            if (cur_it == 0 or (cur_it + 1) % tb_lg_freq == 0 or cur_it + 1 == max_iter) and rank == 0:
                tb_lg.add_scalar('train/loss', train_loss_avg.avg, cur_it)
                tb_lg.add_scalar('train/penalty', penalty_avg.avg, cur_it)
                tb_lg.add_scalar('train/alpha/A_std', alpha_A_avg.avg, cur_it)
                tb_lg.add_scalar('train/alpha/B_std', alpha_B_avg.avg, cur_it)
                tb_lg.add_scalar('train/acc', train_acc_avg.avg, cur_it)
                tb_lg.add_scalars(f'step/model/lr', {'scheduled': sche_mlr}, cur_it)
                tb_lg.add_scalars(f'step/auger/lr', {'scheduled': sche_alr}, cur_it)
                # if cur_it < clipping_iters:
                tb_lg.add_scalar(f'step/model/orig_norm', orig_m_norm, cur_it)
                tb_lg.add_scalars(f'step/model/lr', {'actual': actual_mlr}, cur_it)
                tb_lg.add_scalar(f'step/auger/orig_norm', orig_a_norm, cur_it)
                tb_lg.add_scalar(f'step/auger/orig_loss_norm', loss_grads_norm, cur_it)
                tb_lg.add_scalar(f'step/auger/orig_pena_norm', pena_grads_norm, cur_it)
                tb_lg.add_scalars(f'step/auger/lr', {'actual': actual_alr}, cur_it)
            
            if it == iters_per_train_ep - 1:  # last iter
                test_loss, test_acc = test(model, iters_per_test_ep, test_iterator)
                topk_acc1s.push_q(test_acc)
                better = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
                test_t = time.time()
                if rank == 0:
                    remain_time, finish_time = epoch_speed.time_preds(cfg.epochs - (epoch + 1))
                    sea_lg.create_or_upd_row(
                        cfg.data.name,
                        be_ac=best_acc, tk_ac=topk_acc1s.mean,
                        pr=(epoch + 1) / cfg.epochs, rem=remain_time.seconds, end_t=(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')) + datetime.timedelta(seconds=remain_time.seconds)).strftime('%Y-%m-%d %H:%M:%S'),
                    )
                    tb_lg.add_scalar('test/loss', test_loss, cur_it)
                    tb_lg.add_scalar('test/acc', test_acc, cur_it)
                    tb_lg.add_scalar('test_ep/loss', test_loss, epoch)
                    tb_lg.add_scalar('test_ep/acc', test_acc, epoch)
                    
                    if epoch % lg_freq == 0 or epoch == cfg.epochs - 1:
                        ep_str = f'%{len(str(cfg.epochs))}d'
                        ep_str %= epoch + 1
                        lg.info(
                            f'ep[{ep_str}/{cfg.epochs}]:'
                            f' te-ac[{test_acc:5.2f}],'
                            f' te-lo[{test_loss:.4f}]'
                            f' tr-ac[{train_acc_avg.last:5.2f}] ({train_acc_avg.avg:5.2f}),'
                            f' tr-lo[{train_loss_avg.last:.4f}] ({train_loss_avg.avg:.4f}),'
                            f' pnt[{penalty_avg.last:.4f}] ({penalty_avg.avg:.4f}), '
                            f' mlr[{sche_mlr:.3g}] ({actual_mlr:.3g}),'
                            f' alr[{sche_alr:.3g}] ({actual_alr:.3g})        be[{best_acc:5.2f}]\n    '
                            
                            f' dat_t[{data_t - last_t:.3f}],'
                            f' cud_t[{cuda_t - data_t:.3f}],'
                            f' fea_t[{feat_t - cuda_t:.3f}],'
                            f' aug_t[{aug__t - feat_t:.3f}],'
                            f' for_t[{forw_t - aug__t:.3f}],'
                            f' bac_t[{back_t - forw_t:.3f}],'
                            f' pen_t[{pena_t - back_t:.3f}],'
                            f' cli_t[{clip_t - pena_t:.3f}],'
                            f' opt_t[{optm_t - clip_t:.3f}],'
                            f' tes_t[{test_t - optm_t:.3f}],'
                            f' rem-t[{str(remain_time)}] ({finish_time})'
                        )
                
                if rank == 0 and better:
                    model_ckpt_path = os.path.join(args.save_path, f'best.pth.tar')
                    print(colorama.Fore.CYAN + f'==> The best ckpt saved (ep[{epoch}], it[{it}], acc{test_acc:5.2f}) at {os.path.abspath(model_ckpt_path)}')
                    torch.save({
                        'start_ep': epoch,
                        'model': model.state_dict(),
                        'model_op': model_op.state_dict(),
                        'model_sc': model_sc.state_dict(),
                        'auger': auger.state_dict(),
                        'auger_op': auger_op.state_dict(),
                        'auger_sc': auger_sc.state_dict(),
                    }, model_ckpt_path)
                
            last_t = time.time()
        
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
        epoch_speed.update(time.time() - epoch_start_t)
    
    topk_acc = topk_acc1s.mean
    if rank == 0:
        if link_dist:
            best_accs = torch.zeros(world_size)
            topk_accs = torch.zeros(world_size)
            best_accs[rank] = best_acc
            topk_accs[rank] = topk_acc
            link.allreduce(best_accs)
            link.allreduce(topk_accs)
            best_acc = best_accs.mean().item()
            topk_acc = topk_accs.mean().item()
            performance_str = (
                f' topk acc: mean: {topk_accs.mean().item():5.2f} (max: {topk_accs.max().item():5.2f}, std: {topk_accs.std():4.2f}), vals: {str(topk_accs).replace(chr(10), " ")}'
                f'@best acc: mean: {best_accs.mean().item():5.2f} (max: {best_accs.max().item():5.2f}, std: {best_accs.std():4.2f}), vals: {str(best_accs).replace(chr(10), " ")}\n'
            )
        
        else:
            performance_str = (
                f'@best acc: {best_acc:5.2f},'
                f' topk acc: {topk_acc:5.2f}'
            )
        
        if rank == 0:
            lg.info(
                f'==> End training,'
                f' total time cost: {time.time() - start_train_t:.3f}\n'
                f'{performance_str}'
            )
            lines = performance_str.splitlines()
            strs = ''.join([f'# {l}\n' for l in lines])
            with open(args.sh_name, 'a') as fp:
                print(f'\n# {args.exp_dir_name}:\n{strs}', file=fp)
            sea_lg.create_or_upd_row(
                cfg.data.name, vital=True,
                be_ac=best_acc, tk_ac=topk_acc,
                pr=1, rem=0, end_t=datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
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

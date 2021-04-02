import os
import pprint
import time
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from colorama import Fore

from data.aug import emd
from utils.misc import time_str


ALL_NAMES = {'_EMI_ratio0.2', '_EMI_ratio0.5', '_EMI_ratio0.8', 'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG200', 'ECG5000', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'}


def __read_data(ckpt, root_path, dname, cache_fname, normalize=True, eemd=True, eemd_name='EEMD', num_workers=None):
    tr_path = os.path.join(root_path, dname, f'{dname.replace(" (1)", "").strip("_")}_TRAIN.tsv')
    te_path = os.path.join(root_path, dname, f'{dname.replace(" (1)", "").strip("_")}_TEST.tsv')

    def get_dl(path):
        items = np.loadtxt(path, delimiter='\t', dtype=np.float32)
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
    avg_ratio = ckpt['curr_emd']
    tr_emd = []
    for x in ckpt['curr_emd']:
        tr_emd.append(x.numpy())
    assert len(tr_emd) == ckpt['start_i']
    
    bar = tqdm(tr_data)
    for i, d in enumerate(bar):
        if i < ckpt['start_i']:
            continue
        n, o = emd(d, eemd=eemd, num_workers=num_workers)
        ratio = 100 * np.abs(np.where(n > 0.1, n, np.zeros_like(n))).mean() / np.abs(d).mean()
        avg_ratio += ratio
        tr_emd.append(np.stack((n, o)))
        if i % 100 == 0:
            t = torch.from_numpy
            torch.save({
                'start_i': i,
                'curr_emd': t(np.stack(tr_emd)).float(),
                'ratio': avg_ratio,
            }, cache_fname)
        # bar.set_description(f'[{dname}]: emd, large_ratio={100 * (n > 0.2).sum() / n.shape[0]:.2f}%')
        bar.set_description(f'[{dname}.{eemd_name}], ratio={ratio:.2f}%')

    avg_ratio /= tr_data.shape[0]
    tr_emd = np.stack(tr_emd)
    print(f'{time_str()}[{dname}.{eemd_name}] cost={time.time() - start_t:.2f}s, avg_ratio={avg_ratio:.2f}%')   # 4 worker: 23s; no parallel: 12s ?
    
    t = torch.from_numpy
    desc = {
        'train_data': t(tr_data).float(),
        'train_emd': t(tr_emd).float(),
        'train_label': t(tr_labels).long(),
        'test_data': t(te_data),
        'test_label': t(te_labels).long(),
        'num_classes': num_classes,
        'ratio': avg_ratio,
    }
    torch.save(desc, cache_fname)
    return desc, avg_ratio


def cache_UCR(root_path: str, fold_idx: int, num_workers=None):
    cached_root = os.path.join(os.path.split(root_path)[0], 'UCRTensorData')
    emd_names = ['EEMD', 'CEEM']
    for name in emd_names:
        if not os.path.exists(f'{cached_root}_{name}'):
            os.makedirs(f'{cached_root}_{name}')
    
    ratios = defaultdict(int)
    dnames = []
    for dname in sorted(os.listdir(root_path)):
        if len(dnames) > 0 and dname.replace(' (1)', '') == dnames[-1]:
            dnames.pop()
        dnames.append(dname)

    fold = [
        ['Worms', 'NonInvasiveFetalECGThorax1', 'ECGFiveDays', 'Computers', 'UWaveGestureLibraryX', 'Earthquakes', 'SemgHandSubjectCh2', 'Phoneme', 'SonyAIBORobotSurface2', 'PickupGestureWiimoteZ', 'ArrowHead', 'ToeSegmentation1', 'GunPoint', 'FiftyWords', 'InsectWingbeatSound', 'ElectricDevices', 'Beef', 'DiatomSizeReduction', 'CinCECGTorso', 'MiddlePhalanxOutlineCorrect', 'EOGVerticalSignal', 'PhalangesOutlinesCorrect', 'PigAirwayPressure', 'GunPointMaleVersusFemale', 'OSULeaf', 'SyntheticControl', 'GunPointAgeSpan', 'Ham', 'MoteStrain', 'HandOutlines', 'Mallat', 'Chinatown', 'SonyAIBORobotSurface1', 'EthanolLevel', 'MelbournePedestrian', 'FordB', 'MixedShapesRegularTrain', 'PigCVP', 'FreezerRegularTrain', 'GestureMidAirD2', 'FaceAll', 'Wafer'],
        ['ACSF1', 'Meat', 'DistalPhalanxOutlineCorrect', 'SmallKitchenAppliances', 'DistalPhalanxTW', 'AllGestureWiimoteX', 'MiddlePhalanxOutlineAgeGroup', 'Strawberry', 'ToeSegmentation2', 'Plane', 'InsectEPGRegularTrain', 'HouseTwenty', 'OliveOil', 'UWaveGestureLibraryY', 'ProximalPhalanxOutlineCorrect', 'Coffee', 'Yoga', 'ChlorineConcentration', 'WordSynonyms', 'PLAID', 'Wine', 'Trace', 'UWaveGestureLibraryZ', 'SmoothSubspace', 'ECG200', 'ECG5000', 'InsectEPGSmallTrain', 'Herring', 'Fungi', 'UWaveGestureLibraryAll', 'SwedishLeaf', 'DodgerLoopDay', 'Crop', 'NonInvasiveFetalECGThorax2', 'FacesUCR', 'PigArtPressure', 'BeetleFly', 'BME', 'Adiac', 'FaceFour', 'FreezerSmallTrain', 'AllGestureWiimoteY', 'GesturePebbleZ2'],
        ['ScreenType', 'StarLightCurves', 'CricketX', 'MiddlePhalanxTW', 'FordA', 'CBF', 'LargeKitchenAppliances', 'BirdChicken', 'ShakeGestureWiimoteZ', 'WormsTwoClass', 'Rock', 'DodgerLoopGame', 'GestureMidAirD3', 'SemgHandGenderCh2', 'ProximalPhalanxTW', 'Haptics', 'Lightning7', 'DistalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'Lightning2', 'CricketY', 'CricketZ', 'PowerCons', 'AllGestureWiimoteZ', 'ProximalPhalanxOutlineAgeGroup', 'TwoLeadECG', 'ShapeletSim', 'GestureMidAirD1', 'InlineSkate', 'MedicalImages', 'DodgerLoopWeekend', 'RefrigerationDevices', 'Symbols', 'SemgHandMovementCh2', 'ShapesAll', 'GesturePebbleZ1', 'Car', 'GunPointOldVersusYoung', 'Fish', 'UMD', 'ItalyPowerDemand'],
        ['_EMI_ratio0.2'],
        ['_EMI_ratio0.5'],
        ['_EMI_ratio0.8'],
    ][fold_idx]

    if len(dnames) != 127 + 3:
        pprint.pprint(dnames)
        pprint.pprint(len(dnames))
        raise AttributeError
    
    for it, org_dname in enumerate(dnames):
        # if len(dname) < 6:
        #     continue
        dname = org_dname.replace(' (1)', '')
        if dname not in fold:
            continue
        for eemd, name in zip([True, False], emd_names):
            cache_fname = os.path.join(f'{cached_root}_{name}', f'{dname}.pth')
            ckpt = {'start_i': 0, 'curr_emd': [], 'curr_ratio': 0}
            if os.path.exists(cache_fname):
                ckpt = torch.load(cache_fname)
            if 'start_i' not in ckpt:
                print(f'{time_str()}[{dname}.{name}] already cached')
            else:
                desc, ratio = __read_data(ckpt, root_path, org_dname, cache_fname, normalize=True, eemd=eemd, eemd_name=name, num_workers=num_workers)
                ratios[name] += ratio
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
            if emd:
                data, tar = desc['train_emd'], desc['train_label']
                tensors = data[:, 0], data[:, 1], tar
            else:
                tensors = desc['train_data'], desc['train_label']
        else:
            tensors = desc['test_data'], desc['test_label']
        for i, t in enumerate(tensors[::-1]):
            if i > 0:
                t.unsqueeze_(dim=1)
                # data: (data_len, 1, input_size)
                # tar:  (data_len,)
        print(f'shapes={[t.shape for t in tensors]}')
        self.input_size = int(tensors[0].shape[-1])
        self.num_classes = desc['num_classes']
        super().__init__(*tensors)

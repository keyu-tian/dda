import random
from typing import Tuple

import numpy as np
import torch as tc
from PyEMD import EEMD, CEEMDAN


def emd(signal: np.ndarray, eemd=True, num_workers=None) -> Tuple[np.ndarray, np.ndarray]:
    kw = dict(trials=100)
    if num_workers is not None:
        kw['parallel'] = True
        kw['processes'] = num_workers
    np_IMFs = (EEMD if eemd else CEEMDAN)(**kw)(signal) # CEEMDAN(parallel=True, processes=4)
    assert len(np_IMFs) > 1
    np_noisy_IMF, np_sum_of_other_IMFs = np_IMFs[0], sum(np_IMFs[1:])
    return np_noisy_IMF, np_sum_of_other_IMFs


def augment_and_aggregate_batch(noise_batch: tc.Tensor, others_batch: tc.Tensor, alpha: tc.Tensor = None, aug_prob=0.5) -> tc.Tensor:
    # batch: (bs, 1, sig_len)
    bs, dev = noise_batch.shape[0], noise_batch.device
    # sig_len = noise_batch[0].shape[-1]
    
    # alpha: (bs, 2)
    if alpha is None:  # random aug
        alpha = 0.05 * tc.rand((bs, 2), device=dev)
    assert alpha.ndim == 2 and alpha.shape[1] == 2
    
    if aug_prob < 1:
        sigmas = alpha * tc.bernoulli(tc.empty(bs, 1), aug_prob).to(device=dev)
    else:
        sigmas = alpha
    
    # sigma: (bs, 1, 1)
    sigma1, sigma2 = sigmas[:, 0:1].unsqueeze(-1), sigmas[:, 1:2].unsqueeze(-1)
    std_norm1 = tc.randn((bs, 1, 1), device=dev)
    std_norm2 = tc.randn((bs, 1, 1), device=dev) / 2
    # A or B: (bs, 1, 1)
    A = sigma1 * std_norm1 + 1
    B = sigma2 * std_norm2
    augmented = A * noise_batch + B # todo: remove B?
    
    return augmented + others_batch


# todo: too slow!
# def _augment_and_aggregate_batch(batch: tc.Tensor, alpha: tc.Tensor = None, k=0.5, aug_prob=0.5) -> tc.Tensor:
#     bs, dev = batch.shape[0], batch.device
#     sig_len = batch[0].shape[-1]
#     aug_len = max(0, min(round(sig_len * k), sig_len))
#
#     # alpha: (bs, 2)
#     if alpha is None:  # random aug
#         alpha = 0.05 * tc.rand((bs, 2), device=dev)
#     assert alpha.ndim == 2 and alpha.shape[1] == 2
#
#     if aug_prob < 1:
#         sigmas = alpha * tc.bernoulli(tc.empty(bs, 1), aug_prob).to(device=dev)
#     else:
#         sigmas = alpha
#
#     # reparameterize
#     sigma1, sigma2 = sigmas[:, 0:1], sigmas[:, 1:2]
#     std_norm1 = tc.randn((bs, aug_len), device=dev)
#     std_norm2 = tc.randn((bs, aug_len), device=dev) / 2
#     A = sigma1 * std_norm1 + 1
#     B = sigma2 * std_norm2  # + 0
#
#     # batch: (bs, num_IMFs, sig_len)
#     assert batch.ndim == 3 and batch.shape[1] == 2
#     # *_IMF*: (bs, sig_len)
#     noisy_IMF, sum_of_other_IMFs = batch[:, 0, :], batch[:, 1, :]
#
#     # augment
#     # todo: too slow!
#     for b in range(bs):
#         aug_bgn = random.randrange(sig_len - aug_len + 1)
#         aug_end = aug_bgn + aug_len
#         noisy_IMF[b, aug_bgn:aug_end] *= A[b]
#         noisy_IMF[b, aug_bgn:aug_end] += B[b]
#
#     # aggregate
#     aggregated = noisy_IMF + sum_of_other_IMFs
#     return aggregated


def test_backward():
    sig = np.random.rand(10)
    noisy_IMF, sum_of_other_IMFs = emd(sig, eemd=False)
    assert np.allclose(sig, noisy_IMF + sum_of_other_IMFs)
    
    import time
    
    bs = 128
    sig_len = 500
    hid_dim = 100
    b1, b2 = tc.rand(bs, 1, sig_len), tc.rand(bs, 1, sig_len)
    fea = tc.rand(bs, hid_dim, requires_grad=True)
    
    aug_net = tc.nn.Linear(hid_dim, 2)
    alpha = aug_net(fea.detach()).sigmoid()
    alpha.retain_grad()
    
    stt = time.time()
    augment_and_aggregate_batch(b1, b2, alpha, aug_prob=0.8).mean().backward()
    print(f'time cost: {(time.time() - stt) * 1000:.2f}ms')
    
    print(alpha.grad.abs().mean(dim=0))
    print(aug_net.weight.grad.abs().mean())
    print(aug_net.bias.grad.abs().mean())


if __name__ == '__main__':
    test_backward()

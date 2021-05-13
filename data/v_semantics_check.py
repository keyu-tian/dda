from PyEMD import EEMD, CEEMDAN
import numpy as np
import pylab as plt

from data import aug
from data.ucr import UCRTensorDataSet
from utils.misc import set_seed


name = 'Adiac'
set_seed(5)
data_set = UCRTensorDataSet(
    r'C:\Users\16333\Desktop\PyCharm\dda\UCRTensorData_EEMD', name,
    train=False, emd=False
)

sin = lambda amp, ome, phi: amp * np.sin(2 * np.pi * ome * t + phi)
normalize = True
S: np.ndarray = data_set[-1][0][0].numpy()
print(S.shape)
t = np.linspace(0, 1, S.shape[0])

# S = sin(np.ones_like(t), 4, 0) + 2 * sin(np.ones_like(t), 16, 0.5)
S = 500 * S
gt = S
inj_noi = 100 * sin(np.ones_like(t), 100, 0)
S += inj_noi

EMD_clz = [
    (False, 'EEMD'.lower()),
    (True, 'CEEMDAN'.lower()),
]

A, B = 0.1 * np.random.randn(1).item(), 0.01 * np.random.randn(1).item()
for using_eemd, clz_name in EMD_clz:

    noise, other = aug.emd(S, eemd=using_eemd)
    auged_noise = (A + 1) * noise + B
    auged_S = other + auged_noise

    inj = int('inj_noi' in globals())
    rows = 7 + inj*2
    plt.figure(figsize=(7, 9))

    if inj:
        plt.subplot(rows, 1, 0 + inj)
        plt.plot(t, globals()['inj_noi'], 'g')
        plt.ylabel('inj')
        plt.locator_params(axis='y', nbins=4)

        plt.subplot(rows, 1, 1 + inj)
        plt.plot(t, globals()['gt'], 'g')
        plt.ylabel('gt')
        plt.locator_params(axis='y', nbins=4)
    
    # S, noise, other
    plt.subplot(rows, 1, 1 + inj*2)
    plt.plot(t, S, 'black')
    plt.ylabel(f'{clz_name}_S')
    plt.locator_params(axis='y', nbins=4)
    
    plt.subplot(rows, 1, 2 + inj*2)
    plt.plot(t, noise, 'red')
    plt.ylabel('noise')
    plt.locator_params(axis='y', nbins=4)
    
    plt.subplot(rows, 1, 3 + inj*2)
    plt.plot(t, other, 'blue')
    plt.ylabel('other')
    plt.locator_params(axis='y', nbins=4)
    
    
    # auged_S, auged_noise
    plt.subplot(rows, 1, 4 + inj*2)
    plt.plot(t, auged_S, 'black')
    plt.ylabel(f'auged_S')
    plt.locator_params(axis='y', nbins=4)
    
    plt.subplot(rows, 1, 5 + inj*2)
    plt.plot(t, auged_noise, 'red')
    plt.ylabel(f'A,B={A:.2g},{B:.2g}')
    plt.locator_params(axis='y', nbins=4)
    
    
    new_noise, new_other = aug.emd(auged_S, eemd=using_eemd)
    # assert np.allclose(new_noise+new_other, auged_S, rtol=1e-4, atol=1e-4)
    # auged_S, auged_noise
    plt.subplot(rows, 1, 6 + inj*2)
    plt.plot(t, new_noise, 'red')
    plt.ylabel(f'new_noise')
    plt.locator_params(axis='y', nbins=4)
    
    plt.subplot(rows, 1, 7 + inj*2)
    plt.plot(t, new_other, 'blue')
    plt.ylabel('new_other')
    plt.locator_params(axis='y', nbins=4)
    
    print(
        f'[{clz_name}]:\n'
        f'  auged_noise-noise (real dis)         =   {np.mean(np.abs(auged_noise-noise))/(np.mean(np.abs(noise)) if normalize else 1):.2g}\n'
        f'  new_noise-auged_noise (~0)   =   {np.mean(np.abs(new_noise-auged_noise))/(np.mean(np.abs(auged_noise)) if normalize else 1):.2g}\n'
        f'  new_other-other (~0)         =   {np.mean(np.abs(new_other-other))/(np.mean(np.abs(other)) if normalize else 1):.2g}\n'
        f'  auged_S-new_noise-new_other (=0)         =   {np.mean(np.abs(auged_S-new_noise-new_other))/(np.mean(np.abs(auged_S)) if normalize else 1):.2g}\n'
    )
    
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

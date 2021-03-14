from PyEMD import EEMD, CEEMDAN
import numpy as np
import pylab as plt


from data.ucr import UCRTensorDataSet
from utils.misc import set_seed


set_seed(0)
data_set = UCRTensorDataSet(
    r'C:\Users\16333\Desktop\PyCharm\dda\UCRTensorData_CEEM', 'ACSF1',
    train=True, emd=False
)


# Define signal
name = 'InlineSkate'
S: np.ndarray = data_set[0][0].numpy()
print(S.shape)
t = np.linspace(0, -1, S.shape[0])
# noise = np.abs(S).mean() * 0.1 * np.random.randn(*t.shape)
# S += noise

# t = np.linspace(0, 1, 200)
# sin = lambda amp, ome, phi: amp * np.sin(2 * np.pi * ome * t + phi)
# noise = 0.5 * np.random.randn(*t.shape)
# # S = 3 * sin(18, 0.2) * (t - 0.2) ** 2
# S = sin(5, 11, 2.7)
# S += sin(3, 14, 1.6)
# S += noise
# # S += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
# # S += t ** 2.1 - t


EMD_clz = [
    (EEMD, 'EEMD'.lower()),
    (CEEMDAN, 'CEEMDAN'.lower()),
]

for clz, clz_name in EMD_clz:
    emd = clz(trials=100)
    
    # Execute EEMD on S
    eIMFs = emd(S)
    nIMFs = eIMFs.shape[0]
    
    # Plot results
    plt.figure(figsize=(7, 9))
    
    inj = 'noise' in globals()
    rows = nIMFs + inj + 3 # raw, sum, aug
    
    if inj:
        plt.subplot(rows, 1, 0 + inj)
        plt.plot(t, globals()['noise'], 'b')
        plt.ylabel('noise')
        plt.locator_params(axis='y', nbins=5)

    a = np.linalg.norm(eIMFs[0]) * 0.1
    aug = eIMFs[0] * (0.6 * a * np.random.randn(*t.shape) + 1)
    aug += 0.1 * a * np.random.randn(*t.shape)
    
    plt.subplot(rows, 1, 1 + inj)
    plt.plot(t, aug, 'b')
    plt.ylabel(f'aug.a={a:.2g}')
    plt.locator_params(axis='y', nbins=5)
    
    for i in range(nIMFs):
        plt.subplot(rows, 1, 1 + inj + i+1)
        plt.plot(t, eIMFs[i], 'g')
        plt.ylabel(f"{clz_name} {i+1}")
        plt.locator_params(axis='y', nbins=5)

    plt.subplot(rows, 1, rows-1)
    plt.plot(t, S, 'r')
    plt.ylabel('raw'+name)
    plt.locator_params(axis='y', nbins=5)

    plt.subplot(rows, 1, rows)
    plt.plot(t, aug + sum(eIMFs[1:]), 'black')
    plt.ylabel('sum')
    plt.locator_params(axis='y', nbins=5)
    
    plt.xlabel("Time [s]")
    plt.tight_layout()

plt.show()

import zarr
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(".."))

from src import orderparameter as op

plt.style.use('../my_style.mplstyle')

S_mean = []
S_std = []

for A in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    folder = f"D:\Sasaki\MTCargoSim\MT\P0.5_A{A:.1f}_kMT0.0904_kcargo0.0226_radius1.18"
    S = op.ensembleS(folder)
    S_mean.append(np.mean(S, axis=0))
    S_std.append(np.std(S, axis=0))


fig, ax = plt.subplots()

for i in range(len(S_mean)):
    ax.plot(np.arange(len(S_mean[i])), S_mean[i], label=f'A={i*0.1:.1f}', color=plt.cm.viridis(i/10))
    ax.fill_between(np.arange(len(S_mean[i])), S_mean[i] - S_std[i], S_mean[i] + S_std[i], alpha=0.3, color=plt.cm.viridis(i/9))


ax.set(xlabel='Time step $\hat{t}$', ylabel='Nematic Order S', xlim = (0, 200))

fig.show()

cd=os.getcwd()

fig.savefig(f"{cd}/figure/neamtic_order_parameter_zoom.png", bbox_inches='tight')
fig.savefig(f"{cd}/figure/nematic_order_parameter_zoom.pdf", bbox_inches='tight')
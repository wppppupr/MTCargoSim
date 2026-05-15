import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


path05 = Path(r"C:\Users\0704w\MTCargoSim\src\julia\test_data\P0.5_A0.5_kMT0.0904_kcargo0.0226_radius0.5")

rs = [0.5]

def get_local(path):
    seeds = sorted(list(path.glob('seed*.zarr')))

    cps = []
    ccs = []

    for seed in seeds:
        cp_path = seed / "center_polar.zarr"
        cc_path = seed / "center_counts.zarr"

        cp_data = zarr.open_array(cp_path, mode='r')[:]
        cc_data = zarr.open_array(cc_path, mode='r')[:]

        cps.append(cp_data)
        ccs.append(cc_data)
    cps = np.array(cps)
    ccs = np.array(ccs)

    thresholds_path = seed / "center_thresholds.zarr"
    threshold = zarr.open_array(thresholds_path, mode='r')[:]

    return cps, ccs, threshold

Acps, Accs, Alps, Alcs = [], [], [], []

for path in [path05]:
    cps, ccs, threshold = get_local(path)
    print(cps.shape)
    Acps.append(cps)

#plt.style.use('../../my_style.mplstyle')

fig, ax = plt.subplots()

for i in range(1):
       cpM = np.mean(Acps[i], axis = (0, 1))
       ax.plot(threshold, cpM, label = f'w/o cargo')

#ax.legend()

ax.set(xlabel = 'Range $\\hat{R}$',
       ylabel = 'Polar order $P$',
       xlim = (1,320),
       ylim=(0.01,1.0),
       xscale = 'log',
       yscale='log')

fig.savefig(r"C:\Users\0704w\MTCargoSim\analysis\graph\figure\center_polar.png")
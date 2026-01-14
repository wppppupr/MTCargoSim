import numpy as np
import glob
import os
import zarr

def orderparameter(orientation):
    cos = np.cos(orientation)
    sin = np.sin(orientation)

    cos2 = 2 * cos**2 - 1
    sin2 = 2 * sin * cos

    S = np.sqrt(np.mean(cos2, axis=1)**2 + np.mean(sin2, axis=1)**2)

    return S

def polarorderparameter(orientation):
    cos = np.cos(orientation)
    sin = np.sin(orientation)

    S = np.sqrt(np.mean(cos, axis=1)**2 + np.mean(sin, axis=1)**2)

    return S

def ensembleS(folder):
    S_list = []

    for f in sorted(glob.glob(os.path.join(folder, "seed*.zarr/order_param.zarr"))):
        S = zarr.open_array(f, mode='r')
        S = S[:]
        S_list.append(S)

    return np.array(S_list)

def ensembleP(folder):
    P_list = []

    for f in sorted(glob.glob(os.path.join(folder, "seed*.zarr/polar_order_param.zarr"))):
        P = zarr.open_array(f, mode='r')
        P = P[:]
        P_list.append(P)

    return np.array(P_list)

if __name__ == "__main__":

    # 実行パスはプロジェクトルートを想定して相対パスを指定
    folder = '/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5'
    save_folder = folder

    print(f"Reading seeds from: {folder}")
    for f in sorted(glob.glob(os.path.join(folder, "seed*.zarr"))):
        print(f"calculate {f}")
        data = zarr.open_array(os.path.join(f, "orientations"), mode='r')
        orientations = data[:].T
        S = orderparameter(orientations)
        zarr.save(os.path.join(f, "nematic_order_param.zarr"), S)
        print(f"    Order parameter S shape: {S.shape}")
        P = polarorderparameter(orientations)
        zarr.save(os.path.join(f, "polar_order_param.zarr"), P)
        print(f"    Polar Order parameter P shape: {P.shape}")

    print("complete!")
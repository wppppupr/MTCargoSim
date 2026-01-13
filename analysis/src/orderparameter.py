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
    files = sorted(glob.glob(os.path.join(folder, "seed*.zarr")))

    if len(files) == 0:
        print(f"Warning: no seed folders found in '{folder}'")
        return np.empty((0,))

    S_lists = []

    for file in files:
        path = os.path.join(file, "orientations")
        if not os.path.exists(path):
            print(f"Warning: orientations file not found: {path}, skipping")
            continue
        orientation = zarr.open_array(path, mode='r')
        orientation = orientation[:].T
        S = orderparameter(orientation)
        S_lists.append(S)
    S_array = np.array(S_lists)

    return S_array

if __name__ == "__main__":

    # 実行パスはプロジェクトルートを想定して相対パスを指定
    folder = '/Volumes/My Passport/Sasaki/MTCargoSim/MT/P0.5_A0.9'
    save_folder = folder

    print(f"Reading seeds from: {folder}")
    for f in sorted(glob.glob(os.path.join(folder, "seed*.zarr"))):
        print(f"calculate {f}")
        data = zarr.open_array(os.path.join(f, "orientations"), mode='r')
        orientations = data[:].T
        S = orderparameter(orientations)
        zarr.save(os.path.join(f, "order_param.zarr"), S)
        print(f"    Order parameter S shape: {S.shape}")

    print("complete!")
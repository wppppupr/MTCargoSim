import numpy as np
import zarr
from pathlib import Path

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
    folder = Path(folder)
    # Match pattern: seed*.zarr/nematic_order_param.zarr
    # Path.glob supports this.
    for f in sorted(folder.glob("seed*.zarr/nematic_order_param.zarr")):
        S = zarr.open_array(str(f), mode='r')
        S = S[:]
        S_list.append(S)

    return np.array(S_list)

def ensembleP(folder):
    P_list = []
    folder = Path(folder)
    for f in sorted(folder.glob("seed*.zarr/polar_order_param.zarr")):
        P = zarr.open_array(str(f), mode='r')
        P = P[:]
        P_list.append(P)

    return np.array(P_list)

if __name__ == "__main__":

    # 実行パスはプロジェクトルートを想定して相対パスを指定
    folder = Path("/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5")
    save_folder = folder

    print(f"Reading seeds from: {folder}")
    for f in folder.glob("seed*.zarr"):
        nematic_path = f / "nematic_order_param.zarr"
        polar_path = f / "polar_order_param.zarr"

        if nematic_path.exists() and polar_path.exists():
                continue
        print(f"calculate {f}")

        orientations_path = f / "orientations"
        data = zarr.open_array(str(orientations_path), mode='r')
        orientations = data[:].T
        S = orderparameter(orientations)
        nematic_zarr = zarr.open(str(nematic_path), mode = 'w', shape=S.shape, dtype = S.dtype)
        nematic_zarr[:] = S
        print(f"    Order parameter S shape: {S.shape}")
        P = polarorderparameter(orientations)
        polar_zarr = zarr.open(str(polar_path), mode = 'w', shape = P.shape, dtype = P.dtype)
        polar_zarr[:] = P
        print(f"    Polar Order parameter P shape: {P.shape}")

    print("complete!")
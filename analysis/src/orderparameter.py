import numpy as np
import glob
import os

def orderparameter(orientation):
    cos = np.cos(orientation)
    sin = np.sin(orientation)

    cos2 = 2 * cos**2 - 1
    sin2 = 2 * sin * cos

    S = np.sqrt(np.mean(cos2, axis=1)**2 + np.mean(sin2, axis=1)**2)

    return S

def ensembleS(folder):
    files = sorted(glob.glob(os.path.join(folder, "seed*")))

    if len(files) == 0:
        print(f"Warning: no seed folders found in '{folder}'")
        return np.empty((0,))

    S_lists = []

    for file in files:
        path = os.path.join(file, "orientations_history.npy")
        if not os.path.exists(path):
            print(f"Warning: orientations file not found: {path}, skipping")
            continue
        orientation = np.load(path)
        S = orderparameter(orientation)
        S_lists.append(S)
    S_array = np.array(S_lists)

    return S_array

if __name__ == "__main__":

    P=0.5
    A=0.5
    # 実行パスはプロジェクトルートを想定して相対パスを指定
    folder = os.path.join("data", "MT", f"P{str(P)}_A{str(A)}")
    save_folder = os.path.join("analysis", "data", "MT", f"P{str(P)}_A{str(A)}")
    os.makedirs(save_folder, exist_ok=True)

    print(f"Reading seeds from: {folder}")
    S = ensembleS(folder)

    if S.size == 0:
        print("No data to save.")
    else:
        outpath = os.path.join(save_folder, "order_param.npy")
        np.save(outpath, S)
        print(f"Saved order parameters to: {outpath}")
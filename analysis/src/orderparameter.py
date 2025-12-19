import numpy as np
import glob
import os

def orderparameter(orientation):
    vec = np.array([np.cos(orientation), np.sin(orientation)])
    cos = vec[0, :, :]
    sin = vec[1, :, :]

    cosS = np.sum(vec[0, :, :], axis=1)
    sinS = np.sum(vec[1, :, :], axis=1)

    mean = np.arctan2(sinS, cosS)

    cosdtheta = cos * np.cos(mean)[:, None] + sin * np.sin(mean)[:, None]
    cos2_Em = np.mean(cosdtheta**2, axis=1)
    S = 2*(cos2_Em - 1/2)

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
    # 実行パスはプロジェクトルートを想定して相対パスを指定
    folder = os.path.join("data", "MT", "P0.5_A0.0")
    save_folder = os.path.join("analysis", "data", "MT", "P0.5_A0.0")
    os.makedirs(save_folder, exist_ok=True)

    print(f"Reading seeds from: {folder}")
    S = ensembleS(folder)

    if S.size == 0:
        print("No data to save.")
    else:
        outpath = os.path.join(save_folder, "order_param.npy")
        np.save(outpath, S)
        print(f"Saved order parameters to: {outpath}")
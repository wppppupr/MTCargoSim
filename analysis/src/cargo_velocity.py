import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

# =============================================================================
# 設定
# =============================================================================
TARGET_PATH = r"/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed*.zarr"
OUTPUT_PLOT = "local_polar_order.png"

shift = 1

# =============================================================================
# 関数定義
# =============================================================================

def get_params(zarr_path):
    """parameters.txt から box_size 等を読み取る"""
    params = {"box_size": 16.0}
    param_path = os.path.join(zarr_path, "parameters.txt")
    
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            for line in f:
                try:
                    if "box_size" in line:
                        params["box_size"] = float(line.split("=")[1].strip())
                except:
                    pass
    return params

def calculate_displacement(zarr_path, shift=1):
    # 1. データ読み込み
    print(f"📂 Loading: {zarr_path}")
    cargo_path = os.path.join(zarr_path, "cargo")
    cargo_zarr = zarr.open_array(cargo_path, mode='r')
    
    # Cargo位置 (Time, 1, 2) または (Time, 2)
    cargo = cargo_zarr[:]
    if cargo.ndim == 3:
        cargo = cargo[:, 0, :]

    cargo = cargo.T

    num_steps, num_particles, dims = cargo.shape
        
    params = get_params(zarr_path)
    L = params["box_size"]
    
    # 3. アンラップ処理 (Vectorized Unwrapping)
    # ループを使わず、numpyで一気に計算します
    
    # 時刻 t と t-1 の差分 (変位) を計算
    # diff[t] = pos[t+1] - pos[t]
    diff = np.diff(cargo, axis=0)

    # 周期境界補正: L/2 以上飛んでいたら ±L して補正
    # round(diff / L) が "何周ワープしたか" の整数になります
    diff -= np.round(diff / L) * L

    # 補正された変位を累積和 (cumsum) して、絶対座標を復元
    # 始点 (t=0) を 0 として、そこからの累積移動量を計算
    unwrapped_trajectory = np.cumsum(diff, axis=0)
        
    # 始点 (0,0) を先頭に追加して形状を元に戻す
    unwrapped_trajectory = np.vstack([
        np.zeros((1, num_particles, dims)), 
        unwrapped_trajectory
    ])

    displacement = unwrapped_trajectory[shift:] - unwrapped_trajectory[:-shift]

    return displacement

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    try:
        for seed in glob.glob(TARGET_PATH):
            displacement = calculate_displacement(seed, shift)

            displacement_path = os.path.join(seed, f"displacement_shift{shift}.zarr")
            polar_output = zarr.open(
                displacement_path,
                mode='w',
                shape = displacement.shape,
                dtype = displacement.dtype
                )
            polar_output[:] = displacement

    except Exception as e:
        print(f"❌ Error: {e}")
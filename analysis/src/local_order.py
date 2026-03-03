import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from get_params import get_params
import argparse
import sys
import os

# Add the project root to sys.path to import data_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_root import data_root

# =============================================================================
# 設定
# =============================================================================
DEFAULT_TARGET_PATH = data_root() / 'Sasaki' / 'MTCargoSim' / 'MTC' / 'P0.5_A0.5_kMT0.0904_kcargo0.0226_radius5.0' / 'seed*.zarr'
OUTPUT_PLOT = "local_polar_order.png"

# =============================================================================
# 関数定義
# =============================================================================

def calculate_local_polar_order(zarr_path, thresholds):
    # 1. データ読み込み
    zarr_path = Path(zarr_path)
    print(f"📂 Loading: {zarr_path}")
    positions_path = zarr_path / "positions"
    orientations_path = zarr_path / "orientations"
    cargo_path = zarr_path / "cargo"

    positions_zarr = zarr.open_array(str(positions_path), mode='r')
    orientations_zarr = zarr.open_array(str(orientations_path), mode='r')
    cargo_zarr = zarr.open_array(str(cargo_path), mode='r')
    
    positions = positions_zarr[:]      
    orientations = orientations_zarr[:] 

    positions = positions.T         # (Time, N, 2)
    orientations = orientations.T   # (Time, N) -> 角度theta
    
    # Cargo位置 (Time, 1, 2) または (Time, 2)
    cargo = cargo_zarr[:]
    if cargo.ndim == 3:
        cargo = cargo[:, 0, :]

    cargo = cargo.T
        
    params = get_params(zarr_path)
    L = params["box_size"]
    
    num_steps, num_particles = orientations.shape
    num_thresholds = len(thresholds)
    print(f"📦 Box Size: {L}, Thresholds: {thresholds}")

    local_polar_orders = np.zeros((num_steps, num_thresholds))
    interacting_counts = np.zeros((num_steps, num_thresholds))

    thresholds_sq = thresholds**2

    # 2. ステップごとに計算
    print("🧮 Calculating local polar order...")
    for t in tqdm(range(num_steps)):
        # 現在の座標と配向
        pos_t = positions[t]   # (N, 2)
        ori_t = orientations[t] # (N,)
        cargo_t = cargo[t]     # (2,)
        
        # --- 距離計算 (PBC考慮) ---
        # Cargoと全粒子の差分ベクトル
        delta = pos_t - cargo_t # Broadcasting: (N, 2) - (2,)
        
        # 周期境界補正 (Nearest Image)
        delta -= np.round(delta / L) * L
        
        # 二乗距離
        dist_sq = np.sum(delta**2, axis=1) # (N,)
        
        # --- 近傍粒子の抽出 (Vectorized over thresholds) ---
        # mask shape: (N, M) where M is num_thresholds
        mask = dist_sq[:, np.newaxis] < thresholds_sq[np.newaxis, :]

        # counts shape: (M,)
        counts = np.sum(mask, axis=0)
        interacting_counts[t] = counts

        # Calculate sums of cos/sin for each threshold
        # mask is boolean, cast to float for matmul?
        # Actually np.dot handles boolean array as 0/1 integers.
        # But explicitly casting might be safer/clearer.

        cos_thetas = np.cos(ori_t) # (N,)
        sin_thetas = np.sin(ori_t) # (N,)

        # Using matrix multiplication: (N,) @ (N, M) -> (M,)
        sum_cos = cos_thetas @ mask
        sum_sin = sin_thetas @ mask

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_cos = sum_cos / counts
            mean_sin = sum_sin / counts
            P = np.sqrt(mean_cos**2 + mean_sin**2)

        # Replace NaNs (where counts == 0) with 0.0
        P[counts == 0] = 0.0

        local_polar_orders[t] = P

    return local_polar_orders, interacting_counts

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate local polar order parameter around cargo.")
    parser.add_argument("target_path", type=str, nargs='?', default=DEFAULT_TARGET_PATH, help="Path pattern for zarr files (e.g. 'data/seed*.zarr')")
    parser.add_argument("--min", type=float, default=0.0, help="Minimum threshold for interaction range")
    parser.add_argument("--max", type=float, default=1.0, help="Maximum threshold for interaction range")
    parser.add_argument("--step", type=float, default=0.01, help="Step size for threshold range")
    parser.add_argument("--threshold", type=float, default=0.28, help="Single threshold value (used if min/max not specified)")

    args = parser.parse_args()

    # Determine thresholds
    if args.min is not None and args.max is not None:
        thresholds = np.arange(args.min, args.max + args.step/1000.0, args.step)
    else:
        thresholds = np.array([args.threshold])

    try:
        target_pattern = Path(args.target_path)
        # Handle wildcard in pattern by checking parent directory
        seeds = list(target_pattern.parent.glob(target_pattern.name))

        if not seeds:
            print(f"⚠️ No files found matching pattern: {target_pattern}")

        for seed in seeds:
            polar_path = seed / "local_polar.zarr"
            counts_path = seed / "counts.zarr"
            thresholds_path = seed / "thresholds.zarr"

            if polar_path.exists():
                 print(f"♻️ Overwriting existing output in {seed}")

            polar_orders, counts = calculate_local_polar_order(seed, thresholds)

            # polar度とカウントの保存
            polar_output = zarr.open(
                str(polar_path),
                mode='w',
                shape = polar_orders.shape,
                dtype = polar_orders.dtype
                )
            polar_output[:] = polar_orders

            counts_output = zarr.open(
                str(counts_path),
                mode = 'w',
                shape = counts.shape,
                dtype = counts.dtype
            )
            counts_output[:] = counts

            # Save thresholds
            thresholds_output = zarr.open(
                str(thresholds_path),
                mode = 'w',
                shape = thresholds.shape,
                dtype = thresholds.dtype
            )
            thresholds_output[:] = thresholds

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

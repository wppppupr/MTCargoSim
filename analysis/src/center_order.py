import zarr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from get_params import get_params
import argparse

# =============================================================================
# 設定
# =============================================================================
DEFAULT_TARGET_PATH = r'/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5_cargo_radius0.59/seed*.zarr'

# =============================================================================
# 関数定義
# =============================================================================

def calculate_center_polar_order(zarr_path, thresholds):
    # 1. データ読み込み
    zarr_path = Path(zarr_path)
    print(f"📂 Loading: {zarr_path}")
    positions_path = zarr_path / "positions"
    orientations_path = zarr_path / "orientations"

    positions_zarr = zarr.open_array(str(positions_path), mode='r')
    orientations_zarr = zarr.open_array(str(orientations_path), mode='r')

    positions = positions_zarr[:]
    orientations = orientations_zarr[:]

    positions = positions.T         # (Time, N, 2)
    orientations = orientations.T   # (Time, N) -> 角度theta

    params = get_params(zarr_path)
    L = params["box_size"]

    num_steps, num_particles = orientations.shape
    num_thresholds = len(thresholds)
    print(f"📦 Box Size: {L}, Thresholds: {thresholds}")

    local_polar_orders = np.zeros((num_steps, num_thresholds))
    interacting_counts = np.zeros((num_steps, num_thresholds))

    # 中心の座標
    center_pos = np.array([L/2, L/2])

    thresholds_sq = thresholds**2

    # 2. ステップごとに計算
    print("🧮 Calculating center polar order...")
    for t in tqdm(range(num_steps)):
        # 現在の座標と配向
        pos_t = positions[t]   # (N, 2)
        ori_t = orientations[t] # (N,)

        # --- 距離計算 (PBC考慮) ---
        # 中心と全粒子の差分ベクトル
        delta = pos_t - center_pos # Broadcasting: (N, 2) - (2,)

        # 周期境界補正 (Nearest Image)
        delta -= np.round(delta / L) * L

        # 二乗距離
        dist_sq = np.sum(delta**2, axis=1) # (N,)

        # --- 近傍粒子の抽出 (Vectorized over thresholds) ---
        # mask shape: (N, M)
        mask = dist_sq[:, np.newaxis] < thresholds_sq[np.newaxis, :]

        # counts shape: (M,)
        counts = np.sum(mask, axis=0)
        interacting_counts[t] = counts

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
    parser = argparse.ArgumentParser(description="Calculate polar order parameter in the center of the box.")
    parser.add_argument("target_path", type=str, nargs='?', default=DEFAULT_TARGET_PATH, help="Path pattern for zarr files (e.g. 'data/seed*.zarr')")
    parser.add_argument("--min", type=float, help="Minimum threshold for interaction range")
    parser.add_argument("--max", type=float, help="Maximum threshold for interaction range")
    parser.add_argument("--step", type=float, default=0.01, help="Step size for threshold range")
    parser.add_argument("--threshold", type=float, default=0.28, help="Single threshold value (used if min/max not specified)")

    args = parser.parse_args()

    # Determine thresholds
    if args.min is not None and args.max is not None:
        thresholds = np.arange(args.min, args.max + args.step/1000.0, args.step)
    else:
        thresholds = np.array([args.threshold])

    target_pattern = Path(args.target_path)

    try:
        # Check if the path is a pattern or a specific file
        # We assume it's a pattern if it contains wildcards or if we want to glob it.
        # However, pathlib doesn't assume wildcards in path construction.
        # But user input might be "seed*.zarr".
        # We use parent.glob(name).

        seeds = list(target_pattern.parent.glob(target_pattern.name))

        if not seeds:
            print(f"⚠️ No files found matching pattern: {target_pattern}")

        for seed in seeds:
            polar_path = seed / "center_polar.zarr"
            counts_path = seed / "center_counts.zarr"
            thresholds_path = seed / "center_thresholds.zarr"

            if polar_path.exists():
                print(f"♻️ Overwriting existing output in {seed}")

            polar_orders, counts = calculate_center_polar_order(seed, thresholds)

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

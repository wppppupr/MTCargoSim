import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
from get_params import get_params
import argparse

# =============================================================================
# 設定
# =============================================================================
DEFAULT_TARGET_PATH = r'/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5/seed*.zarr'
OUTPUT_PLOT = "local_polar_order.png"

# =============================================================================
# 関数定義
# =============================================================================

def calculate_local_polar_order(zarr_path, thresholds):
    # 1. データ読み込み
    print(f"📂 Loading: {zarr_path}")
    positions_path = os.path.join(zarr_path, "positions")
    orientations_path = os.path.join(zarr_path, "orientations")
    cargo_path = os.path.join(zarr_path, "cargo")
    positions_zarr = zarr.open_array(positions_path, mode='r')
    orientations_zarr = zarr.open_array(orientations_path, mode='r')
    cargo_zarr = zarr.open_array(cargo_path, mode='r')
    
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
        
        # --- 近傍粒子の抽出 ---
        for i, threshold in enumerate(thresholds):
            # 閾値以内の粒子のインデックス (Boolean mask)
            mask = dist_sq < threshold**2
            
            count = np.sum(mask)
            interacting_counts[t, i] = count
            
            if count > 0:
                # 近傍粒子の角度を取得
                thetas_local = ori_t[mask]

                # --- ポーラー度 (Polar Order) 計算 ---
                # P = | < e^(i*theta) > |
                #   = sqrt( <cos>^2 + <sin>^2 )

                # ベクトル和をとってから正規化するのと同義
                mean_cos = np.mean(np.cos(thetas_local))
                mean_sin = np.mean(np.sin(thetas_local))

                P = np.sqrt(mean_cos**2 + mean_sin**2)
                local_polar_orders[t, i] = P
            else:
                # 近傍に誰もいない場合は定義できない (0 または NaN)
                local_polar_orders[t, i] = 0.0 # ここでは0とします

    return local_polar_orders, interacting_counts

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate local polar order parameter around cargo.")
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

    try:
        seeds = glob.glob(args.target_path)
        if not seeds:
            print(f"⚠️ No files found matching pattern: {args.target_path}")

        for seed in seeds:
            polar_path = os.path.join(seed, "local_polar.zarr")
            counts_path = os.path.join(seed, "counts.zarr")
            thresholds_path = os.path.join(seed, "thresholds.zarr")

            if os.path.exists(polar_path):
                 print(f"♻️ Overwriting existing output in {seed}")

            polar_orders, counts = calculate_local_polar_order(seed, thresholds)

            # polar度とカウントの保存
            polar_output = zarr.open(
                polar_path,
                mode='w',
                shape = polar_orders.shape,
                dtype = polar_orders.dtype
                )
            polar_output[:] = polar_orders

            counts_output = zarr.open(
                counts_path,
                mode = 'w',
                shape = counts.shape,
                dtype = counts.dtype
            )
            counts_output[:] = counts

            # Save thresholds
            thresholds_output = zarr.open(
                thresholds_path,
                mode = 'w',
                shape = thresholds.shape,
                dtype = thresholds.dtype
            )
            thresholds_output[:] = thresholds

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

import zarr
import numpy as np
import os
from tqdm import tqdm
import glob
from get_params import get_params
import argparse

# =============================================================================
# 設定
# =============================================================================
DEFAULT_TARGET_PATH = r'/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5/seed*.zarr'

# 近傍とみなす距離 (Cargo半径 + α)
# 力学的相互作用範囲、または幾何学的な近接範囲を設定してください
INTERACTION_THRESHOLD = 0.28  # [um]

# =============================================================================
# 関数定義
# =============================================================================

def calculate_center_polar_order(zarr_path, threshold):
    # 1. データ読み込み
    print(f"📂 Loading: {zarr_path}")
    positions_path = os.path.join(zarr_path, "positions")
    orientations_path = os.path.join(zarr_path, "orientations")

    positions_zarr = zarr.open_array(positions_path, mode='r')
    orientations_zarr = zarr.open_array(orientations_path, mode='r')

    positions = positions_zarr[:]
    orientations = orientations_zarr[:]

    positions = positions.T         # (Time, N, 2)
    orientations = orientations.T   # (Time, N) -> 角度theta

    params = get_params(zarr_path)
    L = params["box_size"]

    num_steps, num_particles = orientations.shape
    print(f"📦 Box Size: {L}, Threshold: {threshold}")

    local_polar_orders = np.zeros(num_steps)
    interacting_counts = np.zeros(num_steps)

    # 中心の座標
    center_pos = np.array([L/2, L/2])

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

        # --- 近傍粒子の抽出 ---
        # 閾値以内の粒子のインデックス (Boolean mask)
        mask = dist_sq < threshold**2

        count = np.sum(mask)
        interacting_counts[t] = count

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
            local_polar_orders[t] = P
        else:
            # 近傍に誰もいない場合は定義できない (0 または NaN)
            local_polar_orders[t] = 0.0 # ここでは0とします

    return local_polar_orders, interacting_counts

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate polar order parameter in the center of the box.")
    parser.add_argument("target_path", type=str, nargs='?', default=DEFAULT_TARGET_PATH, help="Path pattern for zarr files (e.g. 'data/seed*.zarr')")
    args = parser.parse_args()

    target_pattern = args.target_path

    try:
        seeds = glob.glob(target_pattern)
        if not seeds:
            print(f"⚠️ No files found matching pattern: {target_pattern}")

        for seed in seeds:
            polar_path = os.path.join(seed, "center_polar.zarr")

            if os.path.exists(polar_path):
                print(f"⏩ Skipping {seed}: output already exists.")
                continue
            else:
                polar_orders, counts = calculate_center_polar_order(seed, INTERACTION_THRESHOLD)

                # polar度とカウントの保存
                polar_output = zarr.open(
                    polar_path,
                    mode='w',
                    shape = polar_orders.shape,
                    dtype = polar_orders.dtype
                    )
                polar_output[:] = polar_orders

                counts_path = os.path.join(seed, "center_counts.zarr")
                counts_output = zarr.open(
                    counts_path,
                    mode = 'w',
                    shape = counts.shape,
                    dtype = counts.dtype
                )
                counts_output[:] = counts

    except Exception as e:
        print(f"❌ Error: {e}")

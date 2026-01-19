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

# 近傍とみなす距離 (Cargo半径 + α)
# 力学的相互作用範囲、または幾何学的な近接範囲を設定してください
INTERACTION_THRESHOLD = 0.28  # [um]

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

def calculate_local_polar_order(zarr_path, threshold):
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
    print(f"📦 Box Size: {L}, Threshold: {threshold}")

    local_polar_orders = np.zeros(num_steps)
    interacting_counts = np.zeros(num_steps)

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
    try:
        for seed in glob.glob(TARGET_PATH):
            if seed == "/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed29.zarr":
                continue

            polar_path = os.path.join(seed, "local_polar.zarr")

            if os.path.exists(polar_path):
                continue
            else:
                polar_orders, counts = calculate_local_polar_order(seed, INTERACTION_THRESHOLD)

                # polar度とカウントの保存
                polar_output = zarr.open(
                    polar_path,
                    mode='w',
                    shape = polar_orders.shape,
                    dtype = polar_orders.dtype
                    )
                polar_output[:] = polar_orders

                counts_path = os.path.join(seed, "counts.zarr")
                counts_output = zarr.open(
                    counts_path,
                    mode = 'w',
                    shape = counts.shape,
                    dtype = counts.dtype 
                )   
                counts_output[:] = counts     

    except Exception as e:
        print(f"❌ Error: {e}")
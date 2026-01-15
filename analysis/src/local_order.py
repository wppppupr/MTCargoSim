import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# =============================================================================
# 設定
# =============================================================================
TARGET_PATH = r"/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed1.zarr"
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
    store = zarr.open_group(zarr_path, mode='r')
    
    positions = store['positions'][:]      # (Time, N, 2)
    orientations = store['orientations'][:] # (Time, N) -> 角度theta
    
    # Cargo位置 (Time, 1, 2) または (Time, 2)
    cargo = store['cargo'][:]
    if cargo.ndim == 3:
        cargo = cargo[:, 0, :]
        
    params = get_params(zarr_path)
    L = params["box_size"]
    
    num_steps, num_particles = orientations.shape
    print(f"📦 Box Size: {L}, Threshold: {threshold} um")

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
        polar_orders, counts = calculate_local_polar_order(TARGET_PATH, INTERACTION_THRESHOLD)
        
        # 時間平均の表示
        # 粒子がいるときだけの平均をとる（0を除外）
        valid_indices = counts > 0
        if np.any(valid_indices):
            avg_p = np.mean(polar_orders[valid_indices])
            print(f"📊 Average Local Polar Order: {avg_p:.3f}")
        else:
            print("⚠️ No interaction detected throughout the simulation.")

        # --- プロット ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        time_steps = np.arange(len(polar_orders))

        # 左軸: ポーラー度
        color = 'tab:red'
        ax1.set_xlabel('Time Step (saved)')
        ax1.set_ylabel('Local Polar Order $P$', color=color)
        ax1.plot(time_steps, polar_orders, color=color, alpha=0.6, linewidth=1, label='Polar Order')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)

        # 移動平均（太線）
        window = 100
        if len(polar_orders) > window:
            ma = np.convolve(polar_orders, np.ones(window)/window, mode='valid')
            ax1.plot(np.arange(len(ma)) + window//2, ma, color='darkred', linewidth=2, label='Moving Avg')

        # 右軸: 粒子数（参考用）
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Number of Neighbors', color=color)  
        ax2.plot(time_steps, counts, color=color, alpha=0.15, linewidth=0.5, label='Count')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Local Polar Order around Cargo (r < {INTERACTION_THRESHOLD} um)')
        fig.tight_layout()  
        
        plt.savefig(OUTPUT_PLOT, dpi=150)
        print(f"✅ Plot saved to: {OUTPUT_PLOT}")

    except Exception as e:
        print(f"❌ Error: {e}")
import zarr
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 設定
# =============================================================================
TARGET_PATH = r"data/MTC/P0.5_A0.5/seed1.zarr"
OUTPUT_PLOT = "interacting_agents.png"

# --- 相互作用の判定距離 (Threshold) ---
# シミュレーション設定に合わせて調整してください。
# 目安: cargo_radius (0.59) + DNA長さなど。
# 力が exp(-(r/ra)^2) で減衰する場合、ra * 2 程度まで見れば十分です。
INTERACTION_THRESHOLD = 0.8  # [um]

# =============================================================================
# 関数定義
# =============================================================================

def get_params(zarr_path):
    """parameters.txt から box_size 等を読み取る"""
    params = {"box_size": 16.0, "cargo_radius": 0.59}
    param_path = os.path.join(zarr_path, "parameters.txt")
    
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            for line in f:
                try:
                    key, val = line.strip().split("=")
                    key = key.strip()
                    val = float(val.strip())
                    params[key] = val
                except:
                    pass
    return params

def count_agents(zarr_path, threshold):
    # 1. データ読み込み
    store = zarr.open_group(zarr_path, mode='r')
    
    # 形状: (Time, N_particles, 2)
    positions = store['positions'][:]
    
    # 形状: (Time, 1, 2) -> (Time, 2) に整形
    cargo = store['cargo'][:]
    if cargo.ndim == 3:
        cargo = cargo[:, 0, :]
        
    params = get_params(zarr_path)
    L = params["box_size"]
    
    num_steps, num_particles, _ = positions.shape
    print(f"📦 Box Size: {L}, Threshold: {threshold} um")

    # 2. 距離計算 (PBC考慮)
    # Numpyのブロードキャスト機能を使って一括計算
    # pos: (T, N, 2), cargo: (T, 1, 2)
    
    # Cargoの位置を (T, 1, 2) に変形して引き算できるようにする
    cargo_reshaped = cargo[:, np.newaxis, :]
    
    # 差分ベクトル (dx, dy)
    delta = positions - cargo_reshaped
    
    # 周期境界補正 (Minimum Image Convention)
    # 箱の半分(L/2)より遠い場合は、反対側の方が近いとみなす
    delta -= np.round(delta / L) * L
    
    # 二乗距離 r^2
    dist_sq = np.sum(delta**2, axis=2)
    
    # 3. カウント
    # 距離が閾値以下のものを数える (Trueは1, Falseは0として足される)
    count_t = np.sum(dist_sq < threshold**2, axis=1)
    
    return count_t

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    try:
        print(f"🚀 Analyzing: {TARGET_PATH}")
        
        # パラメータファイルから閾値を自動推定する場合の例
        params = get_params(TARGET_PATH)
        # 例: Cargo半径 + DNAの遊び + マージン
        estimated_threshold = params.get("cargo_radius", 0.59) + 0.2
        print(f"📏 Estimated Threshold: {estimated_threshold:.3f} um")
        
        # 計算実行
        counts = count_agents(TARGET_PATH, estimated_threshold)
        
        # 時間平均の表示
        print(f"📊 Average interacting agents: {np.mean(counts):.2f}")

        # --- プロット ---
        plt.figure(figsize=(10, 5))
        
        time_steps = np.arange(len(counts))
        plt.plot(time_steps, counts, linewidth=0.5, color='blue', alpha=0.7)
        
        # 移動平均線（トレンドを見やすくするため）
        window = 100
        if len(counts) > window:
            ma = np.convolve(counts, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(len(ma)) + window//2, ma, color='red', linewidth=2, label='Moving Avg')

        plt.xlabel("Time Step (saved)")
        plt.ylabel("Number of Interacting Agents")
        plt.title(f"Interacting Agents (r < {estimated_threshold:.2f} um)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(OUTPUT_PLOT, dpi=150)
        print(f"✅ Plot saved to: {OUTPUT_PLOT}")

    except Exception as e:
        print(f"❌ Error: {e}")
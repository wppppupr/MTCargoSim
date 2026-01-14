import zarr
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 設定
# =============================================================================
# 解析したいZarrデータのパス (Windowsのパス形式に対応)
# 文字列の前に r を付けると \ をそのまま扱えます
TARGET_PATH = r"data/MTC/P0.5_A0.5/seed1.zarr"

# 保存するグラフのファイル名
OUTPUT_PLOT = "cargo_msd.png"

# =============================================================================
# 関数定義
# =============================================================================

def get_box_size(zarr_path):
    """
    parameters.txt から box_size を読み取る関数
    """
    param_path = os.path.join(zarr_path, "parameters.txt")
    box_size = 16.0 # デフォルト値（読み込み失敗時用）
    
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            for line in f:
                if "box_size" in line:
                    try:
                        # "box_size = 16.0" のような形式を想定
                        box_size = float(line.split("=")[1].strip())
                        print(f"📦 Box Size detected: {box_size}")
                    except:
                        pass
    return box_size

def cargo_msd(zarr_path):
    """
    Zarrからcargoデータを読み込み、MSDを計算する
    """
    # 1. データ読み込み
    # mode='r' で読み取り専用モード
    store = zarr.open_group(zarr_path, mode='r')
    
    if 'cargo' not in store:
        raise FileNotFoundError(f"'cargo' array not found in {zarr_path}")

    # 配列としてメモリにロード (Time, N_cargo, 2)
    # 形状: [時間, 粒子数, 座標(xy)]
    cargo_pos = store['cargo'][:]
    
    num_steps, num_particles, dims = cargo_pos.shape
    print(f"📄 Data loaded: {num_steps} steps, {num_particles} cargo(s)")

    # 2. パラメータ取得 (周期境界のサイズ L)
    L = get_box_size(zarr_path)

    # 3. アンラップ処理 (Vectorized Unwrapping)
    # ループを使わず、numpyで一気に計算します
    
    # 時刻 t と t-1 の差分 (変位) を計算
    # diff[t] = pos[t+1] - pos[t]
    diff = np.diff(cargo_pos, axis=0)

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

    # 4. MSD計算
    # (r(t) - r(0))^2
    # 今回は r(0) を 0 に基準化しているので、単に二乗するだけ
    sq_displacement = np.sum(unwrapped_trajectory**2, axis=2) # x^2 + y^2
    
    # 粒子方向の平均をとる (cargoが複数ある場合)
    #msd = np.mean(sq_displacement, axis=1)

    return sq_displacement

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    try:
        print(f"🚀 Analyzing: {TARGET_PATH}")
        msd_data = cargo_msd(TARGET_PATH)
        
        # 時間軸 (ステップ数)
        time_steps = np.arange(len(msd_data))

        # --- プロット ---
        plt.figure(figsize=(8, 6))
        
        # 両対数プロット
        plt.loglog(time_steps, msd_data, label='Cargo MSD', linewidth=2)
        
        # ガイドライン (傾き1: 普通の拡散, 傾き2: バリスティック)
        # 最後の点を基準に線を引く
        if len(time_steps) > 10:
            ref_idx = -10
            # Slope = 1 (Normal Diffusion ~ t^1)
            plt.loglog(time_steps[10:], 
                       time_steps[10:] * (msd_data[ref_idx]/time_steps[ref_idx]), 
                       '--', color='gray', label='Slope=1 (Diffusive)')
            
            # Slope = 2 (Ballistic ~ t^2)
            plt.loglog(time_steps[10:], 
                       (time_steps[10:]**2) * (msd_data[ref_idx]/(time_steps[ref_idx]**2)), 
                       ':', color='red', label='Slope=2 (Ballistic)')

        plt.xlabel(r'Time Step $\Delta t$')
        plt.ylabel(r'Mean Squared Displacement $\langle \Delta r^2 \rangle$')
        plt.title('Cargo Mean Squared Displacement (MSD)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        plt.savefig(OUTPUT_PLOT, dpi=300)
        print(f"✅ Saved plot to: {OUTPUT_PLOT}")
        
        # 簡易的な拡散係数の推定 (D = Slope / 4)
        if len(msd_data) > 100:
            # 後半のデータで傾きを計算
            slope = (msd_data[-1] - msd_data[-100]) / (time_steps[-1] - time_steps[-100])
            print(f"📊 Estimated Diffusion Coefficient D ≈ {slope/4:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
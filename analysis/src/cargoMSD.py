import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to sys.path to import data_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_root import data_root

# =============================================================================
# 設定
# =============================================================================
# 解析したいデータの親ディレクトリ
BASE_PATH = data_root() / 'Sasaki' / 'MTCargoSim' / 'MTC' / 'P0.5_A0.5_kMT0.0904_kcargo0.0226_radius5.0'
# seed*.zarr にマッチするパスのリストを取得
TARGET_SEEDS = sorted(list(BASE_PATH.glob('seed*.zarr')))

# 出力設定
OUTPUT_DATA = BASE_PATH / 'cargo_msd_seeds.zarr'

# =============================================================================
# 関数定義
# =============================================================================

def get_box_size(zarr_path):
    """
    parameters.txt から box_size を読み取る関数
    """

    param_path = zarr_path / "parameters.txt"
    box_size = 16.0 # デフォルト値（読み込み失敗時用）
    
    if param_path.exists():
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
    zarr_path = Path(zarr_path)
    cargo = zarr.open_array(str(zarr_path / "cargo"), mode='r')

    # 配列としてメモリにロード (Time, N_cargo, 2)
    # 形状: [時間, 粒子数, 座標(xy)]
    cargo_pos = cargo[:].T
    
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
    sq_displacement = sq_displacement.reshape(-1)  # 形状: (Time, N_cargo)
    
    # 粒子方向の平均をとる (cargoが複数ある場合)
    #msd = np.mean(sq_displacement, axis=1)

    return sq_displacement

def plot_msd(msd_data, output_path):
    """
    MSDデータをプロットして保存する関数
    """
    plt.figure(figsize=(8,6))
    time_steps = np.arange(msd_data.shape[0])
    
    # 各cargoのMSDを個別にプロット
    for i in range(msd_data.shape[1]):
        plt.plot(time_steps, msd_data[:, i], alpha=0.3)
    
    # 全cargoの平均MSDを太線でプロット
    mean_msd = np.mean(msd_data, axis=1)
    plt.plot(time_steps, mean_msd, color='black', linewidth=2, label='Mean MSD')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Squared Displacement (MSD)")
    plt.title("Cargo Mean Squared Displacement Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Plot saved to: {output_path}")

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    if not TARGET_SEEDS:
        print("❌ No seed directories found. Check your PATH.")
    else:
        try:
            print(f"🚀 Analyzing {len(TARGET_SEEDS)} seeds...")
            msd_list = []
            
            for seed_path in TARGET_SEEDS:
                msd_val = cargo_msd(seed_path)
                msd_list.append(msd_val)
            
            # すべてのseedで Time と N_particles が同じであることを前提としてスタック
            msds = np.array(msd_list) # Shape: (Seeds, Time, Particles)
            print(f"📦 Aggregated Shape: {msds.shape}")

            # Zarr形式で保存
            zarr.save(str(OUTPUT_DATA), msds)
            print(f"💾 MSD data saved to {OUTPUT_DATA}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error: {e}")

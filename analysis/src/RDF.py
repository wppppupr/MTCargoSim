import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
from get_params import get_params

# =============================================================================
# 設定
# =============================================================================
TARGET_PATH = r"data/MTC/P0.5_A0.5"

# RDFの計算設定
MAX_R = 8.0     # 計算する最大距離 (Box size / 2 が目安)
BIN_WIDTH = 0.1 # ヒストグラムのビンの幅

# =============================================================================
# 関数定義
# =============================================================================

def calculate_rdf(zarr_path, max_r, bin_width):
    # 1. データ読み込み
    print(f"📂 Loading: {zarr_path}")
    positions_path = os.path.join(zarr_path, "positions")
    cargo_path = os.path.join(zarr_path, "cargo")
    positions_zarr = zarr.open_array(positions_path, mode='r')
    cargo_zarr = zarr.open_array(cargo_path, mode='r')
    
    positions = positions_zarr[:]      

    positions = positions.T         # (Time, N, 2)
    
    # Cargo位置 (Time, 1, 2) または (Time, 2)
    cargo = cargo_zarr[:]
    if cargo.ndim == 3:
        cargo = cargo[:, 0, :]

    cargo = cargo.T

    params = get_params(zarr_path)
    L = params["box_size"]
    N_total = positions.shape[1]
    
    num_steps = positions.shape[0]
    
    # ヒストグラムのビン準備
    bins = np.arange(0, max_r + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 全ステップのヒストグラムを積算するための配列
    total_hist = np.zeros(len(bin_centers))
    
    print("🧮 Calculating Radial Distribution Function (RDF)...")
    
    # ステップごとに計算して平均をとる
    for t in tqdm(range(num_steps)):
        pos_t = positions[t]
        cargo_t = cargo[t]
        
        # Cargoからの距離計算 (PBC考慮)
        delta = pos_t - cargo_t
        delta -= np.round(delta / L) * L
        dist = np.sqrt(np.sum(delta**2, axis=1))
        
        # ヒストグラム作成
        hist, _ = np.histogram(dist, bins=bins)
        total_hist += hist

    # --- 正規化 (RDFにする処理) ---
    # 平均個数密度 (Global density)
    rho = N_total / (L**2)
    
    # 各ビンの面積 (2次元のドーナツ型領域: π(r+dr)^2 - πr^2 ≈ 2πr dr)
    # これで割ることで「ランダムなら1」になるようにする
    dr = bin_width
    area_shell = np.pi * (bins[1:]**2 - bins[:-1]**2)
    
    # 時間平均をとる
    avg_hist = total_hist / num_steps
    
    # g(r) = (その距離にいた個数) / (面積 * 平均密度)
    g_r = avg_hist / (area_shell * rho)
    
    return bin_centers, g_r

# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    try:
        path = os.path.join(TARGET_PATH, "seed*.zarr")
        for seed in glob.glob(path):
            if seed == "/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed29.zarr":
                continue
            r, g_r = calculate_rdf(TARGET_PATH, MAX_R, BIN_WIDTH)
            RDF = np.array([r, g_r])
            RDF_zarr = zarr.open(os.path.join(seed, "RDF.zarr"), mode='w', shape=RDF.shape, dtype = RDF.dtype)
            RDF_zarr[:] = RDF


    except Exception as e:
        print(f"❌ Error: {e}")
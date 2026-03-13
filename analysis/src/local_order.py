import zarr
import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the project root to sys.path to import data_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_root import data_root
from get_params import get_params

# =============================================================================
# 設定
# =============================================================================
DEFAULT_TARGET_PATH = data_root() / 'Sasaki' / 'MTCargoSim' / 'MTC' / 'P0.5_A0.5_kMT0.0904_kcargo0.0226_radius0.59' / 'seed*.zarr'
OUTPUT_PLOT = "local_polar_order.png"

# =============================================================================
# 関数定義
# =============================================================================

def calculate_local_polar_order(zarr_path, thresholds):
    zarr_path = Path(zarr_path)
    
    positions_path = zarr_path / "positions"
    orientations_path = zarr_path / "orientations"
    cargo_path = zarr_path / "cargo"

    # --- 高速化ポイント1: メモリの連続化 ---
    # np.ascontiguousarrayを使うことで、CPUキャッシュ効率を劇的に改善します
    positions_zarr = zarr.open_array(str(positions_path), mode='r')[:]
    orientations_zarr = zarr.open_array(str(orientations_path), mode='r')[:]
    cargo_zarr = zarr.open_array(str(cargo_path), mode='r')[:]
    
    positions = np.ascontiguousarray(positions_zarr.T)      # (Time, N, 2)
    orientations = np.ascontiguousarray(orientations_zarr.T) # (Time, N)
    
    if cargo_zarr.ndim == 3:
        cargo = cargo_zarr[:, 0, :]
    else:
        cargo = cargo_zarr
    cargo = np.ascontiguousarray(cargo.T) # (Time, 2)
        
    params = get_params(zarr_path)
    L = params["box_size"]
    
    num_steps, num_particles = orientations.shape
    num_thresholds = len(thresholds)

    local_polar_orders = np.zeros((num_steps, num_thresholds))
    interacting_counts = np.zeros((num_steps, num_thresholds), dtype=int)

    thresholds_sq = thresholds**2

    # --- 高速化ポイント2: 事前計算 ---
    cos_all = np.cos(orientations)
    sin_all = np.sin(orientations)

    # 内部ループのtqdmは並列化時に表示が崩れるため削除（メインプロセスで進捗管理します）
    for t in range(num_steps):
        pos_t = positions[t]   
        cargo_t = cargo[t]     
        
        delta = pos_t - cargo_t 
        delta -= np.round(delta / L) * L
        dist_sq = np.sum(delta**2, axis=1) 
        
        # --- 高速化ポイント3: ソートと累積和 (Cumsum) による $O(N \log N)$ アルゴリズム ---
        # 巨大な真偽値行列を作る代わりに、距離順に並べ替えて累積和を取ります
        sort_idx = np.argsort(dist_sq)
        sorted_dist_sq = dist_sq[sort_idx]
        
        # 距離が近い順にcos, sinの累積和を計算
        cumsum_cos = np.cumsum(cos_all[t, sort_idx])
        cumsum_sin = np.cumsum(sin_all[t, sort_idx])
        
        # 各閾値の内側に何個の粒子が含まれるかを二分探索で一括取得
        counts = np.searchsorted(sorted_dist_sq, thresholds_sq, side='left')
        interacting_counts[t] = counts

        # 粒子が存在する閾値のみ計算
        valid_mask = counts > 0
        valid_counts = counts[valid_mask]
        
        # 累積和配列から「O(1)」で該当閾値までの和を取得して平均化
        mean_cos = cumsum_cos[valid_counts - 1] / valid_counts
        mean_sin = cumsum_sin[valid_counts - 1] / valid_counts
        
        local_polar_orders[t, valid_mask] = np.sqrt(mean_cos**2 + mean_sin**2)

    return local_polar_orders, interacting_counts

def process_single_seed(seed, thresholds):
    """マルチプロセス用のラッパー関数"""
    polar_path = seed / "local_polar.zarr"
    counts_path = seed / "counts.zarr"
    thresholds_path = seed / "thresholds.zarr"

    polar_orders, counts = calculate_local_polar_order(seed, thresholds)

    # データの保存
    polar_output = zarr.open(str(polar_path), mode='w', shape=polar_orders.shape, dtype=polar_orders.dtype)
    polar_output[:] = polar_orders

    counts_output = zarr.open(str(counts_path), mode='w', shape=counts.shape, dtype=counts.dtype)
    counts_output[:] = counts

    thresholds_output = zarr.open(str(thresholds_path), mode='w', shape=thresholds.shape, dtype=thresholds.dtype)
    thresholds_output[:] = thresholds

    return seed.name

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

    if args.min is not None and args.max is not None:
        thresholds = np.arange(args.min, args.max + args.step/1000.0, args.step)
    else:
        thresholds = np.array([args.threshold])

    target_pattern = Path(args.target_path)
    seeds = list(target_pattern.parent.glob(target_pattern.name))

    if not seeds:
        print(f"⚠️ No files found matching pattern: {target_pattern}")
        sys.exit(0)

    print(f"🚀 Found {len(seeds)} files. Starting parallel processing...")

    # --- 高速化ポイント4: ProcessPoolExecutorによる並列処理 ---
    # M3チップの全コアをフル稼働させて、複数のZarrファイルを同時に処理します
    max_workers = min(os.cpu_count(), len(seeds))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # タスクをキューに投入
        futures = {executor.submit(process_single_seed, seed, thresholds): seed for seed in seeds}
        
        # 全体の進捗をtqdmで表示
        for future in tqdm(as_completed(futures), total=len(seeds), desc="Processing files"):
            try:
                seed_name = future.result()
            except Exception as e:
                seed = futures[future]
                print(f"\n❌ Error in {seed.name}: {e}")
                import traceback
                traceback.print_exc()
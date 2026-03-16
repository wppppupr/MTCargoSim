import numpy as np
import matplotlib.pyplot as plt
import argparse
import zarr
from pathlib import Path
from matplotlib.animation import FuncAnimation
import sys
import os
import re
from matplotlib.collections import LineCollection

# Add the project root to sys.path to import data_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data_root import data_root

def extract_radius(data_folder):
    """
    Extracts the cargo_radius parameter from parameters.txt inside the data_folder.
    """
    param_file = Path(data_folder) / "parameters.txt"
    if not param_file.exists():
        print(f"Warning: parameters.txt not found at '{param_file}'. Using default radius 0.5.")
        return 0.5
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                if line.startswith("cargo_radius"):
                    return float(line.split("=")[1].strip())
    except Exception as e:
        print(f"Warning: Error reading parameters.txt: {e}")
        
    print("Warning: 'cargo_radius' not found or invalid in parameters.txt. Using default radius 0.5.")
    return 0.5

def animate(data_folder, save_path=None, box_size=16, max_frames=None):
    """
    データからアニメーションを生成し、動画ファイルとして保存します。
    Cargoのサイズはフォルダ名から抽出したradiusを反映します。
    """
    data_folder = Path(data_folder)
    if save_path is None:
        save_path = data_folder
    else:
        save_path = Path(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"データ '{data_folder}' をロード中...")
    try:
        cargo_history = zarr.open_array(str(data_folder / "cargo"), mode='r')
        positions_history = zarr.open_array(str(data_folder / "positions"), mode='r')
        orientations_history = zarr.open_array(str(data_folder / "orientations"), mode='r')
        cargo_history = cargo_history[:].T
        positions_history = positions_history[:].T
        orientations_history = orientations_history[:].T
    except Exception as e:
        print(f"エラー: データファイルが見つかりません。パス '{data_folder}' が正しいか確認してください。\n詳細: {e}")
        return

    if max_frames is not None:
        cargo_history = cargo_history[:max_frames]
        positions_history = positions_history[:max_frames]
        orientations_history = orientations_history[:max_frames]

    print(f"データロード完了。全 {len(positions_history)} フレームをアニメーション化します。")
    
    # parameters.txtからradiusを抽出
    radius_val = extract_radius(data_folder)
    print(f"抽出されたCargo Radius: {radius_val}")

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 最初のフレームのデータで初期化
    initial_positions = positions_history[0]
    initial_orientations = orientations_history[0]
    initial_cargo_position = cargo_history[0]

    ax.set_aspect('equal', adjustable='box')

    # marker sizeの計算
    # 円の半径 (cm)
    marker_radius_cm = radius_val
    # 円の直径 (cm)
    marker_diameter_cm = marker_radius_cm * 2
    # 円の直径 (inch)
    marker_diameter_inch = marker_diameter_cm / 2.54
    # markersize (points) - markersize is diameter in points
    marker_size_pt = marker_diameter_inch * 72

    quiver = ax.quiver(
        initial_positions[:, 0], initial_positions[:, 1],
        np.cos(initial_orientations), np.sin(initial_orientations),
        color="#44AA99", scale=30, headwidth=2, headlength=3, pivot='middle'
    )
    
    # Cargoの軌跡 (trail) 用のLineCollectionプロット
    trail = LineCollection([], cmap='plasma', alpha=0.6, linewidths=2)
    trail.set_clim(vmin=0, vmax=len(positions_history))
    ax.add_collection(trail)
    
    cargo_plot, = ax.plot(initial_cargo_position[0, 0], initial_cargo_position[0, 1], 'o', markersize=marker_size_pt, label="Cargo", color="#332288", alpha=0.8)

    def update(frame):
        current_positions = positions_history[frame]
        current_orientations = orientations_history[frame]
        current_cargo_position = cargo_history[frame]

        quiver.set_offsets(current_positions)
        quiver.set_UVC(np.cos(current_orientations), np.sin(current_orientations))
        
        # 軌跡を更新 (LineCollection用)
        if frame > 0:
            trail_data = cargo_history[:frame+1, 0, :]
            # (N, 1, 2) にして、N-1 個の線分 (start, end) に変換
            points = trail_data.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 周期境界のまたぎを検知して線を切る (box_size/2 以上のジャンプは境界またぎと判定)
            diffs = np.abs(points[1:, 0, :] - points[:-1, 0, :])
            jump_mask = np.any(diffs > box_size / 2.0, axis=1)
            segments[jump_mask] = np.nan
            
            trail.set_segments(segments)
            trail.set_array(np.arange(frame))
        
        cargo_plot.set_data(current_cargo_position[:, 0], current_cargo_position[:, 1])
        ax.set_title(f"time: {frame + 1}/{len(positions_history)} | radius: {radius_val}")
        if (frame + 1) % 50 == 0:
            print(f"  ...アニメーションフレーム {frame + 1}/{len(positions_history)} を処理中")
        return [quiver, trail, cargo_plot]

    ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    
    out_file = save_path / "animation.mov"
    print(f"アニメーションを '{out_file}' に保存しています...")
    ani = FuncAnimation(fig, update, frames=len(positions_history), blit=True, interval=0.1)
    ani.save(str(out_file), writer='ffmpeg', fps=10, dpi=100)
    plt.close(fig)
    print(f"保存が完了しました: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create animations with dynamic cargo radius.")
    parser.add_argument("data_folder", type=str, help="Path to the simulation seed directory (e.g. MTC/P0.5...radius0.315/seed1.zarr)")
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save the animation. Default is inside the data_folder.")
    parser.add_argument("--box_size", type=float, default=16, help="Box size for the animation (default: 16)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to render")
    
    args = parser.parse_args()
    
    try:
        plt.style.use('my_style.mplstyle')
    except Exception:
        print("Warning: Could not load 'my_style.mplstyle'. Proceeding with default styles.")
        
    animate(args.data_folder, args.save_path, args.box_size, args.max_frames)

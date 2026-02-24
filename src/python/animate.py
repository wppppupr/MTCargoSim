import numpy as np
import matplotlib.pyplot as plt
import argparse
import zarr
from pathlib import Path
from matplotlib.animation import FuncAnimation

def animate_onlyMT(data_folder, save_path = "animation/MT", box_size=16):
        """
        保存された時系列データからアニメーションを生成し、動画ファイルとして保存します。

        Args:
            data_prefix (str): データファイル名のプレフィックス (例: "path/to/sim_data_test")
            save_path (str): 保存する動画ファイルのパス (例: "path/to/animation.mov")
        """


        # 保存フォルダを作成
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        data_folder = Path(data_folder)

        print(f"データ '{data_folder}' をロード中...")
        try:
            positions_history = zarr.open_array(str(data_folder / "positions"), mode='r')
            orientations_history = zarr.open_array(str(data_folder / "orientations"), mode='r')
            positions_history = positions_history[:].T
            orientations_history = orientations_history[:].T
        except FileNotFoundError:
            print(f"エラー: データファイルが見つかりません。プレフィックス '{data_folder}' が正しいか確認してください。")
            return

        print(f"データロード完了。全 {len(positions_history)} フレームをアニメーション化します。")

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 最初のフレームのデータで初期化
        initial_positions = positions_history[0]
        initial_orientations = orientations_history[0]

        quiver = ax.quiver(
            initial_positions[:, 0], initial_positions[:, 1],
            np.cos(initial_orientations), np.sin(initial_orientations),
            color='#44AA99', scale=30, headwidth=2, headlength=3, pivot='middle'
        )

        def update(frame):
            current_positions = positions_history[frame]
            current_orientations = orientations_history[frame]

            quiver.set_offsets(current_positions)
            quiver.set_UVC(np.cos(current_orientations), np.sin(current_orientations))
            ax.set_title(f"time: {frame + 1}/{len(positions_history)}")
            if (frame + 1) % 50 == 0:
                print(f"  ...アニメーションフレーム {frame + 1}/{len(positions_history)} を処理中")
            return [quiver]

        ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.legend()
        
        print(f"アニメーションを '{save_path}' に保存しています...")
        ani = FuncAnimation(fig, update, frames=len(positions_history), blit=True, interval=50)
        ani.save(str(save_path / "animation.mov"), writer='ffmpeg', fps=10, dpi=100)
        plt.close(fig)
        print(f"保存が完了しました: {save_path}")

def animate(data_folder, save_path = "animation/MT", box_size=16):
        """
        保存された時系列データからアニメーションを生成し、動画ファイルとして保存します。

        Args:
            data_prefix (str): データファイル名のプレフィックス (例: "path/to/sim_data_test")
            save_path (str): 保存する動画ファイルのパス (例: "path/to/animation.mov")
        """

        # 保存フォルダを作成
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        data_folder = Path(data_folder)

        print(f"データ '{data_folder}/*.npy' をロード中...")
        try:
            cargo_history = zarr.open_array(str(data_folder / "cargo"), mode='r')
            positions_history = zarr.open_array(str(data_folder / "positions"), mode='r')
            orientations_history = zarr.open_array(str(data_folder / "orientations"), mode='r')
            cargo_history = cargo_history[:].T
            positions_history = positions_history[:].T
            orientations_history = orientations_history[:].T
        except FileNotFoundError:
            print(f"エラー: データファイルが見つかりません。プレフィックス '{data_folder}' が正しいか確認してください。")
            return

        print(f"データロード完了。全 {len(positions_history)} フレームをアニメーション化します。")

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 最初のフレームのデータで初期化
        initial_positions = positions_history[0]
        initial_orientations = orientations_history[0]
        initial_cargo_position = cargo_history[0]

        ax.set_aspect('equal', adjustable='box')

        # marker sizeの計算

        # 円の半径 (cm)
        marker_radius_cm = 0.5
        # 円の直径 (cm)
        marker_diameter_cm = marker_radius_cm * 2
        # 円の直径 (inch)
        marker_diameter_inch = marker_diameter_cm / 2.54
        # markersize (points)
        marker_size_pt = marker_diameter_inch * 72


        quiver = ax.quiver(
            initial_positions[:, 0], initial_positions[:, 1],
            np.cos(initial_orientations), np.sin(initial_orientations),
            color="#44AA99", scale=30, headwidth=2, headlength=3, pivot='middle'
        )
        cargo_plot, = ax.plot(initial_cargo_position[0, 0], initial_cargo_position[0, 1], 'ro', markersize=marker_size_pt, label="Cargo", color = "#CC6677", alpha = 0.8)

        def update(frame):
            current_positions = positions_history[frame]
            current_orientations = orientations_history[frame]
            current_cargo_position = cargo_history[frame]

            quiver.set_offsets(current_positions)
            quiver.set_UVC(np.cos(current_orientations), np.sin(current_orientations))
            cargo_plot.set_data(current_cargo_position[:, 0], current_cargo_position[:, 1])
            ax.set_title(f"time: {frame + 1}/{len(positions_history)}")
            if (frame + 1) % 50 == 0:
                print(f"  ...アニメーションフレーム {frame + 1}/{len(positions_history)} を処理中")
            return [quiver, cargo_plot]

        ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        #ax.legend()
        
        print(f"アニメーションを '{save_path}' に保存しています...")
        ani = FuncAnimation(fig, update, frames=len(positions_history), blit=True, interval=0.1)
        ani.save(str(save_path / "animation.mov"), writer='ffmpeg', fps=10, dpi=100)
        plt.close(fig)
        print(f"保存が完了しました: {save_path}")

if __name__ == "__main__":
    data = '/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5/seed1.zarr'
    save_path = '/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5/seed1.zarr'

    plt.style.use('my_style.mplstyle')

    animate(data, save_path=save_path)
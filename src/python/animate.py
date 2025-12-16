import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.animation import FuncAnimation

def animate_onlyMT(P, A, seed=1, box_size=16):
        """
        保存された時系列データからアニメーションを生成し、動画ファイルとして保存します。

        Args:
            data_prefix (str): データファイル名のプレフィックス (例: "path/to/sim_data_test")
            save_path (str): 保存する動画ファイルのパス (例: "path/to/animation.mov")
        """

        data_folder = f"data/P{P}_A{A}/seed{seed}"
        save_folder = f"animation/P{P}_A{A}"
        save_path = f"{save_folder}/seed{seed}.mov"

        # 保存フォルダを作成
        os.makedirs(save_folder, exist_ok=True)

        print(f"データ '{data_folder}/*.npy' をロード中...")
        try:
            positions_history = np.load(f"{data_folder}/positions_history.npy")
            orientations_history = np.load(f"{data_folder}/orientations_history.npy")
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
            ax.set_title(f"Frame: {frame + 1}/{len(positions_history)}")
            if (frame + 1) % 50 == 0:
                print(f"  ...アニメーションフレーム {frame + 1}/{len(positions_history)} を処理中")
            return [quiver]

        ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.legend()
        
        print(f"アニメーションを '{save_path}' に保存しています...")
        ani = FuncAnimation(fig, update, frames=len(positions_history), blit=True, interval=50)
        ani.save(save_path, writer='ffmpeg', fps=20, dpi=150)
        plt.close(fig)
        print(f"保存が完了しました: {save_path}")

def animate(P, A, seed=1, box_size=16):
        """
        保存された時系列データからアニメーションを生成し、動画ファイルとして保存します。

        Args:
            data_prefix (str): データファイル名のプレフィックス (例: "path/to/sim_data_test")
            save_path (str): 保存する動画ファイルのパス (例: "path/to/animation.mov")
        """
        data_folder = f"data/P{P}_A{A}/seed{seed}"
        save_folder = f"animation/P{P}_A{A}"
        save_path = f"{save_folder}/seed{seed}.mov"

        # 保存フォルダを作成
        os.makedirs(save_folder, exist_ok=True)

        print(f"データ '{data_folder}/*.npy' をロード中...")
        try:
            positions_history = np.load(f"{data_folder}/positions_history.npy")
            orientations_history = np.load(f"{data_folder}/orientations_history.npy")
            cargo_history = np.load(f"{data_folder}/cargo_history.npy")
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
            ax.set_title(f"Frame: {frame + 1}/{len(positions_history)}")
            if (frame + 1) % 50 == 0:
                print(f"  ...アニメーションフレーム {frame + 1}/{len(positions_history)} を処理中")
            return [quiver, cargo_plot]

        ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        #ax.legend()
        
        print(f"アニメーションを '{save_path}' に保存しています...")
        ani = FuncAnimation(fig, update, frames=len(positions_history), blit=True, interval=0.1)
        ani.save(save_path, writer='ffmpeg', fps=100, dpi=150)
        plt.close(fig)
        print(f"保存が完了しました: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create animation from simulation data")
    parser.add_argument("--P", type=float, default=0.5, help="Packing fraction")
    parser.add_argument("--A", type=float, default=0.5, help="Alignment strength")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()

    # コマンドライン引数を使って関数を実行
    animate(P=args.P, A=args.A, seed=args.seed)
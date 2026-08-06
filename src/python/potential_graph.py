import os
import numpy as np
import matplotlib.pyplot as plt

def setup_style() -> None:
    """Load the custom matplotlib style sheet."""
    # スクリプトのディレクトリを基準に絶対パスを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, '../../my_style.mplstyle')
    
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        print(f"Warning: Style file not found at {style_path}. Using default styles.")

def ring_gaussian(r: np.ndarray, H: float, r_ring: float, w: float) -> np.ndarray:
    """リング状ガウシアンポテンシャルを計算する"""
    return -H * np.exp(- (r - r_ring)**2 / (2.0 * w**2))

def dot_ring_gaussian(r, H, r_0, w):
    return  - H * ((r-r_0)/w**2) * np.exp(- (r - r_0)**2 / (2.0 * w**2))

def ring_gaussian_cut(r, H, r_0, w, r_cut):
    return np.where(r < r_cut, ring_gaussian(r, H, r_0, w)-ring_gaussian(r_cut, H, r_0, w) + dot_ring_gaussian(r_cut, H, r_0, w)*(r-r_cut), 0)

def calc_r_0(radius: float, d: float) -> float:
    """ポテンシャルが最も深くなる半径 (r_0) を計算する"""
    return np.sqrt(2 * radius * d / (1 + d / (2 * radius))**2)

def calc_r_cut(radius: float, d: float, l: float) -> float:
    """カットオフ距離 (r_cut) を計算する"""
    return np.sqrt((d + 2 * l) * (2 * radius + 2 * l) / (1 + (2 * l + d / 2) / radius)**2)

def main() -> None:
    # グラフのスタイル設定を読み込み
    setup_style()

    # ==========================================
    # パラメータ設定
    # ==========================================
    H = 1
    r_ring_val = 0.4
    w = 0.8
    r_cut_val = r_ring_val + 4*w

    cmap = 'viridis_r'
    
    # ==========================================
    # データ準備
    # ==========================================
    # 描画範囲 (r_cutよりも少し広い範囲を設定)
    limit = 5
    
    # 描画の滑らかさとパフォーマンスのバランスをとるため、グリッドサイズを 500x500 に設定
    x = np.linspace(-limit, limit, 500)
    y = np.linspace(-limit, limit, 500)
    X, Y = np.meshgrid(x, y)
    
    # 原点からの距離 r を計算
    R = np.sqrt(X**2 + Y**2)
    
    # 2次元グリッド上でポテンシャルを計算
    Z = ring_gaussian_cut(R, H, r_ring_val, w, r_cut_val)
    
    # ==========================================
    # 2D 等高線（ヒートマップ）プロット
    # ==========================================

    fig, ax = plt.subplots()


    r = np.linspace(0, 5, 5000)
    # ③ リング状ガウシアン (特定の距離だけポコッと凹む相互作用)
    ax.plot(r, ring_gaussian_cut(r, H, r_ring_val, w, r_cut_val))
    ax.set_xlabel('$\\tilde{r}$')
    ax.set_ylabel('Dimensionless Effective Potential')
    
    #fig.savefig(f'figures/potential.svg')
    fig.savefig(f'/Volumes/data/Sasaki/MTsingleBeads/figure/potential.svg')

    fig1, ax1 = plt.subplots()
    
    # 等高線マップ
    contour = ax1.contourf(X, Y, Z, levels=50, cmap=cmap)
    cbar1 = fig1.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Dimensionless Effective Potential', rotation=270, labelpad=25)
    
    # 最もポテンシャルが深いリング(r_ring)を描画
    circle_ring = plt.Circle(
        (0, 0), 1, color='#ffffff', fill=False, 
        linestyle='--', linewidth=2, label=f'$D_C/2$'
    )
    ax1.add_patch(circle_ring)
    
    
    # カットオフ距離(r_cut)を描画
    circle_cut = plt.Circle(
        (0, 0), r_cut_val, color='#CC6677', fill=False, 
        linestyle=':', linewidth=2, alpha=0.8, label='$r_\mathrm{cut}$'
    )
    ax1.add_patch(circle_cut)
    
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('$2x/D_c$')
    ax1.set_ylabel('$2y/D_c$')
    #ax1.set_title(f'2D Ring Gaussian Potential')
    ax1.legend()

    fig1.savefig(f'/Volumes/data/Sasaki/MTsingleBeads/figure/potential_2d.svg')
    
    # ==========================================
    # 3D 曲面（サーフェス）プロット
    # ==========================================
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # 曲面を描画
    surf = ax2.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.85)
    cbar2 = fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    cbar2.set_label('Dimensionless Effective Potential', rotation=270, labelpad=20)
    
    # 3Dプロットの下部に等高線を投影し、奥行きとポテンシャルの底を分かりやすくする
    z_min = np.min(Z)
    ax2.contour(X, Y, Z, zdir='z', offset=z_min - 0.2, levels=30, cmap=cmap, alpha=0.5)
    

    ax2.set_zlim(z_min - 0.2, np.max(Z) + 0.1)
    ax2.set_xlabel('$2x/D_c$', labelpad=10)
    ax2.set_ylabel('$2y/D_c$', labelpad=10)
    ax2.set_zlabel('Dimensionless Effective Potential', labelpad=15)
    

    #ax2.set_title(f'3D Ring Gaussian Potential')
    
    # 見やすいように初期の視点（カメラアングル）を調整
    #ax2.view_init(elev=35, azim=45)

    fig2.tight_layout()
    fig2.savefig(f'/Volumes/data/Sasaki/MTsingleBeads/figure/potential_3d.svg', bbox_inches='tight')

if __name__ == '__main__':
    main()

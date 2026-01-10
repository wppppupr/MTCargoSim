using Pkg
# プロジェクト環境を有効化
Pkg.activate(".")

# MTC.jl の読み込み
# ※ src/julia/MTC.jl にあると仮定しています
include("MTC.jl")
using .MTC

println("🍎 Mac用テストを開始します...")

# 1. パラメータの設定 (軽く動かすための設定)
params = Parameters(
    packing_fraction = 0.1,
    A = 0.5,
    dt = 0.1,
    seed = 999
)

# 2. 保存先の指定 (Mac用に書き換え)
# プロジェクトフォルダの中に "test_data" というフォルダを作ってそこに保存します
mac_base_path = joinpath(@__DIR__, "test_data")

println("📂 保存先: $mac_base_path")

# 3. シミュレーション実行 (1000ステップだけ回す)
# base_path 引数でデフォルトのWindowsパスを上書きするのがポイントです
run_simulation(params, 1000, 1000; 
    save_interval = 10, 
    base_path = mac_base_path
)

println("✅ テスト完了！ 'test_data' フォルダを確認してください。")
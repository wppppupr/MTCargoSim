using Pkg
# プロジェクト環境を有効化
Pkg.activate(".")

# MTC.jl の読み込み
# ※ src/julia/MTC.jl にあると仮定しています
include("MTC.jl")
using .MTC

# コマンドライン引数の処理 (デフォルト値あり)
cargo_radius     = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 3.34
packing_fraction = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.5
A_val            = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.5
k_MT_val         = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.0904
k_cargo_val      = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 0.0226
seed_val         = length(ARGS) > 5 ? parse(Int, ARGS[6])     : 1000

println("🍎 Mac用テストを開始します...")
println("Parameters: cargo_radius=$cargo_radius, packing_fraction=$packing_fraction, A=$A_val, k_MT=$k_MT_val, k_cargo=$k_cargo_val, seed=$seed_val")

# 1. パラメータの設定 (軽く動かすための設定)
params = Parameters(
    packing_fraction = packing_fraction,
    A = A_val,
    k_MT = k_MT_val,
    k_cargo = k_cargo_val,
    dt = 0.01,
    seed = seed_val,
    cargo_radius = cargo_radius,
    omega = 0.0
)

# 2. 保存先の指定 (Mac用に書き換え)
# プロジェクトフォルダの中に "test_data" というフォルダを作ってそこに保存します
mac_base_path = joinpath(@__DIR__, "test_data")

println("📂 保存先: $mac_base_path")

# 3. シミュレーション実行 (10000ステップだけ回す)
# base_path 引数でデフォルトのWindowsパスを上書きするのがポイントです
run_simulation(params, 10000, 100000; 
    save_interval = 10, 
    base_path = mac_base_path
)

# 保存されたパスを出力してシェルスクリプトに渡す
folder_path = joinpath(mac_base_path, "P$(params.packing_fraction)_A$(params.A)_kMT$(params.k_MT)_kcargo$(params.k_cargo)_radius$(params.cargo_radius)", "seed$(params.seed).zarr")
println("SAVED_PATH: $folder_path")

println("✅ テスト完了！ 'test_data' フォルダを確認してください。")
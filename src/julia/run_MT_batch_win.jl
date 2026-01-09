using Pkg
# プロジェクト環境を有効化
Pkg.activate(".")

# MTC.jl の読み込み
# ※ src/julia/MTC.jl にあると仮定しています
include("MTC.jl")
using .MTC

steps = 300000

println("MTs simulationを開始します...")

for A in [0.0:0.1:1.0;]
    println("Testing with A = $A")
    for seed in 1:10
        # 1. パラメータの設定 (軽く動かすための設定)
        params = Parameters(
            packing_fraction = 0.5,
            A = A,
            seed = seed
        )
        # 3. シミュレーション実行
        MT_simulation(params, steps; 
            save_interval = 100 
        )
    end
end

println("✅ 完了！ フォルダを確認してください。")
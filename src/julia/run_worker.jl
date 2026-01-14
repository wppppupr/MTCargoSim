# src/julia/run_worker.jl
using Pkg
try
    Pkg.activate(dirname(dirname(@__DIR__))) # sim-projectルートを探す
catch
    Pkg.activate(".")
end

using Zarr
using LinearAlgebra
using Distributions

# MTC.jl の読み込み (同じフォルダにある前提)
include(joinpath(@__DIR__, "MTC.jl"))
using .MTC

# --- 設定 ---
const WARMUP = 1500
const STEPS = 1000000
const SAVE_INT = 100
const A = 0.5
const dt = 0.01
const start_seed = 1
const end_seed = 100  

# 全タスクリストを作成 (A: 0.5, Seed: 1~100)
# ※ここを変更すれば計算内容が変わります
const ALL_TASKS = []

for seed in [16, 17, 18, 19, 20, 36, 37, 38, 39, 40, 55, 56, 57, 58, 59, 60, 76, 77, 78, 79, 80, 96, 97, 98, 99, 100]#start_seed:end_seed
    push!(ALL_TASKS, seed)
end

# --- メイン処理 ---
function main()
    # 引数を受け取る (例: julia run_worker.jl 1 20)
    # my_id: 自分の番号 (1〜total_workers)
    # total_workers: 総ワーカー数
    if length(ARGS) < 2
        error("引数が足りません: worker_id total_workers")
    end
    
    my_id = parse(Int, ARGS[1])
    total_workers = parse(Int, ARGS[2])

    println("👷 Worker $my_id / $total_workers 起動: 担当タスクを探します...")

    # 自分の担当分だけループする
    # index が my_id, my_id + total, my_id + 2*total ... のものだけ実行
    count = 0
    for (i, seed) in enumerate(ALL_TASKS)
        # 割り当て判定 (モジュロ演算)
        if (i - 1) % total_workers == (my_id - 1)
            println("  👉 [Worker $my_id] 実行中: Seed=$seed")
            
            try
                params = MTC.Parameters(
                    packing_fraction = 0.5,
                    A = A,
                    dt = dt,
                    seed = seed
                )

                MTC.run_simulation(params, WARMUP, STEPS; save_interval = SAVE_INT)

                # メモリ解放
                GC.gc()
                count += 1
            catch e
                println("  ❌ [Worker $my_id] エラー (A=$A, Seed=$seed): $e")
            end
        end
    end
    
    println("✅ [Worker $my_id] 完了 (処理数: $count)")
end

main()
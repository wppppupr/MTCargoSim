# src/julia/run_worker_vary_k.jl
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
const cargo_radius = 0.315
const start_seed = 1
const end_seed = 201

# --- パラメータ変動設定 ---
# 以下の配列にシミュレーションしたい値を追加してください
const k_MT_values = [9.04e-2]
const k_cargo_values = [2.26e-2]

# 全タスクリストを作成 (A: 0.5, Seed: 1~100)
# ※ここを変更すれば計算内容が変わります
const ALL_TASKS = []

for k_MT in k_MT_values
    for k_cargo in k_cargo_values
        for seed in start_seed:end_seed
            push!(ALL_TASKS, (seed, k_MT, k_cargo))
        end
    end
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
    for (i, (seed, k_MT, k_cargo)) in enumerate(ALL_TASKS)
        # 割り当て判定 (モジュロ演算)
        if (i - 1) % total_workers == (my_id - 1)
            println("  👉 [Worker $my_id] 実行中: Seed=$seed, k_MT=$k_MT, k_cargo=$k_cargo")

            try
                params = MTC.Parameters(
                    packing_fraction = 0.5,
                    A = A,
                    dt = dt,
                    seed = seed,
                    cargo_radius = cargo_radius,
                    k_MT = k_MT,
                    k_cargo = k_cargo
                )

                MTC.run_simulation(params, WARMUP, STEPS; save_interval = SAVE_INT)

                # メモリ解放
                GC.gc()
                count += 1
            catch e
                println("  ❌ [Worker $my_id] エラー (A=$A, Seed=$seed, k_MT=$k_MT, k_cargo=$k_cargo): $e")
                showerror(stdout, e, catch_backtrace())
                println()
            end
        end
    end

    println("✅ [Worker $my_id] 完了 (処理数: $count)")
end

main()

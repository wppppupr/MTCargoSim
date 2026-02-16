using Distributed
using Pkg

# --- 1. プロセスの追加 ---
if nprocs() == 1
    addprocs(Sys.CPU_THREADS)
end

println("現在のワーカー数: $(nprocs())")

# --- 2. 全ワーカーでプロジェクト環境を有効化 ---
@everywhere using Pkg
@everywhere Pkg.activate(".")

# --- 3. コードの読み込み ---
@everywhere begin
    core_path = joinpath(@__DIR__, "MTC.jl")
    if !isfile(core_path)
        error("ファイルが見つかりません: $core_path")
    end
    include(core_path)
    using .MTC
    # NPZ が必要なら有効化 (存在しない環境だとエラーになるので任意)
    try
        using NPZ
    catch
        @warn "NPZ not available on worker; continue without it"
    end
end

# --- メイン処理 ---
function run_for_a(p::Float64, a::Float64; steps::Int=300000, max_seed::Int=201, cargo_radius::Float64=0.315)
    println("開始: packing_fraction=$(p), A=$(a), seeds=1:$max_seed, steps=$steps, cargo_radius = $cargo_radius")

    pmap(1:max_seed) do seed
        params = Parameters(
            packing_fraction = 0.5,
            A = a,
            steps = steps,
            seed = seed,
            cargo_raius = cargo_radius
        )
        run_simulation(params, steps; base_path=base_path)
        return nothing
    end
end

# スクリプトとして直接実行された場合の挙動
if abspath(PROGRAM_FILE) == @__FILE__
    # デフォルト設定: a を 0.0 から 0.5 まで 0.01 刻み、seed は 1..1000
    a = 0.5
    max_seed = 201
    steps = 300000
    cargo_radius = 0.315

    for a in a_values
        @time run_for_a(0.5, a; steps=steps, max_seed=max_seed, cargo_raius = cargo_radius)
    end

    println("✅ 全ての並列計算が完了しました。")
end
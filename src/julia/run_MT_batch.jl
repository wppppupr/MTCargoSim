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
function run_for_a(p::Float64, a::Float64; steps::Int=300000, max_seed::Int=1000)
    println("開始: packing_fraction=$(p), A=$(a), seeds=1:$max_seed, steps=$steps, cargo_raius = $cargo_raius")

    pmap(1:max_seed) do seed
        params = Parameters(
            packing_fraction = p,
            A = a,
            steps = steps,
            seed = seed,
            cargo_raius = cargo_raius
        )
        MT_simulation(params, steps)
        return nothing
    end
end

# スクリプトとして直接実行された場合の挙動
if abspath(PROGRAM_FILE) == @__FILE__
    # デフォルト設定: a を 0.0 から 0.5 まで 0.01 刻み、seed は 1..1000
    a_min = 0.0
    a_max = 0.5
    a_step = 0.01
    max_seed = 1000
    steps = 300000

    # シンプルなコマンドライン引数のサポート: ARGS を順に読む
    # 使い方例: julia run_MT_batch.jl 0.0 0.5 0.01 1000 300000
    if length(ARGS) >= 1
        a_min = parse(Float64, ARGS[1])
    end
    if length(ARGS) >= 2
        a_max = parse(Float64, ARGS[2])
    end
    if length(ARGS) >= 3
        a_step = parse(Float64, ARGS[3])
    end
    if length(ARGS) >= 4
        max_seed = parse(Int, ARGS[4])
    end
    if length(ARGS) >= 5
        steps = parse(Int, ARGS[5])
    end

    a_values = collect(range(a_min, stop=a_max, step=a_step))

    for a in a_values
        @time run_for_a(0.5, a; steps=steps, max_seed=max_seed)
    end

    println("✅ 全ての並列計算が完了しました。")
end
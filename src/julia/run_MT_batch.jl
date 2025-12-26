using Distributed
using Pkg
using Logging

"""
Robust batch runner for MTC simulations.
"""

# 関数1: ワーカーを追加するだけ（usingは含まない）
function setup_workers(; extra_procs::Int=0)
    if nprocs() == 1 && Sys.CPU_THREADS > 1
        addprocs(max(1, Sys.CPU_THREADS - 1 + extra_procs))
    elseif extra_procs > 0
        addprocs(extra_procs)
    end
    @info "現在のワーカー数: $(nprocs())"
end

function run_for_a(p::Float64, a::Float64; steps::Int=300_000, max_seed::Int=1_000)
    @info "開始" packing_fraction=p A=a seeds=1:max_seed steps=steps

    # pmapで並列実行
    res = pmap(1:max_seed) do seed
        try
            # MTCモジュール内の関数を呼び出す
            params = MTC.Parameters(
                packing_fraction = p,
                A = a,
                seed = seed,
            )
            MTC.MT_simulation(params, steps)
            return true
        catch e
            @error "worker error" seed=seed exception=(e, catch_backtrace())
            return false
        end
    end

    n_ok = count(==(true), res)
    @info "完了" succeeded=n_ok total=length(res)
    return res
end

function parse_args(args)
    a_min = 0.0
    a_max = 0.1
    a_step = 0.01
    max_seed = 2
    steps = 30

    if length(args) >= 1; a_min = parse(Float64, args[1]); end
    if length(args) >= 2; a_max = parse(Float64, args[2]); end
    if length(args) >= 3; a_step = parse(Float64, args[3]); end
    if length(args) >= 4; max_seed = parse(Int, args[4]); end
    if length(args) >= 5; steps = parse(Int, args[5]); end

    return a_min, a_max, a_step, max_seed, steps
end

if abspath(PROGRAM_FILE) == @__FILE__
    # 1. ワーカーの準備
    setup_workers()

    # 2. 環境設定とコードの読み込み（ここは関数の外で行う必要があります）
    # パスをマスター側で計算
    project_path = dirname(dirname(@__DIR__))  # プロジェクトルート
    core_path = joinpath(@__DIR__, "MTC.jl")   # MTC.jlの場所

    @everywhere begin
        using Pkg
        # マスターで計算したパス($project_path)を使う
        Pkg.activate($project_path)
        
        # MTC.jl の読み込み
        if !isfile($core_path)
             error("MTC.jl not found at $($core_path)")
        end
        include($core_path)
        
        # モジュールを使用可能にする
        using .MTC
        
        try
            using NPZ
        catch
            @warn "NPZ not available on worker"
        end
    end

    # 3. 引数解析と実行
    a_min, a_max, a_step, max_seed, steps = parse_args(ARGS)
    a_values = collect(range(a_min, stop=a_max, step=a_step))

    for a in a_values
        @time run_for_a(0.5, a; steps=steps, max_seed=max_seed)
    end

    @info "✅ 全ての並列計算が完了しました。"
end
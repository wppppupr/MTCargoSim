using Distributed
using Pkg

# --- 1. プロセスの追加 ---
# すでにワーカーがいる場合は追加しない（二重起動防止）
if nprocs() == 1
    # 論理コア数分のワーカーを追加
    addprocs(Sys.CPU_THREADS)
end

println("現在のワーカー数: $(nprocs())")

# --- 2. 環境の有効化 ---
@everywhere using Pkg
@everywhere Pkg.activate(".") 

# --- 3. コードの読み込み ---
@everywhere begin
    # MTC.jl のパスを特定
    core_path = joinpath(@__DIR__, "MTC.jl")
    
    if !isfile(core_path)
        error("ファイルが見つかりません: $core_path")
    end

    # ファイルを読み込み
    include(core_path)
    
    # ★修正1: MTC.jl のモジュール名 "MTC" に合わせる
    using .MTC 
    using NPZ
end

# --- メイン処理 ---
function main()
    p = 0.5
    a = 0.5
    
    # ★修正2: 変数名を定義 (steps ではなく sim_steps としています)
    sim_steps = 30#0000
    
    max_seed = 1000 

    println("並列計算を開始します... (Target Seeds: 1 to $max_seed)")

    pmap(1:max_seed) do seed
        # 1. パラメータ生成
        # ★修正3: Parametersにstepsは渡さない (構造体に定義されていないため)
        params = Parameters(
            packing_fraction=p, A=a, seed=seed
        )
        
        # 2. 計算実行
        # ★修正4: ここでステップ数(sim_steps)を渡す
        MT_simulation(params, sim_steps)
        
        return nothing 
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
    println("✅ 全ての並列計算が完了しました。")
end
# julia --project=. run_MT_batch.jl
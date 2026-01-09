using Distributed
using Pkg

# ==============================================================================
# 1. パスとファイルの特定 (最重要)
# ==============================================================================
# このスクリプト(run_MT_batch_parallel.jl)と同じフォルダにある "MTC.jl" を探す
const CURRENT_DIR = @__DIR__
const MTC_PATH = joinpath(CURRENT_DIR, "MTC.jl")
const PROJECT_ROOT = dirname(dirname(CURRENT_DIR)) # src/julia の2つ上 (sim-project/)

println("📂 作業ディレクトリ: $CURRENT_DIR")
println("📄 MTCファイル:     $MTC_PATH")

# ==============================================================================
# 2. [メインプロセス] での読み込み (ここが前回と違う点)
# ==============================================================================
println("🧪 Mainプロセスで MTC.jl を読み込み中...")

if !isfile(MTC_PATH)
    error("❌ MTC.jl が見つかりません！ パスを確認してください: $MTC_PATH")
end

# Mainプロセスで先にincludeしておくことで、"UndefVarError" を防ぐ
include(MTC_PATH)
using .MTC
println("✅ Mainプロセス: 読み込み成功")

# ==============================================================================
# 3. ワーカープロセスの起動とセットアップ
# ==============================================================================
# Core i9 (32スレッド想定) に合わせてワーカーを追加
# ※ OS用に2スレッド残す設定
const WORKER_COUNT = Sys.CPU_THREADS - 2

if nprocs() == 1
    println("🚀 ワーカープロセスを $(WORKER_COUNT) 個 追加します...")
    addprocs(WORKER_COUNT)
end

println("⚡ 全プロセス数: $(nprocs())")

# ==============================================================================
# 4. [ワーカープロセス] への読み込み
# ==============================================================================
println("📦 全ワーカーに環境とコードを配布中...")

@everywhere begin
    using Pkg
    using Distributed
    
    # ワーカーにもプロジェクト環境を強制
    try
        Pkg.activate($PROJECT_ROOT)
    catch
        Pkg.activate(".")
    end
    
    # 必要なパッケージ
    using Zarr
    using LinearAlgebra
    using Distributions
    
    # MTC.jl の読み込み
    # Mainで定義したパス変数($MTC_PATH)を使う
    include($MTC_PATH)
    using .MTC
end

println("✅ 全ワーカーのセットアップ完了")

# ==============================================================================
# 5. 並列計算の実行 (Core i9 最適化版)
# ==============================================================================
function main()
    steps = 300000
    save_int = 100
    
    # 計算タスクのリスト作成
    # A: 0.0 ~ 1.0 (0.1刻み), Seed: 1 ~ 10
    tasks = []
    for A in 0.0:0.1:1.0
        for seed in 1:10
            push!(tasks, (A, seed))
        end
    end

    println("🔥 並列計算スタート (タスク数: $(length(tasks)))")

    # pmap で並列実行
    pmap(tasks) do (A, seed)
        # --- ここはワーカーで実行される ---
        
        # MTCモジュールが読み込まれているか最終チェック
        if !isdefined(Main, :MTC)
            error("Worker $(myid()): MTC module is not defined!")
        end

        # パラメータ作成
        params = MTC.Parameters(
            packing_fraction = 0.5,
            A = A,
            seed = seed
        )

        try
            # シミュレーション実行
            # ※ログが混ざるため println は控えめに
            MTC.MT_simulation(params, steps; save_interval = save_int)
        catch e
            # エラー発生時も他の計算を止めないようにキャッチする
            println("❌ Error [A=$A, Seed=$seed]: $e")
        end

        # メモリ解放 (Core i9のメモリ効率維持のため)
        GC.gc()
        
        return nothing
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
    println("🎉 すべてのシミュレーションが完了しました！")
end
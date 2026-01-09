using Distributed
using Pkg

# ==============================================================================
# 1. PCの性能を引き出す設定 (Core i9 & 64GB RAM用)
# ==============================================================================

# 実行したい同時並列数
# Core i9は通常 16~24コア / 32スレッド程度あります。
# RAM 64GBなら、1プロセス2GB消費しても 20〜25プロセスは余裕で動かせます。
# OS操作用に少し余裕を残して、スレッド数 - 2 くらいを設定します。
const WORKER_COUNT = Sys.CPU_THREADS - 2

if nprocs() == 1
    println("🚀 Core i9のパワーを解放します: ワーカー $(WORKER_COUNT) 個を追加中...")
    addprocs(WORKER_COUNT)
end

println("⚡ 現在のプロセス数: $(nprocs()) (Main + Workers)")

# ==============================================================================
# 2. ワーカーへの準備 (コードと環境の配布)
# ==============================================================================

# パスの設定 (Windows対応)
const MTC_PATH = joinpath(@__DIR__, "MTC.jl")
const PROJECT_ROOT = dirname(@__DIR__) # 必要に応じて調整してください (Project.tomlがある場所)

@everywhere begin
    using Pkg
    # ワーカーが環境を見失わないように明示的にactivate
    # パスがうまく認識されない場合は、絶対パスをハードコードするか、try-catchでカレントを使います
    try
        Pkg.activate($PROJECT_ROOT) 
    catch
        Pkg.activate(".")
    end
    
    # 必要なパッケージ
    using Distributed
    using Zarr
    
    # MTC.jl の読み込み
    if !isfile($MTC_PATH)
        error("MTC.jl が見つかりません: $($MTC_PATH)")
    end
    include($MTC_PATH)
    using .MTC
end

# ==============================================================================
# 3. 計算タスクの定義と実行
# ==============================================================================

function main()
    steps = 300000
    save_int = 100
    
    # パラメータの組み合わせリストを作成 (A: 0.0~1.0, seed: 1~10)
    # これで合計 11 x 10 = 110個のタスクになります
    tasks = []
    for A in 0.0:0.1:1.0
        for seed in 1:10
            push!(tasks, (A, seed))
        end
    end

    println("🧪 合計タスク数: $(length(tasks))")
    println("🔥 並列計算を開始します...")

    # pmap: 空いたワーカーから順次タスクを消化する賢い関数
    # batch_size=1 : タスクが重い場合は1つずつ割り振るのが効率的
    pmap(tasks; on_error=ex->Base.display_error(ex, catch_backtrace())) do (A, seed)
        
        # --- ここは各ワーカーで実行される ---
        
        # 1. パラメータ作成
        params = MTC.Parameters(
            packing_fraction = 0.5,
            A = A,
            seed = seed
        )

        # 2. シミュレーション実行
        # ログが混ざらないようにファイルパスなどを表示
        # println("  Worker $(myid()) -> Processing A=$A, Seed=$seed")
        
        try
            MT_simulation(params, steps; save_interval = save_int)
        catch e
            println("❌ Error at A=$A, Seed=$seed: $e")
        end

        # 3. メモリ解放 (連続稼働時のリーク防止)
        GC.gc()
        
        return nothing
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
    println("✅ すべてのシミュレーションが完了しました！")
end
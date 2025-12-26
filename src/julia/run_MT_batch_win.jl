using Distributed
using Pkg

# ==============================================================================
# 1. パスと設定の確定 (Mainプロセス)
# ==============================================================================
# このファイルの場所(@__DIR__)を基準に、MTC.jlの絶対パスを作る
const MTC_FILE_PATH = joinpath(@__DIR__, "MTC.jl")
# プロジェクトのルートディレクトリ
const PROJECT_ROOT = dirname(dirname(@__DIR__)) 

println("📂 プロジェクトルート: $PROJECT_ROOT")
println("📄 MTCファイルパス:   $MTC_FILE_PATH")

# まずメインプロセス(自分のPC)で読み込めるかテスト
# ここでエラーが出れば、ファイル自体に問題があることがわかります
println("🧪 メインプロセスでの読み込みテスト...")
if !isfile(MTC_FILE_PATH)
    error("❌ エラー: ファイルが見つかりません -> $MTC_FILE_PATH")
end
include(MTC_FILE_PATH)
using .MTC
println("✅ メインプロセス: MTCモジュール読み込み成功")


# ==============================================================================
# 2. ワーカープロセスのセットアップ
# ==============================================================================
if nprocs() == 1
    println("👷 ワーカープロセスを追加中 (Core数: $(Sys.CPU_THREADS))...")
    addprocs(Sys.CPU_THREADS)
end
println("⚡ 現在のワーカー数: $(nprocs())")


# ==============================================================================
# 3. ワーカーへの環境・コード配布 (段階的に実行)
# ==============================================================================

# ステップ1: 環境(Project.toml)のアクティベート
println("📦 [Step 1] 全ワーカーの環境設定...")
@everywhere begin
    using Pkg
    # ワーカーにプロジェクトルートを教えて環境を有効化させる
    try
        Pkg.activate($PROJECT_ROOT)
    catch
        Pkg.activate(".") # 保険
    end
end

# ステップ2: ファイルの読み込み (include)
println("📂 [Step 2] 全ワーカーで MTC.jl を読み込み...")
@everywhere include($MTC_FILE_PATH)

# ステップ3: モジュールの使用宣言 (using)
# includeと分けることで、確実に読み込み後に実行させる
println("🔗 [Step 3] 全ワーカーで using .MTC を実行...")
@everywhere using .MTC
@everywhere using NPZ


# ==============================================================================
# 4. シミュレーション実行ロジック
# ==============================================================================
function run_parameter_sweep(p::Float64, a::Float64; steps::Int=300000, max_seed::Int=1000)
    println("🚀 計算開始: P=$(p), A=$(a), Seeds=1:$(max_seed)")

    pmap(1:max_seed) do seed
        # 各ワーカーでの処理
        # MTCモジュールが読み込まれている前提で実行
        params = MTC.Parameters(
            packing_fraction = p,
            A = a,
            seed = seed
        )
        
        MTC.MT_simulation(params, steps)
        return nothing
    end
end

# ==============================================================================
# 5. メイン実行部
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    # コマンドライン引数処理
    a_min = 0.0
    a_max = 0.5
    a_step = 0.1
    max_seed = 1
    steps = 30#0000

    args = ARGS
    if length(args) >= 1; a_min = parse(Float64, args[1]); end
    if length(args) >= 2; a_max = parse(Float64, args[2]); end
    if length(args) >= 3; a_step = parse(Float64, args[3]); end
    if length(args) >= 4; max_seed = parse(Int, args[4]); end
    if length(args) >= 5; steps = parse(Int, args[5]); end

    a_values = collect(a_min:a_step:a_max)
    
    for a in a_values
        @time run_parameter_sweep(0.5, a; steps=steps, max_seed=max_seed)
    end

    println("✅ 全ての計算が完了しました。")
end
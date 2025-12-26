using Distributed
using Pkg

# ==============================================================================
# 1. ワーカープロセスのセットアップ (Windows対応版)
# ==============================================================================
function setup_workers()
    # 既にワーカーがいる場合は追加しない（二重起動防止）
    if nprocs() == 1
        # 論理コア数だけプロセスを追加（メインプロセス分を除く必要があれば調整）
        addprocs(Sys.CPU_THREADS)
    end
    println("✅ ワーカー数: $(nprocs()) (Main + Workers)")
end

# 最初にワーカーを準備
setup_workers()

# ==============================================================================
# 2. 全ワーカーへのコードと環境の配布
# ==============================================================================
# パス関係をメインプロセスで確定させる
const PROJECT_ROOT = dirname(dirname(@__DIR__))  # src/julia/ の2つ上がルートと仮定
const MTC_FILE_PATH = joinpath(@__DIR__, "MTC.jl")

println("📂 プロジェクトルート: $PROJECT_ROOT")
println("📄 MTCファイルパス:   $MTC_FILE_PATH")

# @everywhere ブロック：全プロセスで実行されるコード
@everywhere begin
    using Pkg
    
    # 1. 環境のアクティベート
    # $PROJECT_ROOT を使うことで、ワーカーがどこにいても正しいProject.tomlを見つける
    try
        Pkg.activate($PROJECT_ROOT)
    catch
        # パス計算がずれた場合の保険（カレントディレクトリ）
        Pkg.activate(".") 
    end

    # 2. 必要なパッケージの読み込み
    try
        using NPZ
        using LinearAlgebra
        using Distributions
    catch e
        @error "パッケージの読み込みに失敗しました。'julia --project=. ...' で実行していますか？" exception=e
    end

    # 3. MTC.jl の読み込み
    # モジュールファイルが見つかるかチェック
    if !isfile($MTC_FILE_PATH)
        error("致命的エラー: MTC.jl が見つかりません -> $($MTC_FILE_PATH)")
    end

    include($MTC_FILE_PATH)
    
    # モジュール名は 'MTC.jl' の中身に合わせて 'MTC' とする
    using .MTC
end

# 読み込み確認（診断用）
function check_workers()
    println("🔍 ワーカーの読み込み状況チェック...")
    responses = pmap(w -> myid(), workers())
    println("  -> 全ワーカー($(length(responses))機) が正常に応答しました。")
end

check_workers()

# ==============================================================================
# 3. シミュレーション実行ロジック
# ==============================================================================
function run_parameter_sweep(p::Float64, a::Float64; steps::Int=300000, max_seed::Int=1000)
    println("🚀 計算開始: P=$(p), A=$(a), Seeds=1:$(max_seed)")

    # pmap: 空いているワーカーにタスクを自動配分
    pmap(1:max_seed) do seed
        
        # --- ここは各ワーカーで実行される ---
        
        # 【重要修正】
        # Parameters構造体には 'steps' は含まれていません。
        # steps は MT_simulation 関数に直接渡します。
        params = MTC.Parameters(
            packing_fraction = p,
            A = a,
            seed = seed
        )
        
        # シミュレーション実行
        MTC.MT_simulation(params, steps)
        
        return nothing
    end
end

# ==============================================================================
# 4. メイン実行部
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    # --- コマンドライン引数の処理 ---
    # デフォルト値
    a_min = 0.0
    a_max = 0.5
    a_step = 0.1 # テスト用に少し粗くしています
    max_seed = 10 # テスト用に少なくしています
    steps = 300000

    # 引数があれば上書き (順序: min max step seed steps)
    args = ARGS
    if length(args) >= 1; a_min = parse(Float64, args[1]); end
    if length(args) >= 2; a_max = parse(Float64, args[2]); end
    if length(args) >= 3; a_step = parse(Float64, args[3]); end
    if length(args) >= 4; max_seed = parse(Int, args[4]); end
    if length(args) >= 5; steps = parse(Int, args[5]); end

    # 計算対象のAのリスト
    a_values = collect(a_min:a_step:a_max)
    
    # 実行ループ
    for a in a_values
        @time run_parameter_sweep(0.5, a; steps=steps, max_seed=max_seed)
    end

    println("✅ 全ての計算が完了しました。")
end
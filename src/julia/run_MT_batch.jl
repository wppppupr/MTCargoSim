using Distributed
using Pkg

# --- 1. プロセスの追加 ---
# Windowsでは論理コア数(Sys.CPU_THREADS)を使うのが一般的です
# 既存のプロセスに加えて、コア数分のワーカーを追加します
if nprocs() == 1
    addprocs(Sys.CPU_THREADS)
end

println("現在のワーカー数: $(nprocs())")

# --- 2. 全ワーカーでプロジェクト環境を有効化 (超重要) ---
# Windowsのワーカーは環境設定を引き継がないことがあるため、明示的にactivateします
@everywhere using Pkg
@everywhere Pkg.activate(".") 

# --- 3. コードの読み込み ---
@everywhere begin
    # パスをOS依存しない形 (@__DIR__ と joinpath) で指定します
    # このファイル(run_batch.jl)と同じフォルダにある SimulationCore.jl を探します
    core_path = joinpath(@__DIR__, "MTC.jl")
    
    if !isfile(core_path)
        error("ファイルが見つかりません: $core_path")
    end

    include(core_path)
    using .onlyMT
    using NPZ
end

# --- メイン処理 ---
function main()
    # テスト用パラメータ
    p = 0.5
    a = 0.5
    step = 300000
    max_seed = 1000 # 計算回数

    println("並列計算を開始します... (Target Seeds: 1 to $max_seed)")

    # pmap: 自動的に負荷分散して並列実行してくれる関数
    # プログレスバー等はここに入れると表示が乱れるので、簡易表示にしています
    pmap(1:max_seed) do seed
        # 各ワーカー内での処理
        
        # 1. パラメータ生成
        params = Parameters(
            packing_fraction=p, A=a, steps = steps, seed=seed
        )
        
        # 2. 計算実行 (SimulationCore内の関数)
        # ログが出すぎると遅くなるので、printlnは控えめに
        MT_simulation(params, step)
        
        # 戻り値として何か返したい場合はここで返す（今回はファイル保存済みなので不要）
        return nothing 
    end
end

# スクリプトとして実行された時だけ main() を呼ぶ
if abspath(PROGRAM_FILE) == @__FILE__
    # 計測用 (@time)
    @time main()
    println("✅ 全ての並列計算が完了しました。")
end
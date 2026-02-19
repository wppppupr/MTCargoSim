# launcher_vary_k.jl
# 独立したJuliaプロセスを大量に立ち上げるスクリプト (k_MT, k_cargo 変動版)

# 同時起動数 (PCのスペックに合わせて調整)
# Core i9 / 64GB RAM なら 20〜24 くらいはいけます
const NUM_WORKERS = 20

println("🚀 手動並列化ランチャーを起動します (k_MT/k_cargo 変動)")
println("🔥 同時実行数: $NUM_WORKERS プロセス")

# ワーカーのパス
worker_script = joinpath("src", "julia", "run_worker_vary_k.jl")

# プロセスを次々と立ち上げる
for i in 1:NUM_WORKERS
    println("  -> Worker $i を起動中...")

    # 1. "cmd /c start" を使って、新しいウィンドウでJuliaを起動するコマンドを作る
    #    "Worker $i" はウィンドウのタイトルになります
    cmd = `cmd /c start "Worker $i (vary k)" julia --project=. $worker_script $i $NUM_WORKERS`

    # 2. 実行する
    run(cmd, wait=false)
end

println("🎉 全プロセスの起動指令を出しました！")
println("⚠️  注意: 黒い画面が $NUM_WORKERS 個立ち上がります。")
println("    すべて閉じたら計算終了です。")

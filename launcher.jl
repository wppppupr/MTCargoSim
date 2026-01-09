# launcher.jl
# 独立したJuliaプロセスを大量に立ち上げるスクリプト

# 同時起動数 (PCのスペックに合わせて調整)
# Core i9 / 64GB RAM なら 20〜24 くらいはいけます
const NUM_WORKERS = 20

println("🚀 手動並列化ランチャーを起動します")
println("🔥 同時実行数: $NUM_WORKERS プロセス")

# ワーカーのパス
worker_script = joinpath("src", "julia", "run_worker.jl")

# プロセスを次々と立ち上げる
for i in 1:NUM_WORKERS
    println("  -> Worker $i を起動中...")
    
    # run(..., wait=false) でバックグラウンド実行します
    # open_new_console=true (Windowsのみ) をつけると、
    # 別の黒い画面がたくさん出てきて、それぞれの進捗が見えます（カッコいいです）
    cmd = `julia --project=. $worker_script $i $NUM_WORKERS`
    
    # Windowsで別窓を開くオプション (進捗が見えるように)
    run(Cmd(cmd, windows_verbatim=true, dir=pwd()), wait=false)
end

println("🎉 全プロセスの起動指令を出しました！")
println("⚠️  注意: 黒い画面が $NUM_WORKERS 個立ち上がります。")
println("    すべて閉じたら計算終了です。")
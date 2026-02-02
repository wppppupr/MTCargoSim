using Pkg

println("🛠️  linux環境セットアップを開始します...")

# 1. Juliaパッケージのインストール
println("\n Juliaパッケージをインストール中...")
Pkg.activate(".")
Pkg.instantiate()

println("\n✅ セットアップ完了！")
println("実行するには以下のコマンドを使ってください:")
println("  julia --project=. src/julia/run_batch.jl")
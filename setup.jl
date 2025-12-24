using Pkg

println("🛠️  Windows環境セットアップを開始します...")

# 1. Juliaパッケージのインストール
println("\n[1/2] Juliaパッケージをインストール中...")
Pkg.activate(".")
Pkg.instantiate()

# 2. Python仮想環境の構築
println("\n[2/2] Python仮想環境(venv)を構築中...")

# venvフォルダがなければ作成
if !isdir("venv")
    println("  -> venvを作成します...")
    run(`python -m venv venv`)
else
    println("  -> venvは既に存在します。")
end

# WindowsとMac/Linuxでpipの場所が違うため自動判定
pip_path = ""
if Sys.iswindows()
    pip_path = joinpath("venv", "Scripts", "pip")
else
    pip_path = joinpath("venv", "bin", "pip")
end

# requirements.txt のインストール
if isfile("requirements.txt")
    println("  -> ライブラリをインストールします...")
    run(`$pip_path install -r requirements.txt`)
else
    println("⚠️ requirements.txt が見つかりません。Pythonのセットアップをスキップします。")
end

println("\n✅ セットアップ完了！")
println("実行するには以下のコマンドを使ってください:")
println("  julia --project=. src/julia/run_batch.jl")
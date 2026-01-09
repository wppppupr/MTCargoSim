module MTC

# using NPZ  <-- 削除
using Zarr   # <-- 追加
using LinearAlgebra
using Distributions
using ProgressMeter
using Random

export Parameters, Datas, run_simulation, MT_simulation

# ... (Parameters, Datas, helper関数などはそのまま) ...

# ----------------------------------------------------------------
# MT_simulation 関数の修正
# ----------------------------------------------------------------
function MT_simulation(params::Parameters, num_steps::Int; save_interval::Int=100)
    # ... (初期化やループ処理は変更なし) ...
    # ... (前回の回答でのメモリ対策/間引きロジックはそのまま維持してください) ...

    # --- 保存パスの作成 ---
    # NASパスの修正 (前回の議論に基づき joinpath 推奨)
    base_path = "\\\\NAS-Ebanaru\\data\\Sasaki\\backup_git\\MTCargoSim\\data\\MT"
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)", "seed$(params.seed).zarr") 
    # ★ .zarr という拡張子(フォルダ名)にすると分かりやすいです
    
    # フォルダ自体は zcreate が自動で作る場合もありますが、念のため作成
    mkpath(folder_path)

    # --- パラメータ保存 (テキストはそのまま) ---
    open(joinpath(folder_path, "parameters.txt"), "w") do io
        for field in fieldnames(Parameters)
            value = getfield(params, field)
            println(io, "$field = $value")
        end
        println(io, "num_steps = $num_steps")
        println(io, "save_interval = $save_interval")
    end

    # --- データ保存 (Zarrに変更) ---
    println("Zarr形式でデータを保存中...")

    # 1. 位置情報 (Positions)
    # Python互換のため (Time, Particle, Dim) に変換
    output_pos = permutedims(positions_history, (3, 2, 1))
    
    # zcreate(型, サイズ...; path=保存先, chunks=チャンクサイズ)
    # chunks: 時間方向に少し区切ると、後で読み込む時に速いです
    zpos = zcreate(Float64, size(output_pos)..., path=joinpath(folder_path, "positions"), chunks=(100, size(output_pos, 2), 2))
    zpos[:, :, :] = output_pos

    # 2. 配向情報 (Orientations)
    # Python互換のため (Time, Particle) に変換
    output_ori = orientations_history'
    zori = zcreate(Float64, size(output_ori)..., path=joinpath(folder_path, "orientations"), chunks=(100, size(output_ori, 2)))
    zori[:, :] = output_ori
    
    println("保存完了: $folder_path")

    # メモリ解放のため nothing を返す (前回の対策)
    return nothing
end

# ----------------------------------------------------------------
# run_simulation 関数の修正
# ----------------------------------------------------------------
function run_simulation(params::Parameters, warmup::Int,  num_steps::Int; save_interval::Int=100)
    # ... (初期化処理などはそのまま) ...

    # ★ 注意: 元コードの 36行目に `saved_idx` というタイポがあったので `save_idx` に直しています
    # cargo_history[:, :, save_idx] = data.cargo_positions 
    
    # ... (ループ処理) ...

    # --- 保存パス ---
    base_path = "\\\\NAS-Ebanaru\\data\\Sasaki\\backup_git\\MTCargoSim\\data\\MTC"
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)", "seed$(params.seed).zarr")
    mkpath(folder_path)

    # --- パラメータ保存 ---
    open(joinpath(folder_path, "parameters.txt"), "w") do io
        for field in fieldnames(Parameters)
            value = getfield(params, field)
            println(io, "$field = $value")
        end
    end

    # --- データ保存 (Zarr) ---
    println("Zarr形式でデータを保存中...")

    # Positions
    output_pos = permutedims(positions_history, (3, 2, 1))
    zpos = zcreate(Float64, size(output_pos)..., path=joinpath(folder_path, "positions"), chunks=(100, size(output_pos, 2), 2))
    zpos[:, :, :] = output_pos

    # Orientations
    output_ori = orientations_history'
    zori = zcreate(Float64, size(output_ori)..., path=joinpath(folder_path, "orientations"), chunks=(100, size(output_ori, 2)))
    zori[:, :] = output_ori

    # Cargo
    output_cargo = permutedims(cargo_history, (3, 1, 2))
    zcargo = zcreate(Float64, size(output_cargo)..., path=joinpath(folder_path, "cargo"), chunks=(100, 1, 2))
    zcargo[:, :, :] = output_cargo

    println("保存完了: $folder_path")

    return nothing
end

end
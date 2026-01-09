module MTC

# --- パッケージの読み込み ---
using Zarr            # NPZの代わりにZarrを使用
using LinearAlgebra
using Distributions
using ProgressMeter
using Random

# 外部からアクセス可能な関数・型をエクスポート
export Parameters, Datas, run_simulation, MT_simulation

"""
シミュレーションの全パラメータを保持するstruct。
"""
@kwdef struct Parameters
    # --- ユーザーが指定する基本パラメータ ---
    packing_fraction::Float64           # 密度 (必須)
    A::Float64                          # 整列相互作用の強さ (必須)
    dt::Float64                  # タイムステップ
    seed::Int                           # 乱数シード
    
    # --- デフォルト値を持つ基本パラメータ ---
    cargo_radius::Float64 = 0.59        # 荷物の半径 [um]
    d_MT::Float64 = 0.025               # 微小管の直径 [um]
    r_int::Float64 = 0.1                # 微小管の相互作用半径 [um]
    box_size::Float64 = 16.0            # シミュレーションボックスのサイズ
    tau::Float64 = 1.18                 # 時間スケール [t]

    Dr_exp::Float64 = 0.0125            # 実験から得られた微小管の回転拡散 [rad/s]
    k_cargo::Float64 = 2.26e-2
    k_MT::Float64 = 9.04e-2             # 微小管の速度摩擦係数
    dna::Float64 = 0.01                 # DNAの長さ [µm]
    f::Float64 = 1.13e-2                # DNAの力 [µN]

    # --- 計算によって決まる派生パラメータ ---
    num_particles::Int
    interaction_radius::Float64
    r_a::Float64                        # 貨物と微小管の相互作用範囲
    r_dna::Float64
    dna_l::Float64
    epsilon::Float64                    # DNAのエネルギースケール [µJ]
    Dr::Float64                         # 無次元化した回転拡散係数
end

"""
Parametersオブジェクトを生成するための外部コンストラクタ関数。
"""
function Parameters(;
    packing_fraction::Float64,
    A::Float64,
    dt::Float64,
    seed::Int,
    
    # オプション引数
    cargo_radius::Float64 = 0.59,
    d_MT::Float64 = 0.025,
    r_int::Float64 = 0.1,
    box_size::Float64 = 16.0,
    tau::Float64 = 1.18,
    Dr_exp::Float64 = 0.0125,
    k_cargo::Float64 = 2.26e-2,
    k_MT::Float64 = 9.04e-2,
    dna::Float64 = 0.01,
    f::Float64 = 1.13e-2
)
    # 派生パラメータ計算
    num_particles = round(Int, (packing_fraction * box_size^2) / (pi * r_int^2) )
    interaction_radius = r_int / cargo_radius
    r_a = sqrt(2 * cargo_radius * d_MT/ (1 + d_MT/(2*cargo_radius))^2 ) / cargo_radius
    r_dna = sqrt((d_MT+2*dna)*(2*cargo_radius+2*dna))/(1+(2*dna+d_MT/2)/cargo_radius) / cargo_radius
    dna_l = dna / cargo_radius
    epsilon = sqrt(exp(1)/2) * r_a * f
    Dr = tau * Dr_exp

    return Parameters(
        packing_fraction, A, dt, seed,
        cargo_radius, d_MT, r_int, box_size,
        tau, Dr_exp, k_cargo,
        k_MT, dna, f,
        num_particles,
        interaction_radius,
        r_a, r_dna,
        dna_l, epsilon, Dr
    )
end

mutable struct Datas
    positions::Matrix{Float64}         # 2 x num_particles
    orientations::Vector{Float64}      # num_particles
    cargo_positions::Matrix{Float64}   # 1 x 2
end

# --- ヘルパー関数 ---

function dna_force(epsilon, r, r_a)
    return -2 .* epsilon .* r .* exp.(-(r.^2)./(r_a^2)) ./r_a^2 
end

function initialize(params::Parameters)
    Random.seed!(params.seed)
    
    num_particles = params.num_particles
    box_size = params.box_size

    positions = rand(2, num_particles) .* box_size
    orientations = rand(num_particles) .* 2 * π
    cargo_positions = [box_size / 2 box_size / 2]

    return Datas(positions, orientations, cargo_positions)
end

function apply_periodic_boundary!(positions::Matrix{Float64}, cargo_positions::Matrix{Float64}, box_size::Float64)
    positions .= mod.(positions, box_size)
    cargo_positions .= mod.(cargo_positions, box_size)
end

function step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    box_size = params.box_size
    r_cut = params.interaction_radius
    A = params.A
    dt = params.dt
    tau = params.tau
    Dr = params.Dr
    N = params.num_particles
    
    alignment_term = zeros(N)

    @inbounds for i in 1:N
        x_i = positions[1,i]
        y_i = positions[2,i]
        sum_sin = 0.0
        n_neighbors = 0

        @inbounds for j in 1:N
            if i == j; continue; end
            dx = x_i - positions[1,j]
            dy = y_i - positions[2,j]

            # 周期境界補正
            dx -= round(dx / box_size) * box_size
            dy -= round(dy / box_size) * box_size

            r2 = dx^2 + dy^2

            if r2 < r_cut^2
                n_neighbors += 1
                dtheta = orientations[j] - orientations[i]
                sum_sin += sin(2*dtheta)
            end
        end

        if n_neighbors > 0
            alignment_term[i] = (A / n_neighbors) * sum_sin
        end
    end

    # 向きと位置の更新
    noise = randn(N) .* sqrt(2 * Dr * tau .* dt)
    orientations .+= alignment_term .* dt .+ noise
    orientations .= mod.(orientations, 2π)

    positions[1, :] .+= cos.(orientations) .* dt
    positions[2, :] .+= sin.(orientations) .* dt
end

function transport_step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    cargo_positions = data.cargo_positions
    box_size = params.box_size
    r_cut = params.interaction_radius
    r_a = params.r_a
    A = params.A
    dt = params.dt
    tau = params.tau
    k_cargo = params.k_cargo
    k_MT = params.k_MT
    epsilon = params.epsilon
    Dr = params.Dr
    N = params.num_particles
    
    alignment_term = zeros(N)

    @inbounds for i in 1:N
        x_i = positions[1,i]
        y_i = positions[2,i]
        sum_sin = 0.0
        n_neighbors = 0

        @inbounds for j in 1:N
            if i == j; continue; end
            dx = x_i - positions[1,j]
            dy = y_i - positions[2,j]

            dx -= round(dx / box_size) * box_size
            dy -= round(dy / box_size) * box_size

            r2 = dx^2 + dy^2

            if r2 < r_cut^2
                n_neighbors += 1
                dtheta = orientations[j] - orientations[i]
                sum_sin += sin(2*dtheta)
            end
        end

        if n_neighbors > 0
            alignment_term[i] = (A / n_neighbors) * sum_sin
        end
    end

    # 貨物との相互作用
    x_cargo = cargo_positions[1]
    y_cargo = cargo_positions[2]

    force_cargo = zeros((2, N))
    dx = x_cargo .- positions[1,:]
    dy = y_cargo .- positions[2,:]

    dx -= round.(dx ./ box_size) .* box_size
    dy -= round.(dy ./ box_size) .* box_size

    r2 = dx.^2 + dy.^2
    r = sqrt.(r2)

    f = dna_force.(epsilon, r, r_a)

    force_cargo[1,:] += f .* dx ./ r
    force_cargo[2,:] += f .* dy ./ r
    
    # 更新
    noise = randn(N) .* sqrt(2 * Dr * tau .* dt)
    orientations .+= alignment_term .* dt .+ noise
    orientations .= mod.(orientations, 2π)

    positions[1, :] .+= cos.(orientations) .* dt - (tau/k_MT) .* force_cargo[1, :] .* dt
    positions[2, :] .+= sin.(orientations) .* dt - (tau/k_MT) .* force_cargo[2, :] .* dt
    cargo_positions .+= (tau/k_cargo).* reshape(sum(force_cargo, dims=2), 1, 2) .* dt
end


# --- メインシミュレーション関数 (Zarr対応版) ---

function MT_simulation(params::Parameters, num_steps::Int; save_interval::Int=100, base_path = "E:\\Sasaki\\MTCargoSim\\MT")
    data = initialize(params)

    # 保存用に配列を確保（間引いたサイズ）
    num_saved_steps = div(num_steps, save_interval)
    
    positions_history = Array{Float64, 3}(undef, 2, params.num_particles, num_saved_steps)
    orientations_history = Array{Float64, 2}(undef, params.num_particles, num_saved_steps)

    println("MTシミュレーション開始... (P=$(params.packing_fraction), A=$(params.A), Seed=$(params.seed))")
    
    save_idx = 1
    p = Progress(num_steps)

    for step in 1:num_steps
        step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size)

        if step % save_interval == 0
            if save_idx <= num_saved_steps
                positions_history[:, :, save_idx] = data.positions
                orientations_history[:, save_idx] = data.orientations
                save_idx += 1
            end
        end
        
        next!(p)
    end

    # --- Zarr保存処理 ---
    # NASパスの設定 (joinpathを使用)
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)", "seed$(params.seed).zarr")
    
    mkpath(folder_path)

    # パラメータ保存
    open(joinpath(folder_path, "parameters.txt"), "w") do io
        for field in fieldnames(Parameters)
            value = getfield(params, field)
            println(io, "$field = $value")
        end
        println(io, "num_steps = $num_steps")
        println(io, "save_interval = $save_interval")
    end

    # Zarrデータ書き込み
    # Python互換形状: (Time, Particle, Dim)
    pos_out = permutedims(positions_history, (3, 2, 1))
    zpos = zcreate(Float64, size(pos_out)..., path=joinpath(folder_path, "positions"), chunks=(100, size(pos_out, 2), 2))
    zpos[:, :, :] = pos_out

    # Python互換形状: (Time, Particle)
    ori_out = orientations_history'
    zori = zcreate(Float64, size(ori_out)..., path=joinpath(folder_path, "orientations"), chunks=(100, size(ori_out, 2)))
    zori[:, :] = ori_out
    
    println("保存完了: $folder_path")

    # メモリ解放のため何も返さない
    return nothing
end

function run_simulation(params::Parameters, warmup::Int,  num_steps::Int; save_interval::Int=100, base_path = "E:\\Sasaki\\MTCargoSim\\MT")
    data = initialize(params)
    num_saved_steps = div(num_steps, save_interval)

    positions_history = Array{Float64, 3}(undef, 2, params.num_particles, num_saved_steps)
    orientations_history = Array{Float64, 2}(undef, params.num_particles, num_saved_steps)
    cargo_history = Array{Float64, 3}(undef, 1, 2, num_saved_steps)

    println("ウォームアップ中... ($warmup steps)")
    @showprogress for step in 1:warmup
        step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size)
    end

    println("メインシミュレーション中... ($num_steps steps)")
    save_idx = 1
    p = Progress(num_steps)

    for step in 1:num_steps
        transport_step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size)

        if step % save_interval == 0
            if save_idx <= num_saved_steps
                positions_history[:, :, save_idx] = data.positions
                orientations_history[:, save_idx] = data.orientations
                cargo_history[:, :, save_idx] = data.cargo_positions # タイポ修正済み
                save_idx += 1
            end
        end
        
        next!(p)
    end

    # --- Zarr保存処理 ---
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)", "seed$(params.seed).zarr")
    
    mkpath(folder_path)

    open(joinpath(folder_path, "parameters.txt"), "w") do io
        for field in fieldnames(Parameters)
            value = getfield(params, field)
            println(io, "$field = $value")
        end
    end

    # Positions
    pos_out = permutedims(positions_history, (3, 2, 1))
    zpos = zcreate(Float64, size(pos_out)..., path=joinpath(folder_path, "positions"), chunks=(100, size(pos_out, 2), 2))
    zpos[:, :, :] = pos_out

    # Orientations
    ori_out = orientations_history'
    zori = zcreate(Float64, size(ori_out)..., path=joinpath(folder_path, "orientations"), chunks=(100, size(ori_out, 2)))
    zori[:, :] = ori_out

    # Cargo
    cargo_out = permutedims(cargo_history, (3, 1, 2))
    zcargo = zcreate(Float64, size(cargo_out)..., path=joinpath(folder_path, "cargo"), chunks=(100, 1, 2))
    zcargo[:, :, :] = cargo_out
    
    println("保存完了: $folder_path")

    # メモリ解放
    return nothing
end

end
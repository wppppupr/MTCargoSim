using NPZ
using LinearAlgebra
using Distributions
using ProgressMeter
using Random
using ArgParse

"""
シミュレーションの全パラメータを保持するstruct。
@kwdefにより、キーワード引数でインスタンスを生成できます。
"""
@kwdef struct Parameters
    # --- ユーザーが指定する基本パラメータ ---
    packing_fraction::Float64           # 密度 (必須)
    A::Float64                          # 整列相互作用の強さ (必須)
    force::Float64                      # 微小管と荷物を繋ぐ張力 (必須)
    seed::Int                           # 乱数シード
    
    # --- デフォルト値を持つ基本パラメータ ---
    cargo_radius::Float64 = 0.59        # 荷物の半径 [um]
    d_MT::Float64 = 0.025               # 微小管の直径 [um]
    r_int::Float64 = 0.1                # 微小管の相互作用半径 [um]
    box_size::Float64 = 16.0            # シミュレーションボックスのサイズ
    tau::Float64 = 1.18                 # 時間スケール [t]
    dt::Float64 = 0.1                   # タイムステップ
    noise_std::Float64 = 0.455          # ノイズの標準偏差 [rad]
    k_cargo::Float64 = 0.001
    k_MT::Float64 = 0.004               # 微小管の速度摩擦係数
    dna::Float64 = 0.01                 # DNAの長さ [µm]

    # --- 計算によって決まる派生パラメータ ---
    num_particles::Int
    interaction_radius::Float64
    r_out::Float64                      # 貨物と微小管の相互作用範囲
    dna_l::Float64
end

"""
Parametersオブジェクトを生成するための外部コンストラクタ関数。
"""
function Parameters(;
    # 必須パラメータ
    packing_fraction::Float64,
    A::Float64,
    force::Float64,
    seed::Int,

    # デフォルト値を持つパラメータ
    cargo_radius::Float64 = 0.59,        # 荷物の半径
    d_MT::Float64 = 0.025,               # 微小管の直径
    r_int::Float64 = 0.1,                # 相互作用半径
    box_size::Float64 = 16.0,            # シミュレーションボックスのサイズ
    tau::Float64 = 1.18,                 # 時間スケール
    dt::Float64 = 0.1,                   # タイムステップ
    noise_std::Float64 = 0.455,          # ノイズの標準偏差
    k_cargo::Float64 = 0.001,
    k_MT::Float64 = 0.004,               # 微小管の速度摩擦係数
    dna::Float64 = 0.01                 # DNAの長さ [µm]
)
    # 派生パラメータを計算する
    num_particles = round(Int, (packing_fraction * box_size^2) / (pi * r_int^2) )
    interaction_radius = r_int / cargo_radius
    r_out = sqrt(2 * cargo_radius * d_MT/ (1 + d_MT/(2*cargo_radius))^2 ) / cargo_radius
    dna_l = dna / cargo_radius

    # 全てのパラメータを渡して、Parametersオブジェクトを生成して返す
    # 呼び出しをキーワード引数ではなく位置引数にして、
    # この外部コンストラクタ自身への再帰呼び出しを避ける。
    return Parameters(
        packing_fraction, A, force, seed,
        cargo_radius, d_MT, r_int, box_size,
        tau, dt, noise_std, k_cargo,
        k_MT, dna,
        num_particles,
        interaction_radius,
        r_out,
        dna_l
    )
end

mutable struct Datas
    positions::Matrix{Float64}         # 2 x num_particles
    orientations::Vector{Float64}      # num_particles
    cargo_positions::Matrix{Float64}    # 1 x 2
end

function dna_force(f, r, r_out)
    if r < r_out
        return -f
    else
        return 0.0
    end
end

function initialize(params::Parameters)
    # シード値を設定
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
    cargo_positions .= mod.(cargo_positions, box_size) # ★★★ 修正点: タイポ修正 (cargo_position -> cargo_positions) ★★★
end

function step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    box_size = params.box_size
    r_cut = params.interaction_radius
    A = params.A
    dt = params.dt
    tau = params.tau

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

            # --- 周期境界補正 ---
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

    # --- 向きの更新 ---
    noise = randn(N) .* params.noise_std .* sqrt(tau)
    orientations .+= alignment_term .* dt .+ noise .* dt
    orientations .= mod.(orientations, 2π)

    # --- 位置更新 ---
    positions[1, :] .+= cos.(orientations) .* dt
    positions[2, :] .+= sin.(orientations) .* dt
end

function transport_step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    force = params.force
    cargo_positions = data.cargo_positions
    box_size = params.box_size
    r_cut = params.interaction_radius
    r_out = params.r_out
    A = params.A
    dt = params.dt
    tau = params.tau
    k_cargo = params.k_cargo
    k_MT = params.k_MT

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

            # --- 周期境界補正 ---
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

    # --- 貨物との相互作用力計算 ---
    x_cargo = cargo_positions[1]
    y_cargo = cargo_positions[2]

    force_cargo = zeros((2, N))

    dx = x_cargo .- positions[1,:]
    dy = y_cargo .- positions[2,:]

    # --- 周期境界補正 ---
    dx -= round.(dx ./ box_size) .* box_size
    dy -= round.(dy ./ box_size) .* box_size

    r2 = dx.^2 + dy.^2
    r = sqrt.(r2)

    f = dna_force.(force, r, r_out)

    force_cargo[1,:] += f .* dx ./ r
    force_cargo[2,:] += f .* dy ./ r
    
    # --- 向きの更新 ---
    noise = randn(N) .* params.noise_std .* sqrt(tau)
    orientations .+= alignment_term .* dt .+ noise .* dt
    orientations .= mod.(orientations, 2π)

    # --- 位置更新 ---
    positions[1, :] .+= cos.(orientations) .* dt - (tau/k_MT) .* force_cargo[1, :] .* dt
    positions[2, :] .+= sin.(orientations) .* dt - (tau/k_MT) .* force_cargo[2, :] .* dt
    cargo_positions .+= (tau/k_cargo).* reshape(sum(force_cargo, dims=2), 1, 2) .* dt
end

function run_simulation(params::Parameters, warmup::Int,  num_steps::Int;)
    data = initialize(params)

    # ★★★ 修正点: 履歴を保存するための配列を初期化 ★★★
    positions_history = Array{Float64, 3}(undef, 2, params.num_particles, num_steps)
    orientations_history = Array{Float64, 2}(undef, params.num_particles, num_steps)
    cargo_history = Array{Float64, 3}(undef, 1, 2, num_steps)

    println("ウォームアップステップを実行中...")
    @showprogress for step in 1:warmup
        step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size)
    end

    println("メインシミュレーションを実行中...")
    @showprogress for step in 1:num_steps
        transport_step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size)

        # ★★★ 修正点: 各ステップのデータを履歴に保存 ★★★
        positions_history[:, :, step] = data.positions
        orientations_history[:, step] = data.orientations
        cargo_history[:, :, step] = data.cargo_positions
    end

    folder_path = "data/P$(params.packing_fraction)_A$(params.A)_F$(params.force)/seed$(params.seed)"
    # ディレクトリを作成してデータを保存
    mkpath(folder_path)

    # パラメータを保存(.txt形式)
    open("$(folder_path)/parameters.txt", "w") do io
        for field in fieldnames(Parameters)
            value = getfield(params, field)
            println(io, "$field = $value")
        end
    end

    # Python (numpy)との互換性のために次元を入れ替えて保存
    npzwrite("$(folder_path)/positions_history.npy", permutedims(positions_history, (3, 2, 1)))
    npzwrite("$(folder_path)/orientations_history.npy", orientations_history')
    npzwrite("$(folder_path)/cargo_history.npy", permutedims(cargo_history, (3, 1, 2)))
    println("時系列データの保存が完了しました。")

    return data
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--packing_fraction", "-p"
            help = "Packing fraction (default: 0.5)"
            arg_type = Float64
            default = 0.5
        "--A", "-a"
            help = "Alignment interaction strength (default: 0.5)"
            arg_type = Float64
            default = 0.5
        "--force", "-f"
            help = "Force strength (default: 0.001)"
            arg_type = Float64
            default = 0.001
        "--seed", "-s"
            help = "Random seed (default: 1)"
            arg_type = Int
            default = 1
        "--warmup", "-w"
            help = "Warmup steps (default: 1000)"
            arg_type = Int
            default = 1000
        "--steps", "-n"
            help = "Number of simulation steps (default: 1000)"
            arg_type = Int
            default = 1000
    end

    return parse_args(s)
end

# シミュレーションの実行例（直接実行した場合のみ実行されます）
if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    
    println("実行パラメータ:")
    println("  packing_fraction = $(args["packing_fraction"])")
    println("  A = $(args["A"])")
    println("  force = $(args["force"])")
    println("  seed = $(args["seed"])")
    println("  warmup steps = $(args["warmup"])")
    println("  simulation steps = $(args["steps"])")
    println()

    params = Parameters(
        packing_fraction=args["packing_fraction"],
        A=args["A"],
        force=args["force"],
        seed=args["seed"]
    )
    
    final_data = run_simulation(params, args["warmup"], args["steps"])

    println("\nシミュレーション完了!")
end
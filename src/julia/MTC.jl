module MTC

# --- パッケージの読み込み ---
using Zarr
using LinearAlgebra
using Distributions
using ProgressMeter
using Random

export Parameters, Datas, run_simulation, MT_simulation

@kwdef struct Parameters
    packing_fraction::Float64
    A::Float64
    dt::Float64
    seed::Int
    cargo_radius::Float64
    omega::Float64
    
    d_MT::Float64 = 0.025
    r_int::Float64 = 0.1
    box_size::Float64 = 16.0
    v_MT::Float64 = 0.5
    warmup_dt::Float64 = 0.2
    Dr_exp::Float64 = 0.0125            
    k_cargo::Float64 = 2.26e-2
    k_MT::Float64 = 9.04e-2             
    dna::Float64 = 0.01                 
    f::Float64 = 1.13e-2

    tau::Float64 
    box_size_nd::Float64
    interaction_radius::Float64                
    num_particles::Int
    r_a::Float64                        
    r_dna::Float64
    dna_cut::Float64
    r_dna_cut::Float64
    dna_l::Float64
    epsilon::Float64                    
    Dr::Float64                         

    sigma_LJ::Float64 = 1.0
    epsilon_LJ::Float64 = 1.0
end

function Parameters(;
    packing_fraction::Float64,
    A::Float64,
    dt::Float64,
    seed::Int,
    cargo_radius::Float64,
    omega::Float64,
    d_MT::Float64 = 0.025,
    r_int::Float64 = 0.1,
    box_size::Float64 = 16.0,
    v_MT::Float64 = 0.5,
    warmup_dt::Float64 = 0.2,
    Dr_exp::Float64 = 0.0125,
    k_cargo::Float64 = 2.26e-2,
    k_MT::Float64 = 9.04e-2,
    dna::Float64 = 0.01,
    f::Float64 = 1.13e-2,
    sigma_LJ::Float64 = 1.0,
    epsilon_LJ::Float64 = 1.0
)
    tau = d_MT/v_MT
    box_size_nd = box_size/d_MT
    interaction_radius = r_int / d_MT 
    num_particles = round(Int, (packing_fraction * box_size^2) / (pi * r_int^2) )
    r_a = sqrt(2 * cargo_radius * d_MT/ (1 + d_MT/(2*cargo_radius))^2 ) / d_MT
    r_dna = sqrt((d_MT+2*dna)*(2*cargo_radius+2*dna))/(1+(2*dna+d_MT/2)/cargo_radius) / d_MT
    dna_cut = 2.0 * dna
    r_dna_cut = sqrt((d_MT+2*dna_cut)*(2*cargo_radius+2*dna_cut))/(1+(2*dna_cut+d_MT/2)/cargo_radius) / d_MT
    dna_l = dna / d_MT
    epsilon = sqrt(exp(1)/2) * r_a * f
    Dr = tau * Dr_exp

    return Parameters(
        packing_fraction, A, dt, seed, cargo_radius, omega,
        d_MT, r_int, box_size, v_MT,
        warmup_dt, Dr_exp, k_cargo,
        k_MT, dna, f, tau, box_size_nd,
        interaction_radius, num_particles,
        r_a, r_dna, dna_cut, r_dna_cut,
        dna_l, epsilon, Dr,
        sigma_LJ, epsilon_LJ
    )
end

mutable struct Datas
    positions::Matrix{Float64}         
    orientations::Vector{Float64}      
    cargo_positions::Matrix{Float64}   
    
    # 割り当て削減のためのキャッシュ
    alignment_term::Vector{Float64}
    force_cargo_x::Vector{Float64}
    force_cargo_y::Vector{Float64}
    force_MT_x::Vector{Float64}
    force_MT_y::Vector{Float64}
    noise_buffer::Vector{Float64}
    
    # Cell Listのキャッシュ
    head::Vector{Int}
    list::Vector{Int}
    
    # 計算削減のためのキャッシュ
    sin_2theta::Vector{Float64}
    cos_2theta::Vector{Float64}
    
    # 貨物との相互作用キャッシュ
    cargo_dx::Vector{Float64}
    cargo_dy::Vector{Float64}
    cargo_r_sq::Vector{Float64}
    cargo_f_mag::Vector{Float64}
end

# --- ヘルパー関数 ---

function dna_force(epsilon, r, r_a)
    return -2 * epsilon * r * exp(-(r^2)/(r_a^2)) / r_a^2 
end

# 1. リング状のガウシアンポテンシャル（ベース）
function Ring_Gaussian(r, H, r_ring, w)
    return -H * exp(-(r - r_ring)^2 / (2.0 * w^2))
end

# 2. リング状のガウシアンポテンシャルの微分（力 / ベース）
function dot_Ring_Gaussian(r, H, r_ring, w)
    return H * ((r - r_ring) / w^2) * exp(-(r - r_ring)^2 / (2.0 * w^2))
end

# 3. カットオフ（フォースシフト）を施したリング状ガウシアンポテンシャル（エネルギー）
function Ring_Gaussian_fs(r, H, r_ring, w, r_cut)
    if r <= r_cut
        # 共通部分の計算
        shift = Ring_Gaussian(r_cut, H, r_ring, w)
        dot_shift = dot_Ring_Gaussian(r_cut, H, r_ring, w)
        
        return Ring_Gaussian(r, H, r_ring, w) - shift - (r - r_cut) * dot_shift
    else
        return 0.0
    end
end

# 4. カットオフ（フォースシフト）を施したリング状ガウシアンの微分（力）
function dot_Ring_Gaussian_fs(r, H, r_ring, w, r_cut)
    return r <= r_cut ? dot_Ring_Gaussian(r, H, r_ring, w) - dot_Ring_Gaussian(r_cut, H, r_ring, w) : 0.0
end

# 5. 距離配列専用の最適化関数
function dot_Ring_Gaussian_fs_array!(out::Vector{Float64}, R_sq::Vector{Float64}, H::Float64, r_ring::Float64, w::Float64, r_cut_sq::Float64)
    r_cut = sqrt(r_cut_sq)
    dot_shift = dot_Ring_Gaussian(r_cut, H, r_ring, w)
    
    @simd for i in eachindex(R_sq)
        @inbounds if R_sq[i] <= r_cut_sq
            r = sqrt(R_sq[i])
            out[i] = dot_Ring_Gaussian(r, H, r_ring, w) - dot_shift
        else
            out[i] = 0.0
        end
    end
    return nothing
end

# --- 初期化 ---

function initialize(params::Parameters)
    Random.seed!(params.seed)
    
    num_particles = params.num_particles
    box_size_nd = params.box_size_nd

    positions = rand(2, num_particles) .* box_size_nd
    orientations = rand(num_particles) .* 2 * π
    cargo_positions = [box_size_nd / 2 box_size_nd / 2]

    # メモリアロケーションを避けるための配列初期化
    alignment_term = zeros(num_particles)
    force_cargo_x = zeros(num_particles)
    force_cargo_y = zeros(num_particles)
    force_MT_x = zeros(num_particles)
    force_MT_y = zeros(num_particles)
    noise_buffer = zeros(num_particles)
    
    r_cut = params.interaction_radius
    n_cells = max(1, floor(Int, box_size_nd / r_cut))
    head = zeros(Int, n_cells * n_cells)
    list = zeros(Int, num_particles)
    
    sin_2theta = zeros(num_particles)
    cos_2theta = zeros(num_particles)
    
    cargo_dx = zeros(num_particles)
    cargo_dy = zeros(num_particles)
    cargo_r_sq = zeros(num_particles)
    cargo_f_mag = zeros(num_particles)

    return Datas(positions, orientations, cargo_positions, 
                 alignment_term, force_cargo_x, force_cargo_y, force_MT_x, force_MT_y, noise_buffer, 
                 head, list, sin_2theta, cos_2theta,
                 cargo_dx, cargo_dy, cargo_r_sq, cargo_f_mag)
end

# In-placeでメモリアロケーションを防ぐ
function apply_periodic_boundary!(positions::Matrix{Float64}, cargo_positions::Matrix{Float64}, box_size_nd::Float64)
    @inbounds for i in axes(positions, 2)
        positions[1, i] = mod(positions[1, i], box_size_nd)
        positions[2, i] = mod(positions[2, i], box_size_nd)
    end
    cargo_positions[1] = mod(cargo_positions[1], box_size_nd)
    cargo_positions[2] = mod(cargo_positions[2], box_size_nd)
end

# --- Cell List の更新 (In-place) ---
function update_cell_list!(head::Vector{Int}, list::Vector{Int}, positions::Matrix{Float64}, box_size_nd::Float64, r_cut::Float64)
    N = size(positions, 2)
    n_cells = max(1, floor(Int, box_size_nd / r_cut))
    cell_size = box_size_nd / n_cells
    
    fill!(head, 0)
    
    @inbounds for i in 1:N
        cx = floor(Int, positions[1, i] / cell_size)
        cy = floor(Int, positions[2, i] / cell_size)
        
        cx = clamp(cx, 0, n_cells - 1)
        cy = clamp(cy, 0, n_cells - 1)
        
        cell_idx = cx + cy * n_cells + 1
        
        list[i] = head[cell_idx]
        head[cell_idx] = i
    end
    
    return n_cells, cell_size
end

# --- ステップ処理 ---

function step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    box_size_nd = params.box_size_nd
    r_cut = params.interaction_radius
    dt = params.warmup_dt
    tau = params.tau
    A = params.A * tau
    Dr = params.Dr
    N = params.num_particles

    omega = params.omega
    
    
    # LJ parameters
    sigma_LJ = params.sigma_LJ
    epsilon_LJ = params.epsilon_LJ
    r_cut_LJ = 1.122462048309373 * sigma_LJ  # 2^(1/6) * sigma_LJ
    r_cut_LJ_sq = r_cut_LJ^2
    sigma_LJ_sq = sigma_LJ^2

    k_MT = params.k_MT
    mu_MT = tau / (k_MT * params.d_MT)
    
    alignment_term = data.alignment_term
    force_MT_x = data.force_MT_x
    force_MT_y = data.force_MT_y
    fill!(force_MT_x, 0.0)
    fill!(force_MT_y, 0.0)

    sin_2theta = data.sin_2theta
    cos_2theta = data.cos_2theta
    head = data.head
    list = data.list
    noise_buffer = data.noise_buffer
    
    max_r_cut = max(r_cut, r_cut_LJ)
    n_cells, cell_size = update_cell_list!(head, list, positions, box_size_nd, max_r_cut)
    
    r_cut_sq = r_cut^2

    inv_box = 1.0 / box_size_nd

    # 三角関数の事前計算 (ベクトル計算の削減)
    @inbounds for i in 1:N
        sin_2theta[i] = sin(2 * orientations[i])
        cos_2theta[i] = cos(2 * orientations[i])
    end

    @inbounds for i in 1:N
        x_i = positions[1,i]
        y_i = positions[2,i]
        
        cx = floor(Int, x_i / cell_size)
        cy = floor(Int, y_i / cell_size)
        cx = clamp(cx, 0, n_cells - 1)
        cy = clamp(cy, 0, n_cells - 1)
        
        sum_s2 = 0.0
        sum_c2 = 0.0
        n_neighbors = 0
        
        sum_fMT_x = 0.0
        sum_fMT_y = 0.0

        # 近傍の3x3セルのみを探索
        for dcx in -1:1, dcy in -1:1
            ccx = mod(cx + dcx, n_cells)
            ccy = mod(cy + dcy, n_cells)
            cell_idx = ccx + ccy * n_cells + 1
            
            j = head[cell_idx]
            while j != 0
                if i != j
                    dx = x_i - positions[1,j]
                    dy = y_i - positions[2,j]

                    dx -= round(dx * inv_box) * box_size_nd
                    dy -= round(dy * inv_box) * box_size_nd

                    r2 = muladd(dx, dx, dy*dy)
                    if r2 < r_cut_sq
                        n_neighbors += 1
                        sum_s2 += sin_2theta[j]
                        sum_c2 += cos_2theta[j]
                    end

                    if r2 < r_cut_LJ_sq
                        s2 = sigma_LJ_sq / r2
                        s4 = s2*s2
                        s6 = s4*s2
                        s12 = s6*s6
                        f_mag_over_r = (24.0 * epsilon_LJ / r2) * (2.0 * s12 - s6)
                        sum_fMT_x += f_mag_over_r * dx
                        sum_fMT_y += f_mag_over_r * dy
                    end
                end
                j = list[j]
            end
        end

        if n_neighbors > 0
            sum_sin = sum_s2 * cos_2theta[i] - sum_c2 * sin_2theta[i]
            alignment_term[i] = (A / n_neighbors) * sum_sin
        else
            alignment_term[i] = 0.0
        end

        force_MT_x[i] = sum_fMT_x
        force_MT_y[i] = sum_fMT_y
    end

    # 向きと位置の更新 (In-place)
    noise_std = sqrt(2 * Dr * dt)
    randn!(noise_buffer)
    @inbounds for i in 1:N
        noise = noise_buffer[i] * noise_std
        orientations[i] = mod(orientations[i] + alignment_term[i] * dt + omega * dt + noise, 2π)
        positions[1, i] += cos(orientations[i]) * dt + mu_MT * force_MT_x[i] * dt
        positions[2, i] += sin(orientations[i]) * dt + mu_MT * force_MT_y[i] * dt
    end
end

function transport_step!(data::Datas, params::Parameters)
    positions = data.positions
    orientations = data.orientations
    cargo_positions = data.cargo_positions
    box_size_nd = params.box_size_nd
    r_cut = params.interaction_radius
    r_a = params.r_a
    r_dna_cut = params.r_dna_cut  # 力のカットオフとして使用
    dt = params.dt
    tau = params.tau
    A = params.A * tau
    k_cargo = params.k_cargo
    k_MT = params.k_MT
    epsilon = params.epsilon
    Dr = params.Dr
    N = params.num_particles
    omega = params.omega
    
    # LJ parameters
    sigma_LJ = params.sigma_LJ
    epsilon_LJ = params.epsilon_LJ
    r_cut_LJ = 1.122462048309373 * sigma_LJ
    r_cut_LJ_sq = r_cut_LJ^2
    sigma_LJ_sq = sigma_LJ^2

    alignment_term = data.alignment_term
    force_cargo_x = data.force_cargo_x
    force_cargo_y = data.force_cargo_y
    force_MT_x = data.force_MT_x
    force_MT_y = data.force_MT_y
    fill!(force_MT_x, 0.0)
    fill!(force_MT_y, 0.0)
    
    sin_2theta = data.sin_2theta
    cos_2theta = data.cos_2theta
    head = data.head
    list = data.list
    noise_buffer = data.noise_buffer
    
    max_r_cut = max(r_cut, r_cut_LJ)
    n_cells, cell_size = update_cell_list!(head, list, positions, box_size_nd, max_r_cut)

    r_cut_sq = r_cut^2
    r_dna_cut_sq = r_dna_cut^2

    mu_MT = tau/(k_MT * params.d_MT)
    mu_cargo = tau/(k_cargo * params.d_MT)

    inv_box = 1.0 / box_size_nd
    inv_r_a_sq = 1.0 / (r_a^2)

    # 三角関数の事前計算
    @inbounds for i in 1:N
        sin_2theta[i] = sin(2 * orientations[i])
        cos_2theta[i] = cos(2 * orientations[i])
    end

    # --- 1. 微小管同士の整列とLJ斥力相互作用 (Cell List) ---
    @inbounds for i in 1:N
        x_i = positions[1,i]
        y_i = positions[2,i]
        
        cx = floor(Int, x_i / cell_size)
        cy = floor(Int, y_i / cell_size)
        cx = clamp(cx, 0, n_cells - 1)
        cy = clamp(cy, 0, n_cells - 1)
        
        sum_s2 = 0.0
        sum_c2 = 0.0
        n_neighbors = 0
        
        sum_fMT_x = 0.0
        sum_fMT_y = 0.0
        
        for dcx in -1:1, dcy in -1:1
            ccx = mod(cx + dcx, n_cells)
            ccy = mod(cy + dcy, n_cells)
            cell_idx = ccx + ccy * n_cells + 1
            
            j = head[cell_idx]
            while j != 0
                if i != j
                    dx = x_i - positions[1,j]
                    dy = y_i - positions[2,j]

                    dx -= floor(dx * inv_box + 0.5) * box_size_nd
                    dy -= floor(dy * inv_box + 0.5) * box_size_nd

                    r2 = muladd(dx, dx, dy*dy)
                    if r2 < r_cut_sq
                        n_neighbors += 1
                        sum_s2 += sin_2theta[j]
                        sum_c2 += cos_2theta[j]
                    end
                    
                    if r2 < r_cut_LJ_sq
                        s2 = sigma_LJ_sq / r2
                        s4 = s2*s2
                        s6 = s4*s2
                        s12 = s6*s6
                        f_mag_over_r = (24.0 * epsilon_LJ / r2) * (2.0 * s12 - s6)
                        sum_fMT_x += f_mag_over_r * dx
                        sum_fMT_y += f_mag_over_r * dy
                    end
                end
                j = list[j]
            end
        end

        if n_neighbors > 0
            sum_sin = sum_s2 * cos_2theta[i] - sum_c2 * sin_2theta[i]
            alignment_term[i] = (A / n_neighbors) * sum_sin
        else
            alignment_term[i] = 0.0
        end
        
        force_MT_x[i] = sum_fMT_x
        force_MT_y[i] = sum_fMT_y
    end

    # --- 2. 貨物との相互作用 (力のカットオフ適用) ---
    x_cargo = cargo_positions[1]
    y_cargo = cargo_positions[2]

    sum_fc_x = 0.0
    sum_fc_y = 0.0

    fill!(force_cargo_x, 0.0)
    fill!(force_cargo_y, 0.0)

    cargo_dx = data.cargo_dx
    cargo_dy = data.cargo_dy
    cargo_r_sq = data.cargo_r_sq
    cargo_f_mag = data.cargo_f_mag

    @inbounds for i in 1:N
        dx = x_cargo - positions[1,i]
        dy = y_cargo - positions[2,i]

        dx -= round(dx / box_size_nd) * box_size_nd
        dy -= round(dy / box_size_nd) * box_size_nd

        cargo_dx[i] = dx
        cargo_dy[i] = dy
        cargo_r_sq[i] = dx^2 + dy^2
    end

    # 配列計算で力を一括評価 (リング状ガウシアンポテンシャルを使用)
    # 幅 w は dna_l を用いる
    w_gaussian = params.dna_l * 2
    dot_Ring_Gaussian_fs_array!(cargo_f_mag, cargo_r_sq, epsilon, r_a, w_gaussian, r_dna_cut_sq)

    @inbounds for i in 1:N
        # --- 力のカットオフ：r_dna_cut_sqの距離内でのみ加算 ---
        if cargo_r_sq[i] < r_dna_cut_sq
            # LJポテンシャル力 (cargo_f_mag は力の大きさ。距離で割ってベクトル化する)
            r_safe = max(sqrt(cargo_r_sq[i]), 1e-6)
            f_mag_over_r = cargo_f_mag[i] / r_safe
            
            fc_x = f_mag_over_r * cargo_dx[i]
            fc_y = f_mag_over_r * cargo_dy[i]
            
            force_cargo_x[i] = fc_x
            force_cargo_y[i] = fc_y
            
            sum_fc_x += fc_x
            sum_fc_y += fc_y
        end
    end
    
    # --- 3. 更新 (In-place) ---
    noise_std = sqrt(2 * Dr * dt)
    randn!(noise_buffer)
    @inbounds for i in 1:N
        noise = noise_buffer[i] * noise_std
        orientations[i] = mod(orientations[i] + alignment_term[i] * dt + omega * dt + noise, 2π)

        positions[1, i] += cos(orientations[i]) * dt - mu_MT * force_cargo_x[i] * dt + mu_MT * force_MT_x[i] * dt
        positions[2, i] += sin(orientations[i]) * dt - mu_MT * force_cargo_y[i] * dt + mu_MT * force_MT_y[i] * dt
    end
    
    cargo_positions[1] += mu_cargo * sum_fc_x * dt
    cargo_positions[2] += mu_cargo * sum_fc_y * dt
end

# --- メインシミュレーション関数 ---

function MT_simulation(params::Parameters, num_steps::Int; save_interval::Int=100, base_path = "data")
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
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size_nd)

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
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)_kMT$(params.k_MT)_kcargo$(params.k_cargo)_radius$(params.cargo_radius)", "seed$(params.seed).zarr")
    
    if isdir(folder_path)
        rm(folder_path, recursive=true, force=true)
    end
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

function run_simulation(params::Parameters, warmup::Int,  num_steps::Int; save_interval::Int=100, base_path = "data")
    data = initialize(params)
    num_saved_steps = div(num_steps, save_interval)

    positions_history = Array{Float64, 3}(undef, 2, params.num_particles, num_saved_steps)
    orientations_history = Array{Float64, 2}(undef, params.num_particles, num_saved_steps)
    cargo_history = Array{Float64, 3}(undef, 1, 2, num_saved_steps)

    println("ウォームアップ中... ($warmup steps)")
    @showprogress for step in 1:warmup
        step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size_nd)
    end

    println("メインシミュレーション中... ($num_steps steps)")
    save_idx = 1
    p = Progress(num_steps)

    for step in 1:num_steps
        transport_step!(data, params)
        apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size_nd)

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
    folder_path = joinpath(base_path, "P$(params.packing_fraction)_A$(params.A)_kMT$(params.k_MT)_kcargo$(params.k_cargo)_radius$(params.cargo_radius)", "seed$(params.seed).zarr")
    
    if isdir(folder_path)
        rm(folder_path, recursive=true, force=true)
    end
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
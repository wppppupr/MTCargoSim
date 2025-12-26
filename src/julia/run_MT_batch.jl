using Distributed
using Pkg
using Logging

"""
Robust batch runner for MTC simulations.

Behavior:
- Adds worker processes if needed
- Activates the project environment on all workers
- Includes `MTC.jl` on every worker and references `MTC.MT_simulation` and `MTC.Parameters`
- Provides simple CLI: julia -p N run_MT_batch.jl A_MIN A_MAX A_STEP MAX_SEED STEPS

Example:
  julia -p 2 src/julia/run_MT_batch.jl 0.0 0.1 0.01 10 100
"""

function setup_workers(; extra_procs::Int=0)
    if nprocs() == 1 && Sys.CPU_THREADS > 1
        addprocs(max(1, Sys.CPU_THREADS - 1 + extra_procs))
    elseif extra_procs > 0
        addprocs(extra_procs)
    end
    @info "現在のワーカー数: $(nprocs())"

    # Activate project and include module on all workers using worker-local paths
    @everywhere begin
        using Pkg
        Pkg.activate(dirname(dirname(@__DIR__)))

        # Include core module file located alongside this script on each worker
        core = joinpath(@__DIR__, "MTC.jl")
        if !isfile(core)
            error("MTC.jl not found at $core")
        end
        include(core)

        # Try to load NPZ on workers but don't fail if it's unavailable
        try
            using NPZ
        catch e
            @warn "NPZ not available on worker" exception=(e, catch_backtrace())
        end
    end
end

function run_for_a(p::Float64, a::Float64; steps::Int=300_000, max_seed::Int=1_000)
    @info "開始" packing_fraction=p A=a seeds=1:max_seed steps=steps

    # Use pmap to distribute seeds across workers; ensure module-qualified calls
    res = pmap(1:max_seed) do seed
        try
            params = MTC.Parameters(
                packing_fraction = p,
                A = a,
                seed = seed,
            )
            # run simulation (may write files)
            MTC.MT_simulation(params, steps)
            return true
        catch e
            @error "worker error" seed=seed exception=(e, catch_backtrace())
            return false
        end
    end

    n_ok = count(==(true), res)
    @info "完了" succeeded=n_ok total=length(res)
    return res
end

function parse_args(args)
    a_min = 0.0
    a_max = 0.1
    a_step = 0.01
    max_seed = 2
    steps = 30

    if length(args) >= 1
        a_min = parse(Float64, args[1])
    end
    if length(args) >= 2
        a_max = parse(Float64, args[2])
    end
    if length(args) >= 3
        a_step = parse(Float64, args[3])
    end
    if length(args) >= 4
        max_seed = parse(Int, args[4])
    end
    if length(args) >= 5
        steps = parse(Int, args[5])
    end

    return a_min, a_max, a_step, max_seed, steps
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Setup workers and environment
    setup_workers()

    # Parse CLI
    a_min, a_max, a_step, max_seed, steps = parse_args(ARGS)
    a_values = collect(range(a_min, stop=a_max, step=a_step))

    # Run experiments for each a
    for a in a_values
        @time run_for_a(0.5, a; steps=steps, max_seed=max_seed)
    end

    @info "✅ 全ての並列計算が完了しました。"
end

using Pkg
# Activate the root environment to find dependencies
Pkg.activate(joinpath(@__DIR__, "../.."))

include("MTC.jl")
using .MTC
using LinearAlgebra

println("🧪 Running LJ repulsive potential test...")

# Define test parameters
packing_fraction = 0.5
A = 0.5
dt = 0.01
seed = 1234
cargo_radius = 1.18
omega = 0

# 1. Instantiate Parameters with LJ fields
params = Parameters(
    packing_fraction = packing_fraction,
    A = A,
    dt = dt,
    seed = seed,
    cargo_radius = cargo_radius,
    omega = omega,
    sigma_LJ = 1.0,
    epsilon_LJ = 1.5
)

println("Parameters instantiated successfully!")
println("sigma_LJ = ", params.sigma_LJ)
println("epsilon_LJ = ", params.epsilon_LJ)

# 2. Initialize Datas
data = MTC.initialize(params)
println("Datas initialized successfully!")
println("Number of particles: ", params.num_particles)
println("Initial positions matrix size: ", size(data.positions))

# Check initial minimum distance
function get_min_distance(positions, box_size_nd)
    N = size(positions, 2)
    min_dist = Inf
    inv_box = 1.0 / box_size_nd
    for i in 1:N
        for j in (i+1):N
            dx = positions[1, i] - positions[1, j]
            dy = positions[2, i] - positions[2, j]
            dx -= round(dx * inv_box) * box_size_nd
            dy -= round(dy * inv_box) * box_size_nd
            d = sqrt(dx^2 + dy^2)
            if d < min_dist
                min_dist = d
            end
        end
    end
    return min_dist
end

initial_min_d = get_min_distance(data.positions, params.box_size_nd)
println("Initial minimum distance between MTs: ", initial_min_d)

# 3. Run warmup step (step!)
println("Running 10 warmup steps (step!)...")
for step in 1:10
    MTC.step!(data, params)
    MTC.apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size_nd)
end
println("Warmup steps completed. Positions size: ", size(data.positions))
warmup_min_d = get_min_distance(data.positions, params.box_size_nd)
println("Minimum distance after warmup steps: ", warmup_min_d)

# 4. Run main step (transport_step!)
println("Running 10 transport steps (transport_step!)...")
for step in 1:10
    MTC.transport_step!(data, params)
    MTC.apply_periodic_boundary!(data.positions, data.cargo_positions, params.box_size_nd)
end
println("Transport steps completed.")
final_min_d = get_min_distance(data.positions, params.box_size_nd)
println("Minimum distance after transport steps: ", final_min_d)

# 5. Check if forces were calculated (not all zero)
println("Max force_MT_x: ", maximum(abs.(data.force_MT_x)))
println("Max force_MT_y: ", maximum(abs.(data.force_MT_y)))

# Check that forces are not all zero if there are overlapping particles
if maximum(abs.(data.force_MT_x)) > 0.0 || maximum(abs.(data.force_MT_y)) > 0.0
    println("✅ LJ Forces successfully calculated and applied!")
else
    println("⚠️ Forces are zero. Checking if particles are close enough to interact...")
    r_cut_LJ = 2^(1/6) * params.sigma_LJ
    println("Cutoff radius for LJ: ", r_cut_LJ)
    if final_min_d < r_cut_LJ
        error("Particles are closer than cutoff but force is still zero!")
    else
        println("All particles are further than LJ cutoff (", r_cut_LJ, "), which is why force is zero.")
    end
end

println("🎉 Test execution completed successfully!")

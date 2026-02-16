import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# =============================================================================
# Settings
# =============================================================================
# Path to Zarr data (supports Windows path format)
TARGET_PATH = r'/Volumes/My Passport/Sasaki/MTCargoSim/MTC/P0.5_A0.5/seed*.zarr'

# Output filenames
OUTPUT_PLOT = "analysis/data/cargo_tamsd.png"
OUTPUT_DATA = "analysis/data/cargo_tamsd_seeds.zarr"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# =============================================================================
# Function Definitions
# =============================================================================

def get_box_size(zarr_path):
    """
    Reads box_size from parameters.txt
    """
    param_path = os.path.join(zarr_path, "parameters.txt")
    box_size = 16.0 # Default value

    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            for line in f:
                if "box_size" in line:
                    try:
                        box_size = float(line.split("=")[1].strip())
                        print(f"📦 Box Size detected: {box_size}")
                    except:
                        pass
    return box_size

def calculate_tamsd(trajectory, max_lag=None):
    """
    Calculates Time-Averaged Mean Squared Displacement (TAMSD).

    Parameters:
    -----------
    trajectory : np.ndarray
        Shape (Time, N_particles, Dims). Unwrapped particle positions.
    max_lag : int, optional
        Maximum lag time to calculate. Defaults to Time - 1.

    Returns:
    --------
    lags : np.ndarray
        Array of lag times (1 to max_lag).
    tamsd : np.ndarray
        Shape (Lags, N_particles). TAMSD for each particle at each lag.
    """
    num_steps, num_particles, dims = trajectory.shape

    if max_lag is None:
        max_lag = num_steps - 1

    lags = np.arange(1, max_lag + 1)
    tamsd = np.zeros((len(lags), num_particles))

    for i, lag in enumerate(lags):
        # Calculate displacement for lag tau
        # r(t + tau) - r(t)
        # Valid t range: 0 to num_steps - lag - 1
        displacements = trajectory[lag:] - trajectory[:-lag] # Shape: (Time-lag, N, Dims)

        # Squared displacement: dx^2 + dy^2
        sq_displacements = np.sum(displacements**2, axis=2) # Shape: (Time-lag, N)

        # Time average over all valid start times t
        tamsd[i] = np.mean(sq_displacements, axis=0)

    return lags, tamsd

def cargo_tamsd(zarr_path):
    """
    Loads cargo data from Zarr, unwraps trajectories, and calculates TAMSD.
    """
    # 1. Load data
    try:
        cargo = zarr.open_array(f"{zarr_path}/cargo", mode='r')
    except Exception as e:
        print(f"⚠️ Failed to open {zarr_path}/cargo: {e}")
        return None, None

    # Load into memory (Time, N_cargo, 2)
    cargo_pos = cargo[:].T

    num_steps, num_particles, dims = cargo_pos.shape
    print(f"📄 Data loaded: {num_steps} steps, {num_particles} cargo(s)")

    # 2. Get parameters (Box size L)
    L = get_box_size(zarr_path)

    # 3. Unwrap trajectories
    diff = np.diff(cargo_pos, axis=0)
    # Periodic boundary correction
    diff -= np.round(diff / L) * L

    # Accumulate to get unwrapped trajectory
    unwrapped_trajectory = np.cumsum(diff, axis=0)
    # Prepend initial position (0,0) or actual start?
    # Original cargoMSD.py used 0 as start.
    # But for TAMSD, absolute positions matter relative to each other.
    # So starting from 0 is fine as long as diffs are correct.
    unwrapped_trajectory = np.vstack([
        np.zeros((1, num_particles, dims)),
        unwrapped_trajectory
    ])

    # 4. Calculate TAMSD
    lags, tamsd = calculate_tamsd(unwrapped_trajectory)

    return lags, tamsd

def plot_tamsd(lags, tamsd_data, output_path):
    """
    Plots TAMSD data and saves to file.
    tamsd_data: List of arrays, each shape (Lags, N_particles) from different seeds.
    Assumes all seeds have same lags length for averaging.
    """
    plt.figure(figsize=(8,6))

    # If tamsd_data is a list of arrays (one per seed)
    # We want to plot the average over all particles and seeds.

    all_tamsd = []

    for seed_data in tamsd_data:
        # seed_data shape: (Lags, N_particles)
        # Average over particles for this seed
        mean_seed_tamsd = np.mean(seed_data, axis=1)
        all_tamsd.append(mean_seed_tamsd)

        # Plot individual seed average (lightly)
        plt.plot(lags, mean_seed_tamsd, alpha=0.1, color='gray')

    # Convert to array for averaging over seeds
    all_tamsd = np.array(all_tamsd) # Shape: (Num_seeds, Lags)

    # Overall average
    grand_mean_tamsd = np.mean(all_tamsd, axis=0)

    plt.plot(lags, grand_mean_tamsd, color='black', linewidth=2, label='Mean TAMSD')

    plt.xlabel("Lag Time (steps)")
    plt.ylabel("Time Averaged MSD")
    plt.title("Cargo Time Averaged MSD")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Plot saved to: {output_path}")

    # Also save log-log plot
    plt.figure(figsize=(8,6))
    for seed_data in tamsd_data:
        mean_seed_tamsd = np.mean(seed_data, axis=1)
        plt.loglog(lags, mean_seed_tamsd, alpha=0.1, color='gray')

    plt.loglog(lags, grand_mean_tamsd, color='black', linewidth=2, label='Mean TAMSD')
    plt.xlabel("Lag Time (steps)")
    plt.ylabel("Time Averaged MSD")
    plt.title("Cargo Time Averaged MSD (Log-Log)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(output_path.replace(".png", "_loglog.png"))
    plt.close()
    print(f"📊 Log-Log Plot saved to: {output_path.replace('.png', '_loglog.png')}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    try:
        print(f"🚀 Analyzing: {TARGET_PATH}")
        seeds = glob.glob(TARGET_PATH)
        if not seeds:
            print(f"⚠️ No files found at {TARGET_PATH}")

        results = []
        lags = None

        for seed in seeds:
            l, tamsd = cargo_tamsd(seed)
            if tamsd is not None:
                results.append(tamsd)
                if lags is None:
                    lags = l
                elif not np.array_equal(lags, l):
                    print(f"⚠️ Lags mismatch in {seed}")

        if results:
            # Save results
            try:
                results_array = np.array(results) # Shape: (Seeds, Lags, N_particles)
                output = zarr.open(OUTPUT_DATA, mode='w', shape=results_array.shape, dtype=results_array.dtype)
                output[:] = results_array
                print(f"💾 TAMSD data saved to {OUTPUT_DATA}")
            except Exception as e:
                print(f"⚠️ Could not save aggregated Zarr (maybe inconsistent shapes): {e}")

            if lags is not None:
                plot_tamsd(lags, results, OUTPUT_PLOT)

        else:
            print("❌ No valid data processed.")

    except Exception as e:
        print(f"❌ Error: {e}")

import os

def get_params(zarr_path):
    """parameters.txt から box_size 等を読み取る"""
    params = {
        "packing_fraction": 0.5, 
        "A": 0.5, 
        "dt": 0.01, 
        "seed": 0, 
        "cargo_radius": 0.59, 
        "d_MT": 0.025,
        "r_int": 0.1,
        "box_size": 16.0, 
        "tau": 1.18,
        "warmup_dt": 0.1,
        "Dr_exp": 0.0125,
        "k_cargo": 0.0226,
        "k_MT": 0.0904,
        "dna": 0.01,
        "f": 0.0113,
        "num_particles": 4074,
        "interaction_radius": 0.16949152542372883,
        "r_a": 0.2850716022791315,
        "r_dna": 0.3732995996344498,
        "dna_l": 0.01694915254237288,
        "epsilon": 0.0037554729946478803,
        "Dr": 0.01475
        }
    
    param_path = os.path.join(zarr_path, "parameters.txt")
    
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            for line in f:
                try:
                    key, val = line.strip().split("=")
                    if "packing_fraction" in key: params["packing_fraction"] = float(val)
                    if "A" in key: params["A"] = float(val)
                    if "dt" in key: params["dt"] = float(val)
                    if "seed" in key: params["seed"] = int(val)
                    if "cargo_radius" in key: params["cargo_radius"] = float(val)
                    if "d_MT" in key: params["d_MT"] = float(val)
                    if 'r_int' in key: params["r_int"] = float(val)
                    if "box_size" in key: params["box_size"] = int(val)
                    if "tau" in key: params["tau"] = float(val)
                    if "warmup_dt" in key: params["warmup_dt"] = float(val)
                    if "Dr_exp" in key: params['Dr_exp'] = float(val)
                    if "k_cargo" in key: params['k_cargo'] = float(val)
                    if "k_MT" in key: params['k_MT'] = float(val)
                    if "dna" in key: params['dna'] = float(val)
                    if "f" in key: params["f"] = float(val)
                    if "num_particles" in key: params["num_particles"] = int(val)
                    if "interaction_radius" in key: params['interaction_radius'] = float(val)
                    if "r_a" in key: params['r_a'] = float(val)
                    if "r_dna" in key: params['r_dna'] = float(val)
                    if "dna_l" in key: params['dna_l'] = float(val)
                    if "epsilon" in key: params['epsilon'] = float(val)
                    if "Dr" in key: params['Dr'] = float(val)
                except: pass
    return params

if __name__ == "__main__":
    path = "/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed184.zarr"
    test = get_params(path)
    print(test)
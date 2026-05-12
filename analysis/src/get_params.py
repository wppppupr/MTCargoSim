from pathlib import Path

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
    
    param_path = Path(zarr_path) / "parameters.txt"
    
    if param_path.exists():
        with open(param_path, "r") as f:
            for line in f:
                try:
                    key, val = line.strip().split("=")
                    k = key.strip()
                    v = val.strip()
                    if k == "packing_fraction": params["packing_fraction"] = float(v)
                    elif k == "A": params["A"] = float(v)
                    elif k == "dt": params["dt"] = float(v)
                    elif k == "seed": params["seed"] = int(v)
                    elif k == "cargo_radius": params["cargo_radius"] = float(v)
                    elif k == "d_MT": params["d_MT"] = float(v)
                    elif k == "r_int": params["r_int"] = float(v)
                    elif k == "box_size": params["box_size"] = float(v)
                    elif k == "box_size_nd": params["box_size_nd"] = float(v)
                    elif k == "tau": params["tau"] = float(v)
                    elif k == "warmup_dt": params["warmup_dt"] = float(v)
                    elif k == "Dr_exp": params['Dr_exp'] = float(v)
                    elif k == "k_cargo": params['k_cargo'] = float(v)
                    elif k == "k_MT": params['k_MT'] = float(v)
                    elif k == "dna": params['dna'] = float(v)
                    elif k == "f": params["f"] = float(v)
                    elif k == "num_particles": params["num_particles"] = int(v)
                    elif k == "interaction_radius": params['interaction_radius'] = float(v)
                    elif k == "r_a": params['r_a'] = float(v)
                    elif k == "r_dna": params['r_dna'] = float(v)
                    elif k == "dna_l": params['dna_l'] = float(v)
                    elif k == "epsilon": params['epsilon'] = float(v)
                    elif k == "Dr": params['Dr'] = float(v)
                except: pass
    return params

if __name__ == "__main__":
    path = "/Volumes/data/Sasaki/backup_git/MTCargoSim/data/MTC/P0.5_A0.5/seed184.zarr"
    test = get_params(path)
    print(test)
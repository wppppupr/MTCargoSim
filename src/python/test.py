import zarr
import numpy as np

test = zarr.open(r"C:\Users\0704w\MTCargoSim\src\julia\test_data\P0.5_A0.5_kMT0.0904_kcargo0.0226_radius0.5\seed1.zarr\positions", mode = "r")

print(test[0])
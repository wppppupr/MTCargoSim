import sys
import os
import shutil
import numpy as np
import zarr
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from center_order import calculate_center_polar_order

class TestCenterOrder(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_data_seed.zarr")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

        # Parameters
        self.box_size = 10.0
        self.threshold = 2.0
        self.num_steps = 2
        self.num_particles = 3

        # Create parameters.txt
        with open(self.test_dir / "parameters.txt", "w") as f:
            f.write(f"box_size={int(self.box_size)}\n")

        # Create dummy zarr data
        # positions: (2, N, Time) -> Transposed in script to (Time, N, 2)
        self.positions = np.zeros((2, self.num_particles, self.num_steps))
        # orientations: (N, Time) -> Transposed in script to (Time, N)
        self.orientations = np.zeros((self.num_particles, self.num_steps))

        # Step 0:
        # Particle 0: at center (5, 5) -> dist 0
        # Particle 1: at (6, 5) -> dist 1 (within threshold 2)
        # Particle 2: at (8, 5) -> dist 3 (outside threshold 2)

        # Center is (5, 5)
        self.positions[:, 0, 0] = [5.0, 5.0]
        self.positions[:, 1, 0] = [6.0, 5.0]
        self.positions[:, 2, 0] = [8.0, 5.0]

        # Orientations
        # Particle 0: 0
        # Particle 1: pi/2
        # Particle 2: pi
        self.orientations[0, 0] = 0
        self.orientations[1, 0] = np.pi/2
        self.orientations[2, 0] = np.pi

        # Step 1:
        # All 3 particles at center.
        self.positions[:, 0, 1] = [5.0, 5.0] # Center
        self.positions[:, 1, 1] = [5.0, 5.0] # Center
        self.positions[:, 2, 1] = [5.0, 5.0] # Center

        self.orientations[:, 1] = 0

        # Save to zarr
        zarr.save(str(self.test_dir / "positions"), self.positions)
        zarr.save(str(self.test_dir / "orientations"), self.orientations)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_calculation(self):
        polar_orders, counts = calculate_center_polar_order(str(self.test_dir), self.threshold)

        # Step 0
        # Particles 0 and 1 are within threshold.
        # Angles: 0, pi/2
        # mean_cos = (cos(0) + cos(pi/2))/2 = (1 + 0)/2 = 0.5
        # mean_sin = (sin(0) + sin(pi/2))/2 = (0 + 1)/2 = 0.5
        # P = sqrt(0.5^2 + 0.5^2) = sqrt(0.25 + 0.25) = sqrt(0.5) = 0.70710678

        self.assertEqual(len(polar_orders), 2)
        self.assertAlmostEqual(counts[0], 2)
        self.assertAlmostEqual(polar_orders[0], np.sqrt(0.5))

        # Step 1
        # All 3 particles at center.
        # P = 1. Count = 3.

        self.assertAlmostEqual(counts[1], 3)
        self.assertAlmostEqual(polar_orders[1], 1.0)

if __name__ == '__main__':
    unittest.main()

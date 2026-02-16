import unittest
import numpy as np
import sys
import os

# Add the src directory to path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cargo_time_averaged_MSD import calculate_tamsd

class TestCargoTAMSD(unittest.TestCase):

    def test_calculate_tamsd_linear(self):
        """Test with linear motion: x(t) = v * t"""
        num_steps = 100
        num_particles = 1
        dims = 2

        v = np.array([1.0, 0.0]) # Velocity (1, 0)

        # Create trajectory
        # Shape: (Time, N, Dims)
        trajectory = np.zeros((num_steps, num_particles, dims))
        for t in range(num_steps):
            trajectory[t, 0, :] = v * t

        lags, tamsd = calculate_tamsd(trajectory)

        # Expected MSD = |v * tau|^2 = v^2 * tau^2
        # v^2 = 1^2 + 0^2 = 1
        # MSD = tau^2

        expected_msd = lags**2

        # Check if calculated TAMSD matches expected
        # tamsd shape is (Lags, N)
        np.testing.assert_allclose(tamsd[:, 0], expected_msd, rtol=1e-5)

    def test_calculate_tamsd_stationary(self):
        """Test with stationary particle"""
        num_steps = 50
        num_particles = 1
        dims = 2

        trajectory = np.ones((num_steps, num_particles, dims)) * 5.0

        lags, tamsd = calculate_tamsd(trajectory)

        expected_msd = np.zeros_like(lags)
        np.testing.assert_allclose(tamsd[:, 0], expected_msd, atol=1e-10)

    def test_calculate_tamsd_dimensions(self):
        """Test output dimensions"""
        num_steps = 20
        num_particles = 3
        dims = 2

        trajectory = np.random.rand(num_steps, num_particles, dims)

        lags, tamsd = calculate_tamsd(trajectory)

        self.assertEqual(len(lags), num_steps - 1)
        self.assertEqual(tamsd.shape, (num_steps - 1, num_particles))

    def test_calculate_tamsd_max_lag(self):
        """Test with custom max_lag"""
        num_steps = 20
        num_particles = 1
        dims = 2
        trajectory = np.random.rand(num_steps, num_particles, dims)

        max_lag = 5
        lags, tamsd = calculate_tamsd(trajectory, max_lag=max_lag)

        self.assertEqual(len(lags), max_lag)
        self.assertEqual(tamsd.shape, (max_lag, num_particles))
        self.assertEqual(lags[-1], max_lag)

if __name__ == "__main__":
    unittest.main()

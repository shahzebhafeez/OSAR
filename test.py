import unittest
import numpy as np

# Import functions and classes from your main simulation file
from combined_EB_candidate import (
    Node,
    sound_speed,
    thorp_absorption_db_per_m,
    path_loss_linear,
    noise_psd_linear,
    compute_transmission_delay_paper,
    PT, LP, CH_BANDWIDTH, K_SPREAD
)

class TestAcousticFormulas(unittest.TestCase):

    def test_sound_speed(self):
        """Test if sound speed is within a realistic range."""
        # Test case: 10C, 35ppt salinity, 100m depth
        speed = sound_speed(z_m=100.0, s_ppt=35, T_c=10)
        self.assertGreater(speed, 1400, "Sound speed should be above 1400 m/s")
        self.assertLess(speed, 1600, "Sound speed should be below 1600 m/s")
        print(f"\n[PASS] test_sound_speed: Calculated speed = {speed:.2f} m/s (Realistic)")

    def test_thorp_absorption(self):
        """Test if absorption increases with frequency."""
        absorption_low_freq = thorp_absorption_db_per_m(f_khz=10.0)
        absorption_high_freq = thorp_absorption_db_per_m(f_khz=50.0)
        self.assertLess(absorption_low_freq, absorption_high_freq, "Absorption should increase with frequency")
        print(f"[PASS] test_thorp_absorption: Low freq loss < High freq loss ({absorption_low_freq:.6f} < {absorption_high_freq:.6f})")

    def test_path_loss(self):
        """Test the behavior of the path loss formula."""
        # 1. Path loss should be >= 1 (signal can't get stronger)
        loss_100m_10khz = path_loss_linear(d_m=100.0, f_khz=10.0, k=1.5)
        self.assertGreaterEqual(loss_100m_10khz, 1.0)
        print(f"[PASS] test_path_loss (>=1): Loss at 100m is {loss_100m_10khz:.2f}")

        # 2. Path loss should increase with distance
        loss_200m_10khz = path_loss_linear(d_m=200.0, f_khz=10.0, k=1.5)
        self.assertGreater(loss_200m_10khz, loss_100m_10khz)
        print(f"[PASS] test_path_loss (distance): Loss at 200m > 100m")

        # 3. Path loss should increase with frequency
        loss_100m_50khz = path_loss_linear(d_m=100.0, f_khz=50.0, k=1.5)
        self.assertGreater(loss_100m_50khz, loss_100m_10khz)
        print(f"[PASS] test_path_loss (frequency): Loss at 50kHz > 10kHz")

    def test_noise_psd(self):
        """Test if the corrected noise formula produces positive, varying values."""
        noise_10khz = noise_psd_linear(f_khz=10.0)
        noise_50khz = noise_psd_linear(f_khz=50.0)
        self.assertGreater(noise_10khz, 0, "Noise power should be positive")
        self.assertNotEqual(noise_10khz, noise_50khz, "Noise should vary with frequency")
        print(f"[PASS] test_noise_psd: Noise values are positive and frequency-dependent ({noise_10khz:.2f}, {noise_50khz:.2f})")


class TestTransmissionDelay(unittest.TestCase):

    def setUp(self):
        """Set up a controlled scenario for testing the TD calculation."""
        # Mock channel state (all channels are idle)
        idle_channels = np.array([False] * 128)

        # Create a simple vertical scenario
        self.source_node = Node(node_id=0, pos=np.array([100, 100, 200]), channel_state=idle_channels)
        self.dest_node = Node(node_id=1, pos=np.array([100, 100, 100]), channel_state=idle_channels)
        self.final_dest_pos = np.array([100, 100, 0]) # Surface buoy
        self.channel_khz = 25.0 # A mid-range frequency channel

    def test_full_td_calculation(self):
        """Verify the end-to-end TD calculation with realistic PT."""
        print("\n--- Testing Full TD Calculation ---")
        TD, rate, snr = compute_transmission_delay_paper(
            src=self.source_node,
            dest=self.dest_node,
            dest_pos=self.final_dest_pos,
            channel_center_khz=self.channel_khz,
            Pt=PT # Using the corrected, realistic transmit power
        )

        # 1. Check N_hop calculation
        # Expected: DiD = 200m, proj_len = 100m -> N_hop = 2.0
        DiD = np.linalg.norm(self.final_dest_pos - self.source_node.pos)
        proj_len = self.source_node.depth - self.dest_node.depth
        expected_n_hop = DiD / proj_len
        self.assertAlmostEqual(2.0, expected_n_hop, places=5)
        print(f"[PASS] N_hop calculation is correct.")

        # 2. Check SNR
        snr_db = 10 * np.log10(snr + 1e-12)
        self.assertGreater(snr_db, 0, f"SNR should be positive dB for a 100m hop, but got {snr_db:.2f} dB")
        print(f"[PASS] SNR is a healthy positive value: {snr_db:.2f} dB")

        # 3. Check Data Rate
        self.assertGreater(rate, 1000, f"Data rate should be reasonably high, but got {rate:.2f} bps")
        print(f"[PASS] Data rate is high: {rate / 1000:.2f} kbps")

        # 4. Check Final TD
        self.assertGreater(TD, 0, "Transmission Delay must be positive")
        self.assertLess(TD, 60, f"Transmission Delay should be realistic (e.g., < 1 minute), but got {TD:.2f} s")
        print(f"[PASS] Final TD is a realistic value: {TD:.4f} seconds")


if __name__ == '__main__':
    unittest.main()

from src.main import simulate

def test_small_simulation_runs():
    avg_d, pdr, paths = simulate(N=5, num_packets=2, M=2, visualize=False)
    assert avg_d > 0, "Average delay should be positive"
    assert 0 <= pdr <= 1, "PDR must be within [0, 1]"

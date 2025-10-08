import pytest
from src.channel_model import path_loss, absorption_coeff

def test_absorption_increases_with_freq():
    f1, f2 = 10, 50
    a1, a2 = absorption_coeff(f1), absorption_coeff(f2)
    assert a2 > a1, "Absorption must increase with frequency"

def test_path_loss_increases_with_distance():
    f = 20
    d1, d2 = 100, 200
    pl1, pl2 = path_loss(d1, f), path_loss(d2, f)
    assert pl2 > pl1, "Path loss must increase with distance"

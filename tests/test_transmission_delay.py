from src.transmission_delay import transmission_delay
from src.utils import UnderwaterNode

def test_zero_rate_handling():
    n1 = UnderwaterNode(pos=[0,0,100], depth=100)
    n2 = UnderwaterNode(pos=[0,0,50], depth=50)
    delay = transmission_delay(n1, n2, ch=1, dest=n2)
    assert delay != float('inf'), "Should handle zero division gracefully"

def test_downward_flow_invalid():
    n1 = UnderwaterNode(pos=[0,0,50], depth=50)
    n2 = UnderwaterNode(pos=[0,0,100], depth=100)
    delay = transmission_delay(n1, n2, ch=1, dest=n2)
    assert delay == float('inf'), "Downward transmission should be invalid"

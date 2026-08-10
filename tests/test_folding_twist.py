import numpy as np
from reality_stone.clarus.folding_twist import _peaks


def test_periodic_peak_detection() -> None:
    field = np.array([2.0, 0.0, -1.0, 0.0])
    peaks = _peaks(field)
    assert peaks.tolist() == [True, False, False, False]

import numpy as np
from reality_stone.clarus.fsaverage_geometry import _ridge_predict


def test_ridge_recovers_linear_signal() -> None:
    x = np.arange(20, dtype=float)[:, None]
    y = 2.0 * x[:, 0] + 1.0
    prediction = _ridge_predict(x, y, x, 1e-8)
    assert np.max(np.abs(prediction - y)) < 1e-6

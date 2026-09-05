"""재귀 Gaussian 적분의 해석적 정규화와 잘못된 조립 입력을 검산한다."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("recursive_kernel_under_test", HERE / "recursive_kernel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scalar_marginal_has_exact_precision_and_normalization():
    kernel, log_factor = module.eliminate(np.array([[2., 1.], [1., 2.]]), 1)
    np.testing.assert_allclose(kernel, [[1.5]])
    assert log_factor == pytest.approx(0.5 * math.log(math.pi))


def test_three_variable_chain_preserves_integral_in_two_steps():
    matrix = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
    direct, weight = module.eliminate(matrix, 1)
    middle, first = module.eliminate(matrix, 2)
    staged, second = module.eliminate(middle, 1)
    np.testing.assert_allclose(direct, [[4 / 3]])
    np.testing.assert_allclose(staged, direct)
    assert weight == pytest.approx(math.log(2 * math.pi) - 0.5 * math.log(3))
    assert first + second == pytest.approx(weight)


def test_assembly_rejects_missing_cell_kernel():
    with pytest.raises(ValueError, match="exactly one"):
        module.assemble([(0, 1, 2, 3, 4)], [], [])

@pytest.mark.parametrize("matrix", [
    np.ones((2, 3)),
    np.array([[2., 100.], [1., 2.]]),
    np.array([[1., 2.], [2., 1.]]),
    np.diag([1., np.nan]),
    np.diag([1., np.inf]),
])
def test_elimination_rejects_invalid_gaussian_precision(matrix):
    with pytest.raises(ValueError, match="precision"):
        module.eliminate(matrix, 1)


@pytest.mark.parametrize("retained", [-1, 3, True, 1.5])
def test_elimination_rejects_invalid_partition(retained):
    with pytest.raises(ValueError, match="retained"):
        module.eliminate(np.eye(2), retained)


def test_empty_and_full_marginals_keep_exact_integral():
    matrix = np.array([[2., 1.], [1., 2.]])
    empty, total_weight = module.eliminate(matrix, 0)
    assert empty.shape == (0, 0)
    assert total_weight == pytest.approx(math.log(2 * math.pi) - 0.5 * math.log(3))
    unchanged, no_weight = module.eliminate(matrix, 2)
    np.testing.assert_array_equal(unchanged, matrix)
    assert no_weight == 0


@pytest.mark.parametrize("asymmetric", [True, False])
def test_assembly_rejects_invalid_cell_precision(asymmetric):
    kernel = np.eye(10)
    if asymmetric:
        kernel[0, 1] = 100.
    else:
        kernel[0, 0] = -1.
    with pytest.raises(ValueError, match="precision"):
        module.assemble([(0, 1, 2, 3, 4)], [kernel], [])


@pytest.mark.parametrize("depth,step", [
    (0, 2e-5), (4, 2e-5), (True, 2e-5), (1.5, 2e-5),
    (1, 0.), (1, float("nan")), (1, float("inf")),
])
def test_invalid_run_is_rejected_before_computation(monkeypatch, depth, step):
    monkeypatch.setattr(module, "depth_result", lambda *args: pytest.fail("invalid inputs reached computation"))
    with pytest.raises(ValueError):
        module.run(depth, step)


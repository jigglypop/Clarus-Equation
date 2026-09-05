"""독립 양자 사면체의 형상 일치 에너지와 보존 방출을 검사한다.

SU(2) 사면체 공간, 쌍대 기저, 형상 에너지 M와 초기 진공 사슬은 공급 공리다.
전체 레게 접착, 단일 고전 형상, 0D 시간 발생을 유도하지 않는다.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import eye, kron, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.special import j1

import coherent_tetrahedron_overlap as tetra

HERE = Path(__file__).resolve().parent


def operators(two_j):
    """동일 면적의 네 스핀을 불변 결합한 실제 재결합 기저."""
    if isinstance(two_j, bool) or not isinstance(two_j, int) or two_j < 1:
        raise ValueError("두 배 스핀은 양의 정수여야 한다")
    j = two_j / 2
    casimir = j * (j + 1)
    k = np.arange(two_j + 1, dtype=float)
    first = np.diag((k * (k + 1) - 2 * casimir) / (2 * casimir))
    off = -k[1:] * ((two_j + 1)**2 - k[1:]**2)
    off /= 4 * casimir * np.sqrt(4 * k[1:]**2 - 1)
    second = np.diag(-k * (k + 1) / (4 * casimir))
    second += np.diag(off, 1) + np.diag(off, -1)
    return first, second


def difference(operator, dual=True):
    identity = np.eye(len(operator))
    return np.kron(operator, identity) - np.kron(
        identity, operator.T if dual else operator)


@lru_cache(maxsize=8)
def model(two_j):
    first, second = operators(two_j)
    d = len(first)
    da, db = difference(first), difference(second)
    master = da @ da + db @ db
    eigenvalues, basis = np.linalg.eigh(master)
    omega = np.eye(d).reshape(-1) / math.sqrt(d)
    # 수치 기저의 위상은 고정하지 않고 해석적 영벡터와의 차이를 검사한다.
    return first, second, master, eigenvalues, basis, omega


def spectral_case(two_j):
    first, second, master, energies, basis, omega = model(two_j)
    d = len(first)
    j = two_j / 2
    casimir = j * (j + 1)
    volume = -1j * (first @ second - second @ first)
    dc = difference(volume)
    wrong = master + difference(volume, dual=False) @ difference(volume, dual=False)
    reduced = omega.reshape(d, d) @ omega.reshape(d, d).conj().T
    mean = float(np.trace(reduced @ first).real)
    variance = float(np.trace(reduced @ first @ first).real - mean**2)
    if two_j <= 4:
        raw_a, raw_b, raw_v, closure = tetra.shape_operators(two_j)
        tensor_error = max(np.linalg.norm(first - raw_a / casimir),
                           np.linalg.norm(second - raw_b / casimir),
                           np.linalg.norm(volume - raw_v / casimir**2))
    else:
        tensor_error, closure = None, None
    return {
        "two_j": two_j, "dimension": d, "tensor_error": tensor_error,
        "gauss_closure_error": closure,
        "ground_energy": float(energies[0]), "gap": float(energies[1]),
        "kernel_residual": float(np.linalg.norm(master @ omega)),
        "ground_overlap": float(abs(np.vdot(omega, basis[:, 0]))**2),
        "zero_count": int(np.count_nonzero(abs(energies) < 1e-10)),
        "dual_volume_residual": float(np.linalg.norm(dc @ omega)),
        "wrong_orientation_ground": float(np.linalg.eigvalsh(wrong)[0]),
        "marginal_error": float(np.linalg.norm(reduced - np.eye(d) / d)),
        "mean_shape": mean, "variance_shape": variance,
        "variance_formula": 16 / 45 + 1 / (15 * casimir),
        "minimum_neighbor": float(min(abs(np.diag(second, 1)))),
        "spectrum": energies.tolist(),
    }


def sparse_gap(two_j):
    first, second = operators(two_j)
    identity = eye(len(first), format="csr")
    differences = [kron(csr_matrix(o), identity) -
                   kron(identity, csr_matrix(o.T)) for o in (first, second)]
    master = sum(x @ x for x in differences).tocsr()
    values, vectors = eigsh(master, k=2, which="SM", tol=1e-11,
                           v0=np.linspace(.7, 1.3, master.shape[0]), maxiter=30000)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    residual = max(np.linalg.norm(master @ vectors[:, n] - values[n] * vectors[:, n])
                   for n in range(2))
    j = two_j / 2
    return {"two_j": two_j, "gap": float(values[1]),
            "gap_times_j_jplus1": float(values[1] * j * (j + 1)),
            "eigen_residual": float(residual)}


def survival(tau):
    tau = float(tau)
    if not math.isfinite(tau):
        raise ValueError("진화 매개변수는 유한해야 한다")
    return 1. if tau == 0 else float(j1(2 * tau) / tau)


@lru_cache(maxsize=4)
def chain_basis(size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("사슬 절단은 2 이상의 정수여야 한다")
    return eigh_tridiagonal(np.zeros(size), np.ones(size - 1))


def chain_state(size, tau):
    values, vectors = chain_basis(size)
    return vectors @ (vectors[0, :] * np.exp(-1j * values * tau))


def geometry_input(two_j):
    left = tetra.shape_state(two_j, tetra.REGULAR + [.12, .17])
    right = tetra.shape_state(two_j, tetra.REGULAR + [-.08, -.11])
    return np.kron(left, right.conj())


def cooling_case(two_j, tau, size=128, initial="geometry"):
    first, second, master, energies, basis, omega = model(two_j)
    dim = len(master)
    state = geometry_input(two_j) if initial == "geometry" else (
        math.sqrt(.3) * basis[:, 0] + math.sqrt(.7) * basis[:, 1])
    state = state / np.linalg.norm(state)
    coefficients = basis.conj().T @ state
    eps = np.maximum(energies[1:], 0.)
    g = energies[1] / 4  # 공급 규약이며 실제 미시 결합의 유도는 아니다.
    parameter = tau / g
    wave = chain_state(size, tau)
    amplitudes = coefficients[1:] * np.exp(-1j * eps * parameter)
    root = coefficients[0] * basis[:, 0] + basis[:, 1:] @ (wave[0] * amplitudes)
    bath = (amplitudes[:, None] * wave[None, 1:]).reshape(-1)
    joint = np.column_stack((root, np.outer(basis[:, 0], bath)))
    reduced = joint @ joint.conj().T
    target = np.outer(omega, omega)
    distance = .5 * np.sum(abs(np.linalg.eigvalsh(reduced - target)))
    initial_energy = float(np.vdot(state, master @ state).real)
    system_energy = float(np.trace(reduced @ master).real)
    bath_energy = float(np.dot(abs(coefficients[1:])**2, eps) *
                        np.sum(abs(wave[1:])**2))
    interaction = float(2 * g * np.sum(abs(coefficients[1:])**2) *
                        (wave[0].conjugate() * wave[1]).real)
    bath_hopping = float(2 * g * np.sum(abs(coefficients[1:])**2) *
                        np.vdot(wave[1:-1], wave[2:]).real)
    p = float(np.sum(abs(coefficients[1:])**2))
    root_probability = float(abs(wave[0])**2)
    expected_distance = math.sqrt(p*p*root_probability**2 +
                                  p*(1-p)*root_probability)
    predicted_energy = initial_energy * survival(tau)**2
    return {
        "two_j": two_j, "tau": tau, "size": size, "initial": initial,
        "gap": float(energies[1]), "hopping_g": float(g),
        "positive_energy_lower_bound": float(min(eps) - 2*g),
        "initial_excited_probability": p, "initial_energy": initial_energy,
        "system_energy": system_energy, "bath_energy": bath_energy,
        "interaction_energy": interaction, "bath_hopping_energy": bath_hopping,
        "total_energy_error": abs(system_energy + bath_energy +
                                  interaction + bath_hopping - initial_energy),
        "norm_error": float(abs(np.trace(reduced) - 1)),
        "root_survival": root_probability,
        "infinite_survival": survival(tau)**2,
        "survival_error": abs(root_probability - survival(tau)**2),
        "energy_law_error": abs(system_energy - predicted_energy),
        "trace_distance": float(distance),
        "trace_distance_formula": expected_distance,
        "trace_distance_error": abs(distance - expected_distance),
        "marginal_shape_variance": float(
            np.trace(reduced @ np.kron(first @ first, np.eye(len(first)))).real -
            np.trace(reduced @ np.kron(first, np.eye(len(first)))).real**2),
    }


def information_control(two_j=1, tau=8., size=64):
    """외부 참조계의 주변상태와 서로 다른 입력의 내적을 직접 보존한다."""
    _, _, _, energies, basis, _ = model(two_j)
    dim = len(energies)
    g = energies[1] / 4
    wave = chain_state(size, tau)
    envdim = 1 + (dim - 1) * (size - 1)
    transfer = np.zeros((dim, envdim, dim), dtype=complex)
    transfer[:, 0, 0] = basis[:, 0]
    wrong = np.zeros((dim, size, dim), dtype=complex)
    wrong[:, 0, 0] = basis[:, 0]
    for a in range(1, dim):
        phase = np.exp(-1j * energies[a] * tau / g)
        transfer[:, 0, a] = phase * wave[0] * basis[:, a]
        start = 1 + (a - 1) * (size - 1)
        tail = phase * np.outer(basis[:, 0], wave[1:])
        transfer[:, start:start + size - 1, a] = tail
        wrong[:, 0, a] = phase * wave[0] * basis[:, a]
        # 서로 다른 입력 표지를 지운 대조는 내적을 보존하지 못한다.
        wrong[:, 1:, a] = tail
    matrix = transfer.reshape(-1, dim)
    gram = matrix.conj().T @ matrix
    reference = gram.T / dim
    local = np.einsum("sek,tek->st", transfer, transfer.conj()) / dim
    overlap = float(np.vdot(basis[:, 0], local @ basis[:, 0]).real)
    wrong_matrix = wrong.reshape(-1, dim)
    return {
        "two_j": two_j, "tau": tau, "initial_reference_dimension": dim,
        "isometry_error": float(np.linalg.norm(gram - np.eye(dim))),
        "reference_marginal_error": float(np.linalg.norm(reference - np.eye(dim)/dim)),
        "local_excited_probability": 1-overlap,
        "predicted_excited_probability": (1-1/dim)*abs(wave[0])**2,
        "deleting_bath_retained_norm": float(np.linalg.norm(transfer[:, 0, :])**2/dim),
        "erasing_channel_label_defect": float(np.linalg.norm(
            wrong_matrix.conj().T @ wrong_matrix - np.eye(dim))),
    }


def shared_copy_control(two_j):
    """가운데 형상 공간을 두 번 쓰는 완전 일치의 양립성을 검사한다."""
    first, second, master, energies, _, omega = model(two_j)
    d = len(first)
    identity = np.eye(d)
    pair_projector = np.outer(omega, omega)
    p12 = np.kron(pair_projector, identity)
    p23 = np.kron(identity, pair_projector)
    # 현재 A,B는 실수 대칭이다. 가운데 공간을 쌍대로 써도 이 두 행렬은 같다.
    total = np.kron(master, identity) + np.kron(identity, master)
    values = np.linalg.eigvalsh(total)
    gap = float(energies[1])
    return {
        "two_j": two_j, "dimension": d,
        "ground_energy": float(values[0]),
        "lower_bound": gap * (1-1/d),
        "projector_overlap": float(np.linalg.norm(p12 @ p23, 2)),
        "projector_sum_norm": float(np.linalg.eigvalsh(p12+p23)[-1]),
        "operator_bound_residual": float(np.linalg.eigvalsh(
            total-gap*(2*np.eye(d**3)-p12-p23))[0]),
        "zero_count": int(np.count_nonzero(abs(values) < 1e-10)),
    }


def isolated_control(two_j=2):
    _, _, master, energies, basis, omega = model(two_j)
    state = geometry_input(two_j)
    coefficients = basis.conj().T @ state
    initial_energy = float(np.vdot(state, master @ state).real)
    initial_ground = float(abs(np.vdot(omega, state))**2)
    rows = []
    for parameter in (0., .7, 3., 20.):
        out = basis @ (np.exp(-1j * energies * parameter) * coefficients)
        rows.append({"parameter": parameter,
                     "energy_error": abs(float(np.vdot(out, master @ out).real) - initial_energy),
                     "ground_probability_error": abs(float(abs(np.vdot(omega, out))**2) - initial_ground)})
    return {"initial_energy": initial_energy, "rows": rows}


def recurrence_control():
    rows = []
    for tau in (0., 2., 6., 12., 20.):
        rows.append({"tau": tau, "finite_size": 8,
                     "finite_survival": float(abs(chain_state(8, tau)[0])**2),
                     "infinite_survival": survival(tau)**2})
    return rows


def run():
    spectra = [spectral_case(n) for n in (1, 2, 3, 4)]
    return {
        "status": "조건부 형상 일치와 보존 방출; 전체 공통 계량은 미완성",
        "assumptions": ["독립 SU(2) 사면체·동일 면적·쌍대 식별·M를 공급",
                        "반무한 초기 진공 환경·한 들뜸 직합·결합 g를 공급",
                        "공급된 진화 매개변수이며 0D 물리 시간 발생 아님"],
        "spectra": spectra,
        "gap_exploration": [sparse_gap(n) for n in (4, 8, 16, 32)],
        "cooling": [cooling_case(n, tau) for n in (1, 2, 4)
                    for tau in (0., 1., 3., 8., 20.)],
        "superposition": [cooling_case(1, tau, initial="superposition") for tau in (1., 3., 8.)],
        "cutoff": [{"two_j": n, "tau": tau,
                    "difference": abs(cooling_case(n, tau, 128)["system_energy"] -
                                      cooling_case(n, tau, 256)["system_energy"])}
                   for n in (1, 4) for tau in (8., 20.)],
        "isolated": isolated_control(),
        "information": [information_control(n) for n in (1, 2)],
        "shared_copy": [shared_copy_control(n) for n in (1, 2, 3, 4)],
        "recurrence": recurrence_control(),
        "dependencies": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                         for name in ("tetrahedron_matching_flow.py", "coherent_tetrahedron_overlap.py")},
        "sources": ["https://arxiv.org/abs/1805.05856",
                    "https://arxiv.org/abs/0705.0674",
                    "https://arxiv.org/abs/quant-ph/0611164"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

"""주어진 가우시안 분할의 시간 독립 두 부문 실현과 에너지 장부.

[q,p]=i, 에너지 단위 hbar*omega, tau=omega*t, coupling=g/(hbar*omega).
기존 S의 Fock 유니터리를 U라 두고 V psi=U(psi tensor vacuum^(k-1)).
HP=N+1/2, HC=U(sum N+k/2)U†이면 HC V=V(HP+delta), delta=(k-1)/2.
배터리 HB=gap |1><1|과 W:|P,1,psi> -> |C,0,Vpsi>를 공급한다.
H=Hfree+coupling*(W+W†)는 시간 독립이다. gap=delta이면 자유 에너지도
보존하며 tau=pi/(2*coupling)에서 완전 분할, 두 배 시간에서 역병합한다.

child 기저는 U|n0,...,nk-1>이다. 총 점유 제한은 Hfree,W,W†의 정확한
불변 부분공간을 고르는 것이며 원래 자유 Fock 기저의 가우시안 절단이 아니다.
독립 좌표 검산은 |det M|^-1/2 psi_n((M^-1 q)_0) prod psi_0의 Hermite
미분을 실제 전달된 이차 미분 연산자에 넣는다.

모드 수에 따른 절대 바닥 에너지, U, 배터리, 비국소 등거리 결합은 공급
입력이다. 완전 분할 시 배터리는 소모되고, 역병합 시 함께 복귀한다.
왕복에는 순생성물이나 기록이 없다. 임의 자식 상태의 병합, 불가역 선택,
자율 트리 전체, CE 국소 작용, 환경 결합 또는 공통 계량은 유도하지 않는다.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import split_energy_transport as transport

source = transport.source


def _occupation_basis(k, maximum):
    source.child_count(k)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 12:
        raise ValueError("maximum occupation must be an integer from 0 through 12")
    if k > 8 or 2*(maximum+1+math.comb(maximum+k, k)) > 256:
        raise ValueError("finite invariant witness is limited to 256 basis vectors and 8 children")

    def visit(remaining_modes, budget):
        if remaining_modes == 1:
            yield from ((value,) for value in range(budget + 1))
        else:
            for value in range(budget + 1):
                for tail in visit(remaining_modes - 1, budget - value):
                    yield (value,) + tail
    return list(visit(k, maximum))


def model(k, maximum=3, coupling=.25, battery_gap=None):
    """부모·자식 전체와 배터리 양 상태를 포함한 정확한 불변 블록."""
    occupations = _occupation_basis(k, maximum)
    coupling = float(coupling)
    gap = (k-1)/2 if battery_gap is None else float(battery_gap)
    if not math.isfinite(coupling) or coupling <= 0 or not math.isfinite(gap) or gap < 0:
        raise ValueError("positive finite coupling and nonnegative finite battery gap required")
    parent_count = maximum + 1
    child_start = 2*parent_count
    dimension = child_start + 2*len(occupations)
    oscillator = np.empty(dimension)
    oscillator[:child_start] = np.repeat(np.arange(parent_count) + .5, 2)
    oscillator[child_start:] = np.repeat([sum(n) + k/2 for n in occupations], 2)
    battery = np.tile([0., gap], parent_count + len(occupations))
    child_lookup = {n: i for i, n in enumerate(occupations)}
    parent_indices = 2*np.arange(parent_count) + 1
    child_indices = np.array([child_start + 2*child_lookup[(n,) + (0,)*(k-1)]
                              for n in range(parent_count)])
    w = np.zeros((dimension, dimension))
    w[child_indices, parent_indices] = 1.
    free = np.diag(oscillator + battery)
    interaction = coupling*(w+w.T)
    return {"branching": k, "maximum": maximum, "coupling": coupling, "battery_gap": gap,
            "delta": (k-1)/2, "occupations": occupations, "child_start": child_start,
            "parent_indices": parent_indices, "child_indices": child_indices,
            "oscillator": oscillator, "battery": battery, "w": w, "free": free,
            "interaction": interaction, "hamiltonian": free + interaction}


def propagator(data, tau):
    tau = float(tau)
    if not math.isfinite(tau):
        raise ValueError("finite dimensionless time required")
    energies, vectors = np.linalg.eigh(data["hamiltonian"])
    return (vectors*np.exp(-1j*energies*tau)) @ vectors.T


def transfer_law(k, coupling, battery_gap, tau):
    source.child_count(k)
    coupling, gap, tau = float(coupling), float(battery_gap), float(tau)
    if not all(math.isfinite(x) for x in (coupling, gap, tau)) or coupling <= 0 or gap < 0:
        raise ValueError("finite parameters, positive coupling and nonnegative gap required")
    detuning = (k-1)/2-gap
    frequency = math.hypot(coupling, detuning/2)
    maximum = (coupling/frequency)**2
    return {"detuning": detuning, "frequency": frequency,
            "maximum": maximum, "probability": maximum*math.sin(frequency*tau)**2,
            "first_peak_tau": math.pi/(2*frequency)}


def coordinate_eigen_check(k, occupation, q):
    """Hermite 파동함수의 독립 미분으로 실제 자식 에너지 방정식을 검사한다."""
    source.child_count(k)
    if isinstance(occupation, bool) or not isinstance(occupation, int) or not 0 <= occupation <= 12:
        raise ValueError("occupation must be an integer from 0 through 12")
    q = np.asarray(q, dtype=float)
    if q.shape != (k,) or not np.isfinite(q).all():
        raise ValueError("one finite coordinate per child required")
    s = source.source_dilation(k)
    matrix = s[::2, ::2]
    inverse = np.linalg.solve(matrix, np.eye(k))
    x = inverse @ q
    energy = transport.local_energy(k)
    normalizer = (math.pi**(-k/4) / math.sqrt(abs(np.linalg.det(matrix)))
                  / math.sqrt(2**occupation*math.factorial(occupation)))
    gaussian = normalizer*math.exp(-float(x @ x)/2)
    hermite = np.polynomial.hermite.hermval(x[0], [0.]*occupation+[1.])
    value = gaussian*hermite
    lower = (np.polynomial.hermite.hermval(x[0], [0.]*(occupation-1)+[1.])
             if occupation else 0.)
    derivative = gaussian*(2*occupation*lower-x[0]*hermite)
    hessian = (np.outer(x, x)-np.eye(k))*value
    hessian[0, 0] = (x[0]**2-2*occupation-1)*value
    for axis in range(1, k):
        hessian[0, axis] = hessian[axis, 0] = -x[axis]*derivative
    hessian_q = inverse.T @ hessian @ inverse
    applied = .5*(float(q @ energy[::2, ::2] @ q)*value
                  -np.sum(energy[1::2, 1::2]*hessian_q))
    expected = (occupation+k/2)*value
    return {"branching": k, "occupation": occupation, "coordinate": q.tolist(),
            "wavefunction": float(value), "applied_energy": float(applied),
            "expected_energy": float(expected), "residual": float(abs(applied-expected))}


def gap_ledger(k, depth):
    source.child_count(k)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 1024:
        raise ValueError("depth must be an integer from 0 through 1024")
    return Fraction(k**depth-1, 2)


def evolution_check(k, battery_gap=None, coupling=.25):
    data = model(k, coupling=coupling, battery_gap=battery_gap)
    parent_indices, child_indices = data["parent_indices"], data["child_indices"]
    n = len(parent_indices)
    amplitudes = np.arange(1, n+1) + 1j*np.arange(n, 0, -1)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    initial = np.zeros(len(data["free"]), dtype=complex)
    initial[parent_indices] = amplitudes
    law = transfer_law(k, coupling, data["battery_gap"], 0.)
    times = (0., law["first_peak_tau"]/2, law["first_peak_tau"],
             2*law["first_peak_tau"])
    initial_prob = np.abs(initial)**2
    e0 = float(np.vdot(initial, data["hamiltonian"] @ initial).real)
    rows = []
    for tau in times:
        final = propagator(data, tau) @ initial
        probability = np.abs(final)**2
        pchild = float(probability[data["child_start"]:].sum())
        theory = transfer_law(k, coupling, data["battery_gap"], tau)["probability"]
        oscillator_change = float(data["oscillator"] @ (probability-initial_prob))
        battery_change = float(data["battery"] @ (probability-initial_prob))
        interaction_energy = float(np.vdot(final, data["interaction"] @ final).real)
        rows.append({"tau": tau, "child_probability": pchild,
                     "oscillator_energy_change": oscillator_change,
                     "battery_energy_change": battery_change,
                     "interaction_energy": interaction_energy,
                     "residuals": {
                         "rabi_probability": abs(pchild-theory),
                         "norm": abs(float(probability.sum())-1),
                         "total_energy": abs(float(np.vdot(final, data["hamiltonian"] @ final).real)-e0),
                         "oscillator_flow": abs(oscillator_change-data["delta"]*pchild),
                         "battery_flow": abs(battery_change+data["battery_gap"]*pchild),
                         "interaction_flow": abs(interaction_energy+law["detuning"]*pchild),
                     }})
    return {"branching": k, "coupling": coupling, "battery_gap": data["battery_gap"],
            "delta": data["delta"], "finite_invariant_dimension": len(data["free"]),
            "maximum_probability": law["maximum"],
            "free_energy_commutator_norm": float(np.linalg.norm(
                data["free"] @ data["interaction"] - data["interaction"] @ data["free"])),
            "rows": rows}


def run():
    coordinate = [coordinate_eigen_check(k, n, np.linspace(-.7, 1.1, k)+offset)
                  for k in (2, 3, 4) for n in range(4) for offset in (0., .37)]
    evolution = [evolution_check(k, gap) for k in (2, 3, 4)
                 for gap in (None, 0.)]
    residuals = [row["residual"] for row in coordinate]
    residuals += [value for case in evolution for row in case["rows"]
                  for value in row["residuals"].values()]
    if max(residuals) > 1e-11:
        raise RuntimeError("autonomous split verification failed")
    paths = (Path(__file__), Path(transport.__file__), Path(source.__file__),
             source.SPLIT_SOURCE, Path(__file__).with_name("interface_bath.py"))
    return {
        "scope": "공급한 분할 등거리 사상·부문 에너지·배터리의 시간 독립 유니터리 실현",
        "energy_unit": "hbar*omega", "time_unit": "tau=omega*t",
        "coupling_unit": "g/(hbar*omega)", "numpy": np.__version__,
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "coordinate_checks": coordinate, "evolution_checks": evolution,
        "max_residual": max(residuals),
        "ternary_gap_ledger": [{"depth": d, "energy_exact": str(gap_ledger(3, d))} for d in range(6)],
        "time_independent_hamiltonian_constructed": True,
        "finite_subspace_exactly_invariant": True,
        "bare_fock_gaussian_truncation_used": False,
        "unitary_and_sector_energies_supplied": True,
        "coupling_contains_supplied_nonlocal_isometry": True,
        "absolute_mode_count_ground_offset_supplied": True,
        "reversible_cycle_returns_parent_and_battery": True,
        "cycle_leaves_net_children_or_records": False,
        "split_output_is_stationary": False,
        "merges_arbitrary_child_states": False,
        "global_tree_hamiltonian_constructed": False,
        "fixed_free_energy_bound_replaced": False,
        "CE_local_action_derived": False,
        "bath_coupling_derived": False,
        "common_metric_selection_proved": False,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "max_residual": result["max_residual"],
                      "peak_probabilities": [(x["branching"], x["battery_gap"], x["maximum_probability"])
                                             for x in result["evolution_checks"]]}, ensure_ascii=False))

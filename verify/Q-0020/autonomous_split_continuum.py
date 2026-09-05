"""공급한 분할 사상과 연속 방출 환경의 시간 독립 단일 분할.

E_*=hbar*omega, tau=omega*t, Delta=(k-1)/2, epsilon=b-Delta.
활동 공간에서 H/E_*=H_P/E_*+Delta+h이며
h=[[epsilon,<v|],[|v>,x]], v(x)=sqrt(strength*x*exp(-x)).
strength는 결합 벡터의 노름 제곱이고 앞 Rabi 모형의 kappa와 다르다.
epsilon>strength>0의 순수 연속 스펙트럼 증명은 유도 문서가 소유한다.
여기서는 유한 시간 진폭·에너지, 독립 유한 환경, 간격만 충전한 대조를 검산한다.
연속 환경의 초기 비점유 상태와 충전 배터리는 입력이다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import autonomous_split as split
import continuum_bath as bath


def parameters(k, battery_gap, strength, tau=0.):
    split.source.child_count(k)
    gap, strength, tau = map(float, (battery_gap, strength, tau))
    if not all(math.isfinite(x) for x in (gap, strength, tau)):
        raise ValueError("유한한 배터리 간격·결합·시간이 필요하다")
    if gap < 0 or strength <= 0 or tau < 0:
        raise ValueError("배터리 간격·시간은 음이 아니고 결합은 양수여야 한다")
    return (k-1)/2, gap-(k-1)/2, strength, tau


def response(k=3, battery_gap=2., strength=.25, tau=1.):
    """연속 환경의 에너지 흐름을 반환한다. 적분 오차는 인증 상한이 아니다."""
    delta, epsilon, strength, tau = parameters(k, battery_gap, strength, tau)
    bath.parameters(epsilon, strength)
    # 기존 충돌 함수의 endpoint는 입력+strength다. 빼서 넣어 같은 h를 얻는다.
    # 여기서는 결합을 전환하지 않는다. 그 함수의 전환 일 필드는 사용하지 않는다.
    old = bath.collision_response(epsilon-strength, strength, tau)
    survival = old["survival_probability"]
    child = 1-survival
    oscillator = delta*child
    battery = -battery_gap*child
    outgoing = old["bath_energy_per_initial_number"]
    interaction = old["interaction_energy_per_initial_number"]
    return {
        "branching": k, "battery_gap": float(battery_gap),
        "delta": delta, "endpoint": epsilon, "strength": strength, "tau": tau,
        "amplitude": old["amplitude"],
        "first_energy_amplitude": old["first_energy_amplitude"],
        "parent_probability": survival, "child_probability": child,
        "oscillator_energy_change": oscillator, "battery_energy_change": battery,
        "outgoing_energy": outgoing, "interaction_energy": interaction,
        "energy_balance_residual": abs(oscillator+battery+outgoing+interaction),
        "quadrature_error_estimate": old["quadrature_error_estimate"],
        "moment_relative_residual": old["moment_relative_residual"],
    }


def finite_star(endpoint, strength, modes=64):
    """연속체 적분의 유한 환경 대조. 무한 시간 극한의 증명이 아니다."""
    endpoint, strength = float(endpoint), float(strength)
    if not math.isfinite(endpoint) or not math.isfinite(strength) or strength <= 0:
        raise ValueError("유한한 endpoint와 양의 결합이 필요하다")
    if isinstance(modes, bool) or not isinstance(modes, int) or not 8 <= modes <= 160:
        raise ValueError("환경 모드 수는 8 이상 160 이하 정수여야 한다")
    nodes, weights = np.polynomial.laguerre.laggauss(modes)
    coupling = np.sqrt(strength*nodes*weights)
    h = np.diag(np.r_[endpoint, nodes])
    h[0, 1:] = h[1:, 0] = coupling
    return h


def finite_response(endpoint, strength, tau, modes=64):
    """유한 행렬을 직접 진화하고 각 에너지를 기대값으로 계산한다."""
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("유한한 음이 아닌 시간이 필요하다")
    h = finite_star(endpoint, strength, modes)
    energies, vectors = np.linalg.eigh(h)
    state = vectors @ (vectors[0]*np.exp(-1j*energies*tau))
    interaction = 2*(state[0].conjugate()*(h[0, 1:] @ state[1:])).real
    return {
        "modes": modes, "amplitude": [float(state[0].real), float(state[0].imag)],
        "parent_probability": float(abs(state[0])**2),
        "outgoing_energy": float(np.diag(h)[1:] @ np.abs(state[1:])**2),
        "interaction_energy": float(interaction),
        "energy_balance_residual": abs(float(np.vdot(state, h @ state).real)-endpoint),
        "norm_residual": abs(float(np.vdot(state, state).real)-1),
    }


def gap_only_bound_state(strength=.25):
    """b=Delta일 때 음의 고유값과 endpoint의 스펙트럼 원자 질량을 계산한다."""
    strength = float(strength)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("유한한 양의 결합이 필요하다")

    def denominator(z):
        if z == 0:
            return strength
        integral = quad(lambda x: x*math.exp(-x)/(x-z), 0., math.inf,
                        epsabs=1e-12, epsrel=1e-12)[0]
        return z+strength*integral

    energy = brentq(denominator, -strength, 0., xtol=1e-14)
    derivative = 1+strength*quad(
        lambda x: x*math.exp(-x)/(x-energy)**2, 0., math.inf,
        epsabs=1e-12, epsrel=1e-12)[0]
    atom = 1/derivative
    return {
        "endpoint": 0., "strength": strength, "bound_energy": energy,
        "endpoint_atom_mass": atom,
        "asymptotic_parent_probability": atom**2,
        "asymptotic_child_probability": 1-atom**2,
        "secular_residual": abs(denominator(energy)),
        "interpretation": "원자 질량 w의 진폭 기여는 w*exp(-i*z*tau), 잔류 확률은 w^2",
    }


def run():
    moments = []
    for power, expected in enumerate((1., 1., 1.25)):
        value, error = bath.spectral_moment(1., .25, power)
        moments.append({"power": power, "value": value, "expected": expected,
                        "residual": abs(value-expected), "quadrature_error_estimate": error})
    rows = [response(k, (k-1)/2+1., .25, tau)
            for k in (2, 3, 4) for tau in (0., .25, 1., 3., 20., 100.)]
    cross = []
    for tau in (.25, 1., 3.):
        continuum = response(tau=tau)
        finite = finite_response(1., .25, tau)
        residual = max(abs(complex(*continuum["amplitude"])-complex(*finite["amplitude"])),
                       abs(continuum["outgoing_energy"]-finite["outgoing_energy"]),
                       abs(continuum["interaction_energy"]-finite["interaction_energy"]))
        cross.append({"tau": tau, "finite": finite, "comparison_residual": residual})
    bound = gap_only_bound_state()
    energies, vectors = np.linalg.eigh(finite_star(0., .25, 128))
    bound["finite_128_bound_energy"] = float(energies[0])
    bound["finite_128_atom_mass"] = float(vectors[0, 0]**2)
    residuals = [row["energy_balance_residual"] for row in rows]
    residuals += [row["residual"] for row in moments]
    residuals += [row["comparison_residual"] for row in cross]
    if max(residuals) > 1e-8:
        raise ArithmeticError("연속 분할의 모멘트·에너지·유한 환경 대조가 일치하지 않는다")
    paths = (Path(__file__), Path(split.__file__), Path(bath.__file__),
             Path(split.source.__file__), Path(split.transport.__file__),
             split.source.SPLIT_SOURCE, HERE / "interface_bath.py")
    return {
        "scope": "공급한 분할 사상·배터리·연속 환경에서 자식 확률의 점근적 유지",
        "energy_unit": "hbar*omega", "time_unit": "tau=omega*t",
        "coupling_definition": "integral |v(x)|^2 dx = strength",
        "interpreter": sys.executable, "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "spectral_moments": moments, "continuum_rows": rows, "finite_checks": cross,
        "gap_only_control": bound, "max_checked_residual": max(residuals),
        "strict_decay_condition": "battery_gap-(branching-1)/2 > strength > 0",
        "time_independent_hamiltonian_constructed": True,
        "conditional_asymptotic_child_probability_one": True,
        "reference_information_preserved_on_split_image": True,
        "finite_environment_proves_infinite_time_limit": False,
        "critical_endpoint_equals_strength_tested": False,
        "stationary_global_state_proved": False,
        "physical_pointer_label_record_derived": False,
        "unitary_isometry_and_environment_supplied": True,
        "initial_parent_and_charged_battery_supplied": True,
        "autonomous_initial_preparation_derived": False,
        "battery_recharged_with_children_retained": False,
        "global_tree_hamiltonian_constructed": False,
        "fixed_free_energy_bound_replaced": False,
        "CE_local_action_derived": False,
        "common_metric_selection_proved": False,
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "max_checked_residual": result["max_checked_residual"],
                      "gap_only_control": result["gap_only_control"]}, ensure_ascii=False))


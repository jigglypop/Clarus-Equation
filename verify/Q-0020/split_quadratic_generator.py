"""같은 분할 등거리 사상을 양의 이차 해밀토니안으로 생성한다.

[q,p]=i, H/E_*=R^T K R/2, tau=E_*t/hbar를 쓴다.
보조 진공을 보존하는 수동 회전은 분할 사상의 출력을 전역 위상까지만 바꾼다.
단위원 위 단순 고유값을 가진 정준변환에서 실수 정준 고유기저를 만들고,
양의 회전각 가지를 선택하여 K>0을 구성한다. 기하학적 국소성과 CE 작용은 미유도다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_quantum_source as source


def rotation(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, sine], [-sine, cosine]])


def equivalent_dilation(k, angles):
    """보조 진공을 보존하는 회전만 원래 분할 앞에 넣는다."""
    source.child_count(k)
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (k-1,) or not np.isfinite(angles).all():
        raise ValueError("보조 모드마다 유한 회전각 하나가 필요하다")
    passive = np.eye(2*k)
    for index, angle in enumerate(angles, start=1):
        passive[2*index:2*index+2, 2*index:2*index+2] = rotation(angle)
    return source.source_dilation(k) @ passive


def positive_generator(k, angles, winding=0):
    """회전각에 2*pi*winding을 더한 양의 생성자를 만든다."""
    if isinstance(winding, bool) or not isinstance(winding, int) or winding < 0:
        raise ValueError("회전수는 음이 아닌 정수여야 한다")
    target = equivalent_dilation(k, angles)
    omega = np.kron(np.eye(k), source.J)
    values, vectors = np.linalg.eig(target)
    if np.max(abs(abs(values)-1)) > 1e-9 or np.min(abs(values.imag)) < 1e-7:
        raise ValueError("단순한 비실수 단위원 고유값을 가진 변환이 필요하다")
    modes = []
    for index, value in enumerate(values):
        if value.imag < 0:
            continue
        vector = vectors[:, index]
        real, imag = vector.real, vector.imag
        sign = float(real @ omega @ imag)
        if abs(sign) < 1e-10:
            raise ArithmeticError("정준 고유벡터의 정규화가 특이하다")
        if sign < 0:
            imag, value = -imag, value.conjugate()
            sign = -sign
        frequency = math.atan2(value.imag, value.real) % (2*math.pi)
        modes.append((frequency, real/math.sqrt(sign), imag/math.sqrt(sign)))
    if len(modes) != k:
        raise ArithmeticError("정준 고유쌍 수가 모드 수와 다르다")
    modes.sort(key=lambda item: item[0])
    basis = np.column_stack([column for _, real, imag in modes for column in (real, imag)])
    frequencies = np.array([item[0]+2*math.pi*winding for item in modes])
    inverse = np.linalg.solve(basis, np.eye(2*k))
    metric = inverse.T @ np.diag(np.repeat(frequencies, 2)) @ inverse
    metric = (metric+metric.T)/2
    evolved = expm(omega @ metric)
    residuals = {
        "basis_symplectic": float(np.linalg.norm(basis.T @ omega @ basis-omega)),
        "target_symplectic": float(np.linalg.norm(target.T @ omega @ target-omega)),
        "finite_time_map": float(np.linalg.norm(evolved-target)),
        "generator_energy_invariance": float(np.linalg.norm(target.T @ metric @ target-metric)),
        "parent_map": float(np.linalg.norm(evolved[:, :2]-source.source_dilation(k)[:, :2])),
        "ancilla_noise": float(np.linalg.norm(
            .5*evolved[:, 2:] @ evolved[:, 2:].T
            -.5*source.source_dilation(k)[:, 2:] @ source.source_dilation(k)[:, 2:].T)),
    }
    if min(np.linalg.eigvalsh(metric)) <= 0 or max(residuals.values()) > 1e-8:
        raise ArithmeticError("양성·에너지·분할 사상 검산이 통과하지 못했다")
    prepared = float(np.trace(metric)/4)
    ground = float(sum(frequencies)/2)
    initial_covariance = np.eye(2*k)/2
    final_covariance = target @ initial_covariance @ target.T
    initial_bare = k/2
    final_bare = float(np.trace(final_covariance)/2)
    return {"branching": k, "ancilla_angles": list(map(float, angles)), "winding": winding,
            "target": target, "basis": basis, "frequencies": frequencies,
            "generator": metric, "generator_eigenvalues": np.linalg.eigvalsh(metric),
            "prepared_vacuum_energy": prepared, "generator_ground_energy": ground,
            "prepared_energy_above_ground": prepared-ground,
            "external_switching": {
                "initial_bare_energy": initial_bare, "final_bare_energy": final_bare,
                "switch_on_work": prepared-initial_bare,
                "switch_off_work": final_bare-prepared,
                "net_work": final_bare-initial_bare,
                "switching_is_autonomous": False,
            },
            "residuals": residuals}


def original_dilation_obstruction(k):
    """원래 지정된 전체 S의 불가능성만 검사한다. 같은 V 전체로 확대하지 않는다."""
    source.child_count(k)
    q_block = source.source_dilation(k)[::2, ::2]
    determinant = math.sqrt(k)*(2*k/(k-1))**((k-1)/2)
    return {"branching": k, "q_determinant_magnitude": determinant,
            "spectral_radius_lower_bound": determinant**(1/k),
            "actual_q_spectral_radius": float(max(abs(np.linalg.eigvals(q_block)))),
            "determinant_residual": abs(abs(float(np.linalg.det(q_block)))-determinant)}


ANGLE_FRACTIONS = {2: ((2, 3),), 3: ((13, 9), (4, 5))}


def witness(k, winding=0):
    if isinstance(k, bool) or k not in ANGLE_FRACTIONS:
        raise ValueError("확정한 생성자 증인은 k=2,3에 한정된다")
    return positive_generator(k, [math.pi*n/d for n, d in ANGLE_FRACTIONS[k]], winding)


class _Interval:
    """분모 10^30의 유리수 구간. 모든 연산에서 바깥쪽으로 반올림한다."""

    scale = 10**30

    def __init__(self, low, high):
        self.low, self.high = int(low), int(high)

    @classmethod
    def rational(cls, numerator, denominator=1):
        if denominator <= 0:
            raise ValueError("구간의 유리수 분모는 양수여야 한다")
        top = numerator*cls.scale
        return cls(top//denominator, -((-top)//denominator))

    @classmethod
    def sqrt(cls, numerator, denominator=1):
        floor = math.isqrt(numerator*cls.scale**2//denominator)
        return cls(floor, floor+1)

    def __add__(self, other):
        if isinstance(other, int):
            other = self.rational(other)
        return _Interval(self.low+other.low, self.high+other.high)

    __radd__ = __add__

    def __neg__(self):
        return _Interval(-self.high, -self.low)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int):
            other = self.rational(other)
        products = [a*b for a in (self.low, self.high) for b in (other.low, other.high)]
        return _Interval(min(products)//self.scale, -((-max(products))//self.scale))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, int):
            other = self.rational(other)
        if other.low <= 0 <= other.high:
            raise ArithmeticError("0을 포함한 구간으로 나눌 수 없다")
        squares = self.scale**2
        inverse = _Interval(squares//other.high, -((-squares)//other.low))
        return self*inverse

    def sign(self):
        if self.low > 0:
            return 1
        if self.high < 0:
            return -1
        raise ArithmeticError("유리수 구간이 부호를 결정하지 못했다")

    def exact_bounds(self):
        return [f"{self.low}/{self.scale}", f"{self.high}/{self.scale}"]


def _pi_interval():
    # Machin 식과 교대급수의 다음 항이 주는 엄밀한 나머지 구간이다.
    def arctangent_inverse(denominator):
        total = _Interval.rational(0)
        for j in range(30):
            total += _Interval.rational((-1)**j, (2*j+1)*denominator**(2*j+1))
        error = _Interval.rational(1, 61*denominator**61)
        return total+_Interval(0, error.high)
    return 16*arctangent_inverse(5)-4*arctangent_inverse(239)


def _trigonometric_interval(numerator, denominator):
    angle = _pi_interval()*_Interval.rational(numerator, denominator)
    if max(abs(angle.low), abs(angle.high)) >= 7*_Interval.scale:
        raise ValueError("이 증명 구간은 절댓값 7 미만 각도만 지원한다")
    square = angle*angle
    sine, cosine = angle, _Interval.rational(1)
    sine_term, cosine_term = sine, cosine
    for j in range(1, 40):
        sine_term = -sine_term*square/((2*j)*(2*j+1))
        cosine_term = -cosine_term*square/((2*j-1)*(2*j))
        sine += sine_term
        cosine += cosine_term
    # sin은 79차, cos는 78차 Taylor 다항식. 두 나머지에 공통 상한을 쓴다.
    remainder = _Interval.rational(7**79, math.factorial(79))
    error = _Interval(-remainder.high, remainder.high)
    return cosine+error, sine+error


def exact_spectral_certificate(k):
    """정확한 유리수 구간의 부호로 y=lambda+1/lambda의 모든 근을 가둔다."""
    if isinstance(k, bool) or k not in ANGLE_FRACTIONS:
        raise ValueError("정확한 구간 증인은 k=2,3에 한정된다")
    zero = _Interval.rational(0)
    size = 2*k
    original = [[zero for _ in range(size)] for _ in range(size)]
    passive = [[_Interval.rational(int(i == j)) for j in range(size)] for i in range(size)]
    for i in range(k):
        for j in range(k):
            coefficient = (_Interval.rational(1)/_Interval.sqrt(k) if j == 0 else
                           _Interval.rational(1 if i < j else -j if i == j else 0)
                           /_Interval.sqrt(j*(j+1)))
            gain = _Interval.sqrt(k) if j == 0 else _Interval.sqrt(2*k, k-1)
            original[2*i][2*j] = coefficient*gain
            original[2*i+1][2*j+1] = coefficient/gain
    for mode, (numerator, denominator) in enumerate(ANGLE_FRACTIONS[k], start=1):
        cosine, sine = _trigonometric_interval(numerator, denominator)
        i = 2*mode
        passive[i][i], passive[i][i+1] = cosine, sine
        passive[i+1][i], passive[i+1][i+1] = -sine, cosine

    def multiply(a, b):
        return [[sum((a[i][r]*b[r][j] for r in range(size)), zero)
                 for j in range(size)] for i in range(size)]

    def trace(a):
        return sum((a[i][i] for i in range(size)), zero)

    target = multiply(original, passive)
    squared = multiply(target, target)
    t1, t2 = trace(target), trace(squared)
    a1, a2 = -t1, (t1*t1-t2)/2
    if k == 2:
        coefficients = [_Interval.rational(1), a1, a2-2]
        brackets = [((3, 5), (7, 10)), ((17, 10), (9, 5))]
    else:
        t3 = trace(multiply(squared, target))
        a3 = -(t1*t1*t1-3*t1*t2+2*t3)/6
        coefficients = [_Interval.rational(1), a1, a2-3, a3-2*a1]
        brackets = [((-1, 5), (0, 1)), ((13, 10), (7, 5)), ((19, 10), (39, 20))]
    intervals = []
    for left, right in brackets:
        values = []
        for endpoint in (left, right):
            x = _Interval.rational(*endpoint)
            value = coefficients[0]
            for coefficient in coefficients[1:]:
                value = value*x+coefficient
            values.append(value)
        if values[0].sign()*values[1].sign() != -1:
            raise ArithmeticError("특성다항식의 근 구간에 부호 전환이 없다")
        intervals.append({"left": f"{left[0]}/{left[1]}", "right": f"{right[0]}/{right[1]}",
                          "signs": [v.sign() for v in values],
                          "polynomial_value_bounds": [v.exact_bounds() for v in values]})
    return {"branching": k, "angle_fractions_of_pi": ANGLE_FRACTIONS[k],
            "reciprocal_polynomial_degree": k,
            "coefficient_bounds": [c.exact_bounds() for c in coefficients],
            "root_intervals": intervals,
            "arithmetic": "정수 사칙연산·정수 제곱근·바깥쪽 유리수 반올림",
            "pi_identity": "pi=16*atan(1/5)-4*atan(1/239), 교대급수 30항",
            "trig_remainder_bound": "7^79/79!, |angle|<7",
            "roots_simple_and_strictly_between_minus_two_and_two": True}


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(v) for v in value]
    return value


def run():
    certificates = [exact_spectral_certificate(k) for k in (2, 3)]
    cases = [witness(k, winding) for k in (2, 3) for winding in (0, 1)]
    paths = (Path(__file__), Path(source.__file__), source.SPLIT_SOURCE, HERE/"interface_bath.py")
    result = {
        "scope": "보조 진공을 보존하는 동등한 k=2,3 분할의 양의 시간 독립 이차 생성자",
        "energy_unit": "E_star", "time_unit": "tau=E_star*t/hbar; map evaluated at tau=1",
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "interpreter": sys.executable,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "original_dilation_obstructions": [original_dilation_obstruction(k) for k in (2, 3, 4, 8)],
        "exact_spectral_certificates": certificates, "generator_cases": cases,
        "max_residual": max(v for row in cases for v in row["residuals"].values()),
        "positive_quadratic_generator_for_same_isometry": True,
        "specified_original_full_dilation_generated": False,
        "all_branching_numbers_proved": False,
        "generator_energy_uniquely_fixed_by_split": False,
        "vacuum_ancillas_preexist_and_are_supplied": True,
        "couplings_and_readout_time_supplied": True,
        "external_switch_work_accounted": True,
        "autonomous_switching_or_battery_preparation_derived": False,
        "child_output_permanently_retained": False,
        "joined_to_emission_bath_in_one_local_action": False,
        "CE_local_action_derived": False,
        "common_metric_selection_proved": False,
    }
    return _serializable(result)


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "max_residual": result["max_residual"],
                      "cases": [{"k": r["branching"], "winding": r["winding"],
                                 "min_eigenvalue": min(r["generator_eigenvalues"]),
                                 "preparation_above_ground": r["prepared_energy_above_ground"],
                                 "net_switch_work": r["external_switching"]["net_work"]}
                                for r in result["generator_cases"]]}, ensure_ascii=False))

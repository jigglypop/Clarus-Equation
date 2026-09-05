"""원래 사후 제약의 영모드 누설과 실제 그램 구간의 정준 가지를 검산한다.

공유 길이·내적·주기 경계조건과 대칭 연산자 순서는 공급 조건이다.
유한 누설과 주기 미분 연산자의 도메인 실패를 구별한다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import regge_quantum_constraint_transfer as transfer

full, moves = transfer.full, transfer.moves
E, Y = moves.E_ID, moves.Y_ID
Y0 = math.sqrt(8/3)


def symmetric_g(edge, h):
    """슐레플리 소거 뒤 실제 경계 변 y의 작용 미분."""
    _, _, _, spoke, rim = full.geometry(edge, h)
    q = h*h+1
    coefficient = (2*q-Y0*Y0)/math.sqrt(4*q-Y0*Y0)
    return coefficient*(math.pi-2*spoke)+(Y0/math.sqrt(3))*(math.pi-rim)


def symmetric_g_e(edge, h):
    edge, remainder, _, _, _ = full.geometry(edge, h)
    if np.any((edge <= 0) | (edge >= full.limit(h))):
        raise ValueError("기울기는 열린 그램 구간에서 계산한다")
    return 4*math.sqrt(2)/remainder*((6*h*h-2)/(9*remainder**2+8)-1/3)


def moments(h, order=128, beta=5.):
    edge, weight = transfer.sample_rule((0., full.limit(h)), order)
    weight = weight/full.limit(h)
    values = symmetric_g(edge, h)
    mean = float(np.dot(weight, values))
    variance = float(np.dot(weight, (values-mean)**2))
    return {"mean": mean, "variance": variance, "leakage_squared": beta*beta*variance}


def geometry_check(h, step=2e-5):
    q = transfer.geometry_lengths(h)
    upper = full.limit(h)
    errors, gram = [], []
    for fraction in (.17, .43, .81):
        q[E] = upper*fraction
        data = moves.FINAL.evaluate(q)
        errors.append(abs(data["gradient"][Y]-symmetric_g(q[E], h)))
        gram.append(data["minimum_gram_eigenvalue"])
    lower, true_upper = transfer.fine_interval(q)
    def upper_y(offset):
        current = q.copy()
        current[Y] += offset
        return transfer.fine_interval(current)[1]
    left = (4*(true_upper-upper_y(-step/2))-(true_upper-upper_y(-step)))/step
    right = (4*(upper_y(step/2)-true_upper)-(upper_y(step)-true_upper))/step
    expected_right = -4*Y0/(9*upper)
    support_error = 0.
    for offset in (-.01, .01):
        y = Y0+offset
        new_upper = 2*math.sqrt(h*h+1-Y0**4/(4*Y0*Y0-y*y))
        support_error = max(support_error, abs(upper_y(offset)-min(upper, new_upper)))
    return {"h": h, "gradient_error": float(max(errors)), "minimum_gram": float(min(gram)),
            "interval_error": float(max(abs(lower), abs(true_upper-upper), support_error)),
            "left_upper_derivative": float(left), "right_upper_derivative": float(right),
            "right_derivative_error": float(abs(right-expected_right)),
            "flat_mixed_derivative": float(symmetric_g_e(2*h, h)),
            "flat_mixed_formula_error": float(abs(symmetric_g_e(2*h, h)-3*math.sqrt(2)*(h*h-1)))}


def operator_check(lengths, kind, fraction=.43, beta=5., momentum=.7, step=2e-5):
    """매끄러운 경계 구역 안에서 실제 위상·밀도 파동함수를 독립 차분한다."""
    interval = np.array(transfer.fine_interval(lengths))
    lower, upper = interval
    width = upper-lower
    edge = lower+fraction*width
    def state(offset):
        current = lengths.copy()
        current[Y] += offset
        bounds = np.array(transfer.fine_interval(current))
        length = bounds[0]+fraction*(bounds[1]-bounds[0])
        current[E] = length
        rho, _ = transfer.measure(length, bounds, kind)
        phase = moves.actions(current)["second"]["action"]
        psi = np.exp(1j*(momentum*current[Y]-beta*phase))/math.sqrt((bounds[1]-bounds[0])*rho)
        return psi, bounds
    def central(delta):
        plus, upper_bounds = state(delta)
        minus, lower_bounds = state(-delta)
        return (plus-minus)/(2*delta), (upper_bounds-lower_bounds)/(2*delta)
    coarse, coarse_bounds = central(step)
    fine, fine_bounds = central(step/2)
    derivative = (4*fine-coarse)/3
    bounds_y = (4*fine_bounds-coarse_bounds)/3
    lower_y, upper_y = bounds_y
    width_y = upper_y-lower_y
    speed = lower_y+fraction*width_y
    divergence = (0. if kind == "length" else
                  speed/edge-2*(upper*upper_y-lower*lower_y)/(upper*upper-lower*lower)+width_y/width)
    q = lengths.copy()
    q[E] = edge
    data = moves.actions(q)
    potential = beta*(speed*data["second"]["gradient"][E]-data["first"]["gradient"][Y])
    g = data["first"]["gradient"][Y]+data["second"]["gradient"][Y]
    psi, _ = state(0.)
    actual = -1j*(derivative+.5*divergence*psi)+potential*psi
    expected = (momentum-beta*g)*psi
    omitted = -1j*derivative+potential*psi
    def fixed_state(offset):
        current = lengths.copy()
        current[Y] += offset
        bounds = transfer.fine_interval(current)
        current[E] = edge
        rho, _ = transfer.measure(edge, bounds, kind)
        phase = moves.actions(current)["second"]["action"]
        value = np.exp(1j*(momentum*current[Y]-beta*phase))/math.sqrt((bounds[1]-bounds[0])*rho)
        return value, math.log(rho)
    def fixed_central(delta):
        plus, plus_log = fixed_state(delta)
        minus, minus_log = fixed_state(-delta)
        return (plus-minus)/(2*delta), (plus_log-minus_log)/(2*delta)
    c_coarse, rho_coarse = fixed_central(step)
    c_fine, rho_fine = fixed_central(step/2)
    rho_y = (4*rho_fine-rho_coarse)/3
    original = -1j*((4*c_fine-c_coarse)/3+.5*rho_y*psi)-beta*data["first"]["gradient"][Y]*psi
    scalar_shift = .5j*width_y/width*psi
    return {"kind": kind, "beta": beta, "lower": float(lower), "upper": float(upper),
            "divergence": float(divergence), "operator_error": float(abs(actual-expected)),
            "original_c_error": float(abs(original-expected-scalar_shift)),
            "original_c_lift_relation_error": float(abs(original-actual-scalar_shift)),
            "omitted_connection_residual": float(abs(omitted-expected)),
            "omitted_connection_size": float(abs(.5*divergence*psi)),
            "step_difference": float(abs(fine-coarse))}


def fourier_check(h=1., grid=2**17, max_mode=1024):
    """유한 푸리에 절단의 도메인 진단. 발산 합을 연산자 노름으로 선언하지 않는다."""
    if grid < 2*max_mode+1:
        raise ValueError("격자는 푸리에 절단보다 충분히 커야 한다")
    upper = full.limit(h)
    t = (np.arange(grid)+.5)/grid
    values = symmetric_g(upper*t, h)
    modes = np.arange(1, max_mode+1)
    coefficient = np.fft.rfft(values)[1:max_mode+1]/grid*np.exp(-1j*math.pi*modes/grid)
    power = 2*np.abs(coefficient)**2
    energy = np.cumsum((2*math.pi*modes/upper)**2*power)
    variance = np.cumsum(power)
    jump = float(symmetric_g(upper, h)-symmetric_g(0., h))
    rows = [{"cutoff": cutoff, "partial_variance": float(variance[cutoff-1]),
             "derivative_sum_over_cutoff": float(energy[cutoff-1]/cutoff),
             "n_coefficient": full.pair(cutoff*coefficient[cutoff-1])}
            for cutoff in (16, 64, 256, 1024) if cutoff <= max_mode]
    return {"grid": grid, "jump": jump, "asymptotic_rate": 2*jump*jump/(upper*upper),
            "asymptotic_n_coefficient": full.pair(-jump/(2j*math.pi)), "rows": rows}


def darboux_branches(h=1.):
    """G_e가 0인 내부 점의 양쪽 정준 가지와 역함수의 한계를 대조한다."""
    upper = full.limit(h)
    if h*h <= 7/9:
        return {"h": h, "critical_edge": None, "branches": []}
    critical = math.sqrt(2*(h*h+1))
    remainder = math.sqrt(upper*upper-critical*critical)
    curvature = 24*math.sqrt(2)*critical/(remainder*(9*remainder**2+8))
    minimum = float(symmetric_g(critical, h))
    rows = []
    for offset in (1e-3, 2.5e-4, 6.25e-5):
        target = minimum+offset
        if target >= min(symmetric_g(0., h), symmetric_g(upper, h)):
            continue
        left = brentq(lambda e: float(symmetric_g(e, h)-target), 0., critical, xtol=1e-13)
        right = brentq(lambda e: float(symmetric_g(e, h)-target), critical, upper, xtol=1e-13)
        predicted = math.sqrt(2*offset/curvature)
        rows.append({"g_offset": offset, "left": left, "right": right,
                     "inverse_residual": float(max(abs(symmetric_g(left, h)-target),
                                                   abs(symmetric_g(right, h)-target))),
                     "left_gradient": float(symmetric_g_e(left, h)),
                     "right_gradient": float(symmetric_g_e(right, h)),
                     "separation_ratio": (right-left)/(2*predicted)})
    delta = 1e-5
    numerical_curvature = (symmetric_g_e(critical+delta, h)-symmetric_g_e(critical-delta, h))/(2*delta)
    return {"h": h, "critical_edge": critical, "critical_g": minimum, "curvature": curvature,
            "critical_gradient": float(symmetric_g_e(critical, h)),
            "curvature_error": float(abs(numerical_curvature-curvature)), "branches": rows}


def run():
    geometry, distributions, operators = [], [], []
    for h in (.5, .9, 1., 1.1, math.sqrt(5/3)):
        geometry.append(geometry_check(h))
        low, high = moments(h, 128), moments(h, 256)
        distributions.append({"h": h, **high,
                              "quadrature_difference": max(abs(high[key]-low[key]) for key in ("mean", "variance")),
                              "beta_zero_leakage": moments(h, 128, beta=0.)["leakage_squared"],
                              "endpoint_jump": float(symmetric_g(full.limit(h), h)-symmetric_g(0., h))})
        for offset in (-.01, .01):
            q = transfer.geometry_lengths(h)
            q[Y] += offset
            for kind in ("length", "squared"):
                for beta in (0., 5.):
                    operators.append({"h": h, "y_offset": offset, "asymmetric": False,
                                      **operator_check(q, kind, beta=beta)})
    q = transfer.geometry_lengths(1.)
    q[Y] += .01
    q[moves.FINAL.edge_index[0, 2]] *= 1.02
    for kind in ("length", "squared"):
        operators.append({"h": 1., "y_offset": .01, "asymmetric": True, **operator_check(q, kind)})
    coarse, refined = fourier_check(), fourier_check(grid=2**18)
    grid_difference = max(abs(a["derivative_sum_over_cutoff"]-b["derivative_sum_over_cutoff"])
                          for a, b in zip(coarse["rows"], refined["rows"]))
    dependencies = ("regge_quantum_constraint_transfer.py", "regge_coalescing_kernel.py",
                    "regge_internal_marginal.py", "regge_pachner_constraints.py",
                    "regge_pachner_creation.py", "regge_pachner_transport.py", "regge_tent_transfer.py")
    return {"status": "[산출]", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in dependencies},
            "geometry": geometry, "distributions": distributions, "operators": operators,
            "fourier": refined, "fourier_grid_difference": grid_difference,
            "darboux": [darboux_branches(h) for h in (.5, 1.)],
            "scope": "공급된 주기 F 영모드에 대한 사후 C의 축약·누설 및 국소 정준 가지",
            "unfinished": ["전역 연산자 도메인과 가지 접합, 실제 양자화의 물리 선택",
                           "미시 작용·측도·공통 계량·GR·암흑부문·허블 텐션 연결"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"gradient_error": max(row["gradient_error"] for row in report["geometry"]),
                      "operator_error": max(row["operator_error"] for row in report["operators"]),
                      "original_c_error": max(row["original_c_error"] for row in report["operators"]),
                      "quadrature_difference": max(row["quadrature_difference"] for row in report["distributions"]),
                      "fourier_grid_difference": report["fourier_grid_difference"],
                      "h1": report["distributions"][2]}, ensure_ascii=False))

"""실제 레게 사전 제약의 양자 전달과 전체 구간의 끝점 항을 대조한다.

공유 길이, 양의 내적 밀도, 연산자 도메인과 초기 상태는 공급 조건이다.
제약과 상태를 같은 표현으로 옮기는 것만으로 물리 측도를 선택하지 않는다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import regge_coalescing_kernel as full
import regge_pachner_constraints as moves
from regge_pachner_creation import OLD_LOCAL_EDGES, admissible_interval

E = moves.E_ID
N = len(moves.FINAL.edges)


def geometry_lengths(h):
    full.limit(h)
    points = moves.reference_points()
    points[0, 0], points[1, 0] = -h, h
    return moves.FINAL.lengths(points)


def fine_interval(lengths):
    """상위 두 단체 조건을 넣지 않고 실제 네 그램 구간만 교차한다."""
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape != (N,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("열다섯 길이는 유한한 양수여야 한다")
    values = dict(zip(moves.FINAL.edges, lengths))
    intervals = []
    for cell in moves.FINAL.cells:
        vertices = (*sorted(set(cell)-{0, 1}), 0, 1)
        local = [values[tuple(sorted((vertices[i], vertices[j])))]
                 for i, j in OLD_LOCAL_EDGES]
        intervals.append(admissible_interval(local))
    lower, upper = max(a for a, _ in intervals), min(b for _, b in intervals)
    if lower >= upper:
        raise ValueError("실제 네 단체의 공통 허용 구간이 없다")
    return lower, upper


def measure(edge, interval, kind):
    lower, upper = interval
    if kind == "length":
        return np.ones_like(np.asarray(edge))/(upper-lower), np.zeros_like(np.asarray(edge))
    if kind == "squared":
        return 2*np.asarray(edge)/(upper*upper-lower*lower), 1/np.asarray(edge)
    raise ValueError("길이 또는 길이제곱 밀도를 지정해야 한다")


def sample_rule(interval, order):
    if isinstance(order, bool) or not isinstance(order, int) or order < 32:
        raise ValueError("구적 차수는 32 이상 정수여야 한다")
    angle, weight = full.angular_rule(order)
    lower, upper = interval
    edge = lower+(upper-lower)*np.sin(angle)
    return edge, weight*(upper-lower)*np.cos(angle)


def phase_data(lengths, edge):
    q = lengths.copy()
    q[E] = edge
    data = moves.actions(q)
    return data["second"]["action"], data["second"]["gradient"][E]


def wave(lengths, edge, kind, mode=0, twist=0., beta=5., opposite=False):
    interval = fine_interval(lengths)
    lower, upper = interval
    width = upper-lower
    rho, _ = measure(edge, interval, kind)
    phase, _ = phase_data(lengths, edge)
    sign = 1 if opposite else -1
    phi = np.exp(1j*(2*math.pi*mode+twist)*(edge-lower)/width)/math.sqrt(width)
    return np.exp(sign*1j*beta*phase)*phi/np.sqrt(rho)


def operator_check(lengths, kind, fraction=.43, mode=1, twist=0., beta=5., step=2e-5):
    """원래 B의 기울기와 실제 파동함수 차분을 비교한다."""
    lower, upper = fine_interval(lengths)
    edge = lower+fraction*(upper-lower)
    if not np.isfinite(step) or step <= 0 or not lower+step < edge < upper-step:
        raise ValueError("차분점과 간격은 허용 구간 내부여야 한다")
    def derivative(opposite):
        def central(delta):
            return (wave(lengths, edge+delta, kind, mode, twist, beta, opposite)
                    -wave(lengths, edge-delta, kind, mode, twist, beta, opposite))/(2*delta)
        return (4*central(step/2)-central(step))/3
    psi = wave(lengths, edge, kind, mode, twist, beta)
    wrong = wave(lengths, edge, kind, mode, twist, beta, True)
    _, score = measure(edge, (lower, upper), kind)
    _, gradient = phase_data(lengths, edge)
    frequency = (2*math.pi*mode+twist)/(upper-lower)
    action = -1j*(derivative(False)+.5*score*psi)+beta*gradient*psi
    wrong_action = -1j*(derivative(True)+.5*score*wrong)+beta*gradient*wrong
    return {"operator_error": float(abs(action-frequency*psi)),
            "omitted_connection": float(abs(.5*score*psi)),
            "opposite_phase_residual": float(abs(wrong_action-frequency*wrong)),
            "opposite_phase_formula_error": float(abs(wrong_action-(frequency+2*beta*gradient)*wrong))}


def transfer_check(lengths, kind, order=192, beta=5.):
    interval = fine_interval(lengths)
    lower, upper = interval
    width = upper-lower
    edge, weight = sample_rule(interval, order)
    rho, score = measure(edge, interval, kind)
    phase = np.array([phase_data(lengths, value)[0] for value in edge])
    phase_weight = np.exp(1j*beta*phase)
    kernel = 1/np.sqrt(width*rho)
    rows = []
    for mode, twist in ((0, 0.), (1, 0.), (0, math.pi)):
        frequency = (2*math.pi*mode+twist)/width
        phi = np.exp(1j*(2*math.pi*mode+twist)*(edge-lower)/width)/math.sqrt(width)
        psi = np.conj(phase_weight)*phi/np.sqrt(rho)
        amplitude = np.dot(weight, rho*kernel*phase_weight*psi)
        force = frequency*amplitude
        boundary = -1j*(np.exp(1j*twist)-1)/width
        rows.append({"mode": mode, "twist": twist, "amplitude": full.pair(amplitude),
                     "norm": float(np.dot(weight, rho*np.abs(psi)**2)),
                     "constraint_amplitude": full.pair(force),
                     "boundary_error": float(abs(force-boundary)),
                     "boundary_magnitude": float(abs(boundary))})
    phi = math.sqrt(2/width)*np.sin(math.pi*(edge-lower)/width)
    derivative = math.sqrt(2/width)*math.pi/width*np.cos(math.pi*(edge-lower)/width)
    # k=1인 대조. 파동함수의 끝점은 0이지만 밀도 미분항은 남는다.
    wrong_force = np.dot(weight, -1j*np.sqrt(rho)*derivative)
    bulk = np.dot(weight, .5j*np.sqrt(rho)*score*phi)
    return {"rows": rows, "wrong_kernel_force": full.pair(wrong_force),
            "wrong_kernel_bulk": full.pair(bulk),
            "wrong_kernel_identity_error": float(abs(wrong_force-bulk))}


def action_h(edge, h):
    """슐레플리 항등식으로 각도 미분을 소거한 경계 높이 미분."""
    edge, _, _, spoke, _ = full.geometry(edge, h)
    return (4*edge*h/np.sqrt(4*(h*h+1)-edge**2)*full.deficit(edge, h)
            +24*h/math.sqrt(6*h*h+2)*(math.pi-2*spoke))


def kernel_identity(h, beta, kind, order=192, step=2e-5):
    """전체 커널의 내부 제약 삽입과 움직이는 경계 미분을 계산한다."""
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(step) or step <= 0 or h <= step:
        raise ValueError("beta와 높이 차분 간격을 확인해야 한다")
    upper = full.limit(h)
    upper_h = 4*h/upper
    edge, weight = sample_rule((0., upper), order)
    rho = full.density(edge, h, kind)
    rho_e = np.zeros_like(edge) if kind == "length" else np.full_like(edge, 2/upper**2)
    score_h = -upper_h/upper*(1 if kind == "length" else 2)
    phase = np.exp(1j*beta*full.action(edge, h))
    integral = np.dot(weight, rho*phase)
    force = np.dot(weight, rho*full.gradient(edge, h)*phase)
    bulk = np.dot(weight, rho_e*phase)
    endpoint = (full.density(upper, h, kind)*np.exp(1j*beta*full.action(upper, h))
                -full.density(0., h, kind)*np.exp(1j*beta*full.action(0., h)))
    interior_h = np.dot(weight, rho*(score_h+1j*beta*action_h(edge, h))*phase)
    boundary_h = upper_h*full.density(upper, h, kind)*np.exp(1j*beta*full.action(upper, h))
    def central(delta):
        plus, _ = full.integral(beta, h+delta, kind, order=order)
        minus, _ = full.integral(beta, h-delta, kind, order=order)
        return (plus-minus)/(2*delta)
    derivative = (4*central(step/2)-central(step))/3
    return {"h": h, "beta": beta, "kind": kind, "kernel": full.pair(integral),
            "force": full.pair(force), "bulk": full.pair(bulk), "endpoint": full.pair(endpoint),
            "ward_error": float(abs(1j*beta*force+bulk-endpoint)),
            "derivative_h": full.pair(derivative),
            "derivative_error": float(abs(derivative-interior_h-boundary_h)),
            "omitted_boundary_error": float(abs(derivative-interior_h)),
            "boundary_h_magnitude": float(abs(boundary_h))}


def run():
    operators, transfers, identities, geometry = [], [], [], []
    for h in (.5, .9, 1., 1.1):
        q = geometry_lengths(h)
        interval = fine_interval(q)
        shape_errors = []
        for fraction in (.23, .51, .77):
            edge = interval[0]+fraction*(interval[1]-interval[0])
            current = q.copy()
            current[E] = edge
            data = moves.FINAL.evaluate(current)
            spokes = [i for i, pair in enumerate(moves.FINAL.edges)
                      if len(set(pair) & {0, 1}) == 1]
            independent_h = np.sum(data["gradient"][spokes])*h/math.sqrt(h*h+1)
            shape_errors.append(abs(independent_h-action_h(edge, h)))
        geometry.append({"h": h, "interval": list(interval),
                         "interval_error": float(max(abs(interval[0]), abs(interval[1]-full.limit(h)))),
                         "action_h_error": float(max(shape_errors))})
        for kind in ("length", "squared"):
            operators.append({"h": h, "kind": kind, **operator_check(q, kind)})
            transfers.append({"h": h, "kind": kind, **transfer_check(q, kind)})
            for beta in (0., 1., 5., 20.):
                result = kernel_identity(h, beta, kind)
                higher = kernel_identity(h, beta, kind, order=384)
                result["refined_ward_error"] = higher["ward_error"]
                result["refined_derivative_error"] = higher["derivative_error"]
                result["order_difference"] = float(abs(complex(*result["kernel"])-complex(*higher["kernel"])))
                identities.append(result)
    return {"status": "[산출]",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_coalescing_kernel.py", "regge_internal_marginal.py",
                                          "regge_pachner_constraints.py", "regge_pachner_creation.py",
                                          "regge_pachner_transport.py", "regge_tent_transfer.py")},
            "geometry": geometry, "operators": operators, "transfers": transfers, "identities": identities,
            "scope": "원래 사전 제약 하나의 조건부 전달과 실제 전체 구간의 끝점 항; 전체 제약 대수·물리 측도 유도 아님",
            "unfinished": ["생성 사후 제약을 포함한 전체 양자 제약의 공통 도메인과 전달",
                           "물리 내적·경계조건·준비·측도·공통 계량·전체 GR 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"operator_error": max(r["operator_error"] for r in report["operators"]),
                      "ward_error": max(r["refined_ward_error"] for r in report["identities"]),
                      "derivative_error": max(r["refined_derivative_error"] for r in report["identities"]),
                      "order_difference": max(r["order_difference"] for r in report["identities"])},
                     ensure_ascii=False))


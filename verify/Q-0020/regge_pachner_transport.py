"""움직이는 생성 구간에서 열네 경계 정준 운동량의 조건부 전달을 검산한다.

무차원 촐레스키 좌표 u와 측도 du dy를 공급한다. 전달한 정준 대수가 실제
레게의 모든 중력 제약과 같다고 전제하지 않는다. 주기 경계조건은 공급값이며
곡률의 준비 평균은 새 제약의 게이지 불변 물리 예측이 아니다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from regge_pachner_creation import PachnerCreation, reference_points


MOVE = PachnerCreation()


def chart(u):
    """공유 사면체와 그 양쪽 꼭짓점을 열네 실수로 매개화한다."""
    u = np.asarray(u, dtype=float)
    if u.shape != (14,) or not np.all(np.isfinite(u)):
        raise ValueError("좌표 열네 개가 유한해야 한다")
    with np.errstate(over="ignore", under="ignore"):
        positive = np.exp(u[[0, 1, 2, 9, 13]])
    if not np.all(np.isfinite(positive)) or np.any(positive <= 0):
        raise ValueError("지수화한 길이가 수치 표현 범위를 벗어났다")
    lower = np.array([[positive[0], 0, 0], [u[3], positive[1], 0],
                      [u[4], u[5], positive[2]]])
    points = np.zeros((6, 4))
    points[[1, 2, 5], :3] = lower
    points[3, :3], points[3, 3] = u[6:9], positive[3]
    points[4, :3], points[4, 3] = u[10:13], -positive[4]
    derivative = np.zeros((14, 6, 4))
    for parameter, vertex, component in ((0, 1, 0), (1, 2, 1), (2, 5, 2)):
        derivative[parameter, vertex, component] = positive[parameter]
    for parameter, vertex, component in ((3, 2, 0), (4, 5, 0), (5, 5, 1)):
        derivative[parameter, vertex, component] = 1
    for vertex, start, sign in ((3, 6, 1), (4, 10, -1)):
        for component in range(3):
            derivative[start+component, vertex, component] = 1
        derivative[start+3, vertex, 3] = sign*math.exp(u[start+3])
    lengths, jacobian = [], []
    for i, j in MOVE.old.edges:
        delta = points[i]-points[j]
        length = np.linalg.norm(delta)
        lengths.append(length)
        jacobian.append((derivative[:, i]-derivative[:, j]) @ delta/length)
    lengths, jacobian = np.array(lengths), np.array(jacobian)
    MOVE.old.evaluate(lengths)
    return {"points": points, "lengths": lengths, "jacobian": jacobian}


def inverse_chart(lengths):
    """공유 사면체 그램 행렬과 두 양의 높이로 원래 차트 좌표를 복원한다."""
    MOVE.old.evaluate(lengths)
    squared = dict(zip(MOVE.old.edges, np.asarray(lengths, dtype=float)**2))
    vertices = (1, 2, 5)
    def square(i, j):
        return 0 if i == j else squared[tuple(sorted((i, j)))]
    gram = np.array([[(square(0, i)+square(0, j)-square(i, j))/2
                      for j in vertices] for i in vertices])
    lower = np.linalg.cholesky(gram)
    u = np.zeros(14)
    u[:3] = np.log(np.diag(lower))
    u[3:6] = lower[1, 0], lower[2, 0], lower[2, 1]
    for vertex, start in ((3, 6), (4, 10)):
        inner = np.array([(square(0, vertex)+square(0, i)-square(i, vertex))/2 for i in vertices])
        projection = np.linalg.solve(lower, inner)
        height_squared = square(0, vertex)-projection @ projection
        if height_squared <= 0:
            raise ValueError("꼭짓점 높이가 양수가 아니다")
        u[start:start+3], u[start+3] = projection, .5*math.log(height_squared)
    return u


def reference_coordinates():
    return inverse_chart(MOVE.old.lengths(reference_points()))


def squared_bounds(u):
    """밑면이 xy 평면인 차트에서 구간 끝점의 제곱과 해석적 미분을 준다."""
    data = chart(u)
    points = data["points"]
    difference = points[3, :2]-points[4, :2]
    h3, h4 = np.linalg.norm(points[3, 2:]), np.linalg.norm(points[4, 2:])
    dp, dh3, dh4 = np.zeros(14), np.zeros(14), np.zeros(14)
    dp[6:8], dp[10:12] = 2*difference, -2*difference
    dh3[8], dh3[9] = points[3, 2]/h3, points[3, 3]**2/h3
    dh4[12], dh4[13] = points[4, 2]/h4, points[4, 3]**2/h4
    a_squared = float(difference @ difference+(h3-h4)**2)
    b_squared = float(difference @ difference+(h3+h4)**2)
    da = dp+2*(h3-h4)*(dh3-dh4)
    db = dp+2*(h3+h4)*(dh3+dh4)
    return {**data, "A": a_squared, "B": b_squared, "D": 4*h3*h4,
            "dA": da, "dB": db, "dD": 4*(dh3*h4+h3*dh4)}


def fields(u, y, coordinate="squared"):
    """움직이는 구간의 접선 벡터장과 반발산항, 작용의 미분을 준다."""
    data = squared_bounds(u)
    if not np.isfinite(y) or y <= 0 or not data["A"] < y*y < data["B"]:
        raise ValueError("새 길이는 열린 생성 구간 안에 있어야 한다")
    if coordinate == "squared":
        t = (y*y-data["A"])/data["D"]
        speed = data["D"]/(2*y)
        speed_y = -speed/y
        horizontal = (data["dA"]+t*data["dD"])/(2*y)
        divergence = data["dD"]/data["D"]-horizontal/y
        speed_u = data["dD"]/(2*y)
    elif coordinate == "length":
        if data["A"] <= 1e-24*data["D"]:
            raise ValueError("길이 좌표의 아래끝에서 일반 미분은 정의되지 않는다")
        a, b = math.sqrt(data["A"]), math.sqrt(data["B"])
        da, db = data["dA"]/(2*a), data["dB"]/(2*b)
        speed, speed_y = b-a, 0.0
        speed_u = db-da
        t = (y-a)/speed
        horizontal = da+t*speed_u
        divergence = speed_u/speed
    else:
        raise ValueError("길이 또는 길이제곱 좌표만 지원한다")
    geometry = MOVE.evaluate(data["lengths"], y)
    phase_u = data["jacobian"].T @ geometry["gradient"][MOVE.old_ids]
    phase_y = geometry["gradient"][MOVE.new_id]
    return {**data, "t": float(t), "speed": float(speed), "speed_y": float(speed_y),
            "speed_u": speed_u, "horizontal": horizontal, "divergence": divergence,
            "phase": geometry["increment"], "phase_u": phase_u, "phase_y": float(phase_y),
            "curvature": geometry["curvature"]}


def source_state(u, t, center, excited=True):
    """열네 좌표의 가우스와 주기 섬유 모드의 값을 검사점에서 계산한다."""
    delta = np.asarray(u)-center
    wave_numbers = np.linspace(-.4, .6, 14)
    parent = np.exp(-delta @ delta/4+1j*wave_numbers @ delta)
    oscillation = .3*np.exp(2j*math.pi*t) if excited else 0j
    value = parent*(1+oscillation)
    return value, (-delta/2+1j*wave_numbers)*value, parent*2j*math.pi*oscillation


def wave(u, y, center, coordinate="squared", excited=True):
    data = fields(u, y, coordinate)
    source, _, _ = source_state(u, data["t"], center, excited)
    return np.exp(1j*data["phase"])/math.sqrt(data["speed"])*source


def transport_check(u, t=.43, coordinate="squared", step=2e-5, excited=True):
    """표적 좌표에서 실제 파동함수를 차분하여 열다섯 공액 관계를 대조한다."""
    if not np.isfinite(step) or step <= 0 or not np.isfinite(t) or not 0 < t < 1:
        raise ValueError("양의 유한 차분 간격과 내부 섬유 좌표가 필요하다")
    bounds = squared_bounds(u)
    y = (math.sqrt(bounds["A"]+bounds["D"]*t) if coordinate == "squared" else
         math.sqrt(bounds["A"])+(math.sqrt(bounds["B"])-math.sqrt(bounds["A"]))*t)
    data = fields(u, y, coordinate)
    center = reference_coordinates()
    value = wave(u, y, center, coordinate, excited)
    def derivatives(delta):
        old = np.array([(wave(u+np.eye(14)[i]*delta, y, center, coordinate, excited)-
                         wave(u-np.eye(14)[i]*delta, y, center, coordinate, excited))/(2*delta)
                        for i in range(14)])
        new = (wave(u, y+delta, center, coordinate, excited)-wave(u, y-delta, center, coordinate, excited))/(2*delta)
        return old, new
    first, second = derivatives(step), derivatives(step/2)
    old_derivative, new_derivative = (4*second[0]-first[0])/3, (4*second[1]-first[1])/3
    horizontal, divergence = data["horizontal"], data["divergence"]
    old_action = -1j*(old_derivative+horizontal*new_derivative+divergence*value/2)
    old_action -= (data["phase_u"]+horizontal*data["phase_y"])*value
    new_action = -1j*(data["speed"]*new_derivative+data["speed_y"]*value/2)
    new_action -= data["speed"]*data["phase_y"]*value
    _, expected_old, expected_new = source_state(u, data["t"], center, excited)
    factor = np.exp(1j*data["phase"])/math.sqrt(data["speed"])
    expected_old, expected_new = -1j*factor*expected_old, -1j*factor*expected_new
    vector_derivative = np.column_stack([(fields(u+np.eye(14)[i]*step, y, coordinate)["horizontal"]-
                                          fields(u-np.eye(14)[i]*step, y, coordinate)["horizontal"])/(2*step)
                                         for i in range(14)])
    commutator = vector_derivative.T-vector_derivative+np.outer(horizontal, divergence)-np.outer(divergence, horizontal)
    cross = data["speed_u"]+horizontal*data["speed_y"]-data["speed"]*divergence
    curvature_derivative = (MOVE.evaluate(data["lengths"], y+step)["curvature"]-
                            MOVE.evaluate(data["lengths"], y-step)["curvature"])/(2*step)
    return {"coordinate": coordinate, "excited": excited, "t": t,
            "old_momentum_residual": float(np.linalg.norm(old_action-expected_old)),
            "new_momentum_residual": float(abs(new_action-expected_new)),
            "old_vector_commutator": float(np.linalg.norm(commutator)),
            "cross_vector_commutator": float(np.linalg.norm(cross)),
            "omitted_half_divergence_old_error": float(np.linalg.norm(old_action+.5j*divergence*value-expected_old)),
            "omitted_half_divergence_new_error": float(abs(new_action+.5j*data["speed_y"]*value-expected_new)),
            "curvature_constraint_commutator_coefficient": float(data["speed"]*curvature_derivative)}


def preparation_comparison(order=48):
    """좌표별 상수 준비와 동일 상태의 수동 좌표변환을 분리한다."""
    if not isinstance(order, int) or not 8 <= order <= 64:
        raise ValueError("구적 차수는 8 이상 64 이하의 정수여야 한다")
    u = reference_coordinates()
    bounds = squared_bounds(u)
    length = math.sqrt(bounds["B"])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = (nodes+1)*math.pi/4
    y = length*np.sin(angle)**2
    dy = weights*math.pi/4*2*length*np.sin(angle)*np.cos(angle)
    geometry = [MOVE.evaluate(bounds["lengths"], value) for value in y]
    phase = np.exp(1j*np.array([item["increment"] for item in geometry]))
    curvature = np.array([item["curvature"] for item in geometry])
    length_state = phase/math.sqrt(length)
    squared_state = phase*np.sqrt(2*y/bounds["D"])
    speed = bounds["D"]/(2*y)
    dt = dy/speed
    passive_state = np.sqrt(speed)*phase.conj()*length_state
    mean_length = float(np.dot(dy, abs(length_state)**2*curvature))
    mean_squared = float(np.dot(dy, abs(squared_state)**2*curvature))
    return {"length_norm": float(np.dot(dy, abs(length_state)**2)),
            "squared_norm": float(np.dot(dy, abs(squared_state)**2)),
            "length_curvature_mean": mean_length, "squared_curvature_mean": mean_squared,
            "overlap": float(np.dot(dy, (length_state.conj()*squared_state)).real),
            "passive_norm": float(np.dot(dt, abs(passive_state)**2)),
            "passive_curvature_residual": float(abs(np.dot(dt, abs(passive_state)**2*curvature)-mean_length))}


def run():
    reference = reference_coordinates()
    generic = reference+np.linspace(-.13, .17, 14)
    cases = [transport_check(reference, excited=False), transport_check(reference, excited=True),
             transport_check(generic, coordinate="squared"), transport_check(generic, coordinate="length")]
    delta = 1e-4
    middle = squared_bounds(reference)
    left = squared_bounds(reference-np.eye(14)[6]*delta)
    right = squared_bounds(reference+np.eye(14)[6]*delta)
    comparison = preparation_comparison()
    fine = preparation_comparison(64)
    return {"status": "[산출]", "scope": "공급한 두 좌표·곱 측도의 경계 정준 전달과 운동학적 준비 대조",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_pachner_creation.py", "regge_tent_transfer.py")},
            "reference_coordinates": reference.tolist(),
            "chart_rank": int(np.linalg.matrix_rank(chart(reference)["jacobian"])),
            "chart_smallest_singular": float(np.linalg.svd(chart(reference)["jacobian"], compute_uv=False)[-1]),
            "cases": cases,
            "cusp": {"left_derivative": (math.sqrt(middle["A"])-math.sqrt(left["A"]))/delta,
                     "right_derivative": (math.sqrt(right["A"])-math.sqrt(middle["A"]))/delta,
                     "squared_endpoint_central_derivative": (right["A"]-left["A"])/(2*delta)},
            "preparation_comparison": comparison,
            "quadrature_convergence": max(abs(comparison[k]-fine[k]) for k in comparison),
            "assumptions": ["du dy를 공급하며 평탄 dq dy에는 |det(dq/du)|^(-1/2)를 추가해야 한다",
                            "주기 섬유 운동량은 서로 다른 퇴화 끝점을 인위적으로 식별하는 공급 도메인이다",
                            "자기수반 도메인은 원래 곱 공간 도메인의 단위 사상 영상으로 정의한다"],
            "negative_controls": ["길이 구간의 아래끝 a=0에서는 일반 길이 좌표 미분을 거부한다",
                                  "반발산항을 빼면 실제 열네 운동량 전달 잔차가 사라지지 않는다",
                                  "새 제약을 게이지로 취급하면 곡률은 디랙 관측량이 아니다",
                                  "동일 상태·관측량을 함께 옮기면 평균은 같다; 좌표별 상수 준비는 다른 준비 선택이다"],
            "unfinished": ["전달한 정준 대수와 실제 레게 전체 중력 제약의 동일성 및 비선택 상태의 물리 준비",
                           "물리 측도·방출·환류·환경 에너지와 공통 계량 선택 및 0D에서 3+1 중력으로 가는 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))

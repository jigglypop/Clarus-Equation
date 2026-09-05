"""실제 공동 레게 제약의 에너지 적합성과 퇴화면의 접선 흐름을 검산한다.

시간, 에너지 척도와 후보 에너지는 공급한다. 매끄러운 흐름의 계수 보존을
이산 파흐너 이동의 금지나 물리 중력 에너지의 유도로 확대하지 않는다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import regge_constraint_reduction as reduction
import regge_postconstraint_projection as previous

N, E, Y = reduction.N, reduction.E, reduction.Y
REST, POISSON = reduction.REST, reduction.OMEGA
moves = reduction.moves


def derivative(function, q, index, step):
    """서로 다른 두 간격의 중심 차분으로 4차 외삽을 만든다."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("차분 간격은 유한한 양수여야 한다")
    delta = np.eye(N)[index] * step
    coarse = (function(q + delta) - function(q - delta)) / (2 * step)
    fine = (function(q + delta / 2) - function(q - delta / 2)) / step
    return (4 * fine - coarse) / 3


def g(q):
    # 초기 복합체에는 y가 없으므로 S_y는 최종 작용의 y 미분과 같다.
    return float(moves.FINAL.evaluate(q)["gradient"][Y])


def bracket(q, beta=1., inner_step=1e-4):
    if not np.isfinite(beta) or beta == 0:
        raise ValueError("이 제2종 퇴화 검사는 유한한 비영 beta를 요구한다")
    return float(-beta * derivative(g, q, E, inner_step))


def bracket_gradient(q, beta=1., indices=None, outer_step=.001, inner_step=1e-4):
    indices = range(N) if indices is None else indices
    return np.array([derivative(
        lambda x: bracket(x, beta, inner_step), q, i, outer_step) for i in indices])


def joint_jacobian(q, beta=1., step=2e-5):
    jac = reduction.constraint_jacobian(q, step)
    jac[:, :N] *= beta
    return jac


def full_square_flow(q, beta=1., outer_step=.001, inner_step=1e-4):
    """양의 제곱 후보를 원래 30차원 공간에서 구성하고 독립 접선식과 비교한다."""
    a = bracket(q, beta, inner_step)
    aq = bracket_gradient(q, beta, outer_step=outer_step, inner_step=inner_step)
    jac = joint_jacobian(q, beta)
    gradient = np.r_[a * aq, np.zeros(N)]
    # u=-a_e, v=a_y는 a=0에서도 매끄럽다.
    multipliers = np.array([-aq[E], aq[Y]])
    vector = POISSON @ (gradient + jac.T @ multipliers)
    qdot = vector[:N]
    data = moves.actions(q)

    # 제약면 삽입의 독립 방향 차분: 두 제약 운동량의 변화.
    def constrained_momenta(x):
        values = moves.actions(x)
        return beta * np.array([values["first"]["gradient"][Y],
                                 -values["second"]["gradient"][E]])
    direction_step = 2e-6 / max(1., np.linalg.norm(qdot))
    coarse = (constrained_momenta(q + direction_step*qdot) -
              constrained_momenta(q - direction_step*qdot)) / (2*direction_step)
    half = direction_step / 2
    fine = (constrained_momenta(q + half*qdot) -
            constrained_momenta(q - half*qdot)) / (2*half)
    tangent_p = (4*fine - coarse) / 3

    # 임의의 남은 운동량도 원래 정준 포아송 식으로 진화시킨다.
    b_hessian_direction = (moves.actions(q+direction_step*qdot)["second"]["gradient"] -
                           moves.actions(q-direction_step*qdot)["second"]["gradient"]) / (2*direction_step)
    pi_dot = vector[N+REST] + beta*b_hessian_direction[REST]
    gz = np.array([derivative(g, q, i, 2e-5) for i in REST])
    pi_expected = beta*gz*qdot[Y] - gradient[REST]
    return {
        "a": a, "a_e": float(aq[E]), "a_y": float(aq[Y]),
        "energy": .5*a*a,
        "multipliers": multipliers.tolist(),
        "qdot_e_y": qdot[[E, Y]].tolist(),
        "configuration_speed": float(np.linalg.norm(qdot)),
        "constraint_rate_residual": float(np.max(np.abs(jac @ vector))),
        "energy_rate_residual": float(abs(gradient @ vector)),
        "a_rate_residual": float(abs(aq @ qdot)),
        "momentum_tangent_residual": float(np.max(np.abs(tangent_p-vector[N+np.array([Y,E])]))),
        "shifted_rest_momentum_residual": float(np.max(np.abs(pi_dot-pi_expected))),
        "omitted_shifted_momentum_defect": float(np.max(np.abs(pi_expected))),
        "minimum_gram": float(data["final"]["minimum_gram_eigenvalue"]),
    }


def candidate_check(q, beta=1., length_scale=None):
    if length_scale is None:
        length_scale = previous.full.limit(1.)
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("기준 길이는 유한한 양수여야 한다")
    a = bracket(q, beta)
    gy = float(derivative(g, q, Y, 1e-4))
    jac = joint_jacobian(q, beta)
    pair = jac @ POISSON @ jac.T
    rows = []
    for name, he, hy in (("length", 1/length_scale, 0.),
                         ("action_gradient", -a, beta*gy)):
        obstruction = np.array([-hy, -he])
        # 수치상 작은 a를 역수로 쓰지 않고 정확한 퇴화 조건을 별도 보고한다.
        regular = abs(a) > 1e-7
        multipliers = np.array([obstruction[1]/a, -obstruction[0]/a]) if regular else None
        rows.append({
            "candidate": name, "constraint_obstruction": obstruction.tolist(),
            "multipliers": None if multipliers is None else multipliers.tolist(),
            "consistency_residual": None if multipliers is None else float(
                np.max(np.abs(obstruction + pair @ multipliers))),
        })
    return {"a": a, "g": g(q), "g_y": gy, "cases": rows}


def trajectory(q, beta=1., duration=.002, steps=8):
    """구성 변수의 짧은 RK4 궤도. 운동량과 전체 보존식은 별도 점별 검사다."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("적분 단계는 양의 정수여야 한다")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("적분 구간은 유한한 양수여야 한다")
    q = np.asarray(q, dtype=float).copy()
    initial = q.copy()
    initial_a = bracket(q, beta)
    values = [initial_a]
    gram = [moves.FINAL.evaluate(q)["minimum_gram_eigenvalue"]]
    dt = duration / steps

    def rhs(x):
        ae, ay = bracket_gradient(x, beta, [E, Y])
        result = np.zeros(N)
        result[E], result[Y] = ay, -ae
        return result

    for _ in range(steps):
        k1 = rhs(q)
        k2 = rhs(q + .5*dt*k1)
        k3 = rhs(q + .5*dt*k2)
        k4 = rhs(q + dt*k3)
        q += dt*(k1+2*k2+2*k3+k4)/6
        values.append(bracket(q, beta))
        gram.append(moves.FINAL.evaluate(q)["minimum_gram_eigenvalue"])
    values = np.array(values)
    return {
        "steps": steps, "duration": duration, "beta": beta,
        "initial_a": initial_a, "final_a": float(values[-1]),
        "maximum_a_change": float(np.max(np.abs(values-initial_a))),
        "maximum_energy_change": float(.5*np.max(np.abs(values**2-initial_a**2))),
        "minimum_gram": float(min(gram)),
        "initial_e_y": initial[[E,Y]].tolist(), "final_e_y": q[[E,Y]].tolist(),
        "final_q": q.tolist(),
    }


def exact_fold_certificate():
    """유리 그램 행렬과 정확한 각도 미분으로 G_y의 음의 부호를 증명한다."""
    from itertools import combinations
    import sympy as sp

    points = sp.Matrix([[-1,0,0,0], [1,0,0,0], [0,1,1,1], [0,1,-1,-1],
                        [0,-1,1,-1], [0,-1,-1,1]])
    points[2:,1:] = points[2:,1:] / sp.sqrt(3)
    y = sp.sqrt(sp.Rational(8,3))
    sums = {}

    def normals(matrix):
        result = sp.zeros(5)
        result[1:,1:] = matrix
        for i in range(4):
            result[0,i+1] = result[i+1,0] = -sum(matrix[i,:])
        result[0,0] = sum(matrix)
        return result

    for cell in moves.FINAL.cells:
        distances, changes = sp.zeros(5), sp.zeros(5)
        for i, j in combinations(range(5), 2):
            delta = points[cell[i],:] - points[cell[j],:]
            distances[i,j] = distances[j,i] = sp.simplify(delta.dot(delta))
            if {cell[i], cell[j]} == {3,4}:
                changes[i,j] = changes[j,i] = 2*y
        gram = sp.Matrix(4,4,lambda i,j:
                         (distances[0,i+1]+distances[0,j+1]-distances[i+1,j+1])/2)
        dg = sp.Matrix(4,4,lambda i,j:
                       (changes[0,i+1]+changes[0,j+1]-changes[i+1,j+1])/2)
        inverse = gram.inv()
        normal, dn = normals(inverse), normals(-inverse*dg*inverse)
        for tri in combinations(cell, 3):
            if not {3,4}.issubset(tri):
                continue
            tri = tuple(sorted(tri))
            i, j = [k for k, vertex in enumerate(cell) if vertex not in tri]
            denominator = sp.sqrt(normal[i,i]*normal[j,j])
            cosine = sp.simplify(-normal[i,j]/denominator)
            dc = sp.simplify((-dn[i,j]+normal[i,j]*
                             (dn[i,i]/normal[i,i]+dn[j,j]/normal[j,j])/2)/denominator)
            dtheta = sp.simplify(-dc/sp.sqrt(1-cosine*cosine))
            sums.setdefault(tri, []).append((sp.acos(cosine), dtheta))

    rows, total = [], sp.S.Zero
    edge = sp.Symbol("edge", positive=True)
    for tri, terms in sums.items():
        other = next(vertex for vertex in tri if vertex not in (3,4))
        delta = points[other,:] - points[3,:]
        leg_squared = sp.simplify(delta.dot(delta))
        area = edge*sp.sqrt(4*leg_squared-edge*edge)/4
        ay = sp.simplify(sp.diff(area,edge).subs(edge,y))
        ayy = sp.simplify(sp.diff(area,edge,2).subs(edge,y))
        theta = sp.pi - sum(t[0] for t in terms)
        theta_y = -sum(t[1] for t in terms)
        term = sp.simplify(ayy*theta+ay*theta_y)
        total += term
        rows.append({"triangle":list(tri), "area_y":str(ay), "area_yy":str(ayy),
                     "exterior_angle":str(sp.simplify(theta)),
                     "exterior_angle_y":str(sp.simplify(theta_y)),
                     "term":str(term), "value":float(term)})
    theta_s = sp.pi-2*sp.acos(sp.sqrt(15)/5)
    theta_r = sp.pi-sp.acos(-sp.Rational(4,5))
    expected = -7*sp.sqrt(2)*theta_s/8-5*sp.sqrt(3)*theta_r/9-7*sp.sqrt(3)/30
    return {
        "rows":rows, "g_y_exact":str(expected), "g_y_value":float(expected),
        "symbolic_difference":str(sp.simplify(total-expected)),
        "strict_negative_bound":str(-7*sp.sqrt(3)/30),
        "sign_reason":"두 외재각이 양수이므로 G_y < -7*sqrt(3)/30 < 0",
    }


def symmetric_bracket_e(edge, height, beta=1.):
    """기존 정확한 G_e의 e 미분. y=Y0와 대칭 경계를 고정한다."""
    _, radius, _, _, _ = previous.full.geometry(edge, height)
    denominator = 9*radius*radius+8
    coefficient = 6*height*height-2
    return float(-4*math.sqrt(2)*beta*(
        edge/radius**3*(coefficient/denominator-1/3) +
        18*coefficient*edge/(radius*denominator**2)))


def run():
    points = []
    for height in (.9, 1., 1.1):
        q = reduction.flat_lengths(height)
        for beta in (1., 5.):
            coarse = full_square_flow(q, beta, outer_step=.002, inner_step=2e-4)
            fine = full_square_flow(q, beta)
            points.append({
                "height": height, "beta": beta,
                "flat_a_error": abs(fine["a"]-3*math.sqrt(2)*beta*(1-height**2)),
                "analytic_a_e_error": abs(fine["a_e"]-symmetric_bracket_e(q[E],height,beta)),
                "derivative_step_difference": max(abs(fine[key]-coarse[key]) for key in ("a_e","a_y")),
                "candidates": candidate_check(q, beta, previous.full.limit(height)),
                "square_flow": fine,
            })
    deformed = []
    for e_delta, y_delta, z_delta in ((-.02,.01,0.),(.02,-.01,0.),(.01,.01,.003)):
        q = reduction.flat_lengths(1.)
        q[E] += e_delta
        q[Y] += y_delta
        q[REST[0]] += z_delta
        flow = full_square_flow(q)
        independent = -moves.mixed_derivative(q)
        deformed.append({"offsets":[e_delta,y_delta,z_delta], "square_flow":flow,
                         "independent_action_hessian_error":abs(flow["a"]-independent)})
    curves = []
    for e_delta in (-.02, 0., .02):
        q = reduction.flat_lengths(1.)
        q[E] += e_delta
        coarse = trajectory(q, steps=4)
        fine = trajectory(q, steps=8)
        fine["endpoint_step_difference"] = float(np.max(np.abs(
            np.array(coarse["final_q"])-fine["final_q"])))
        del fine["final_q"]
        curves.append(fine)
    dependencies = ("regge_constraint_reduction.py", "regge_postconstraint_projection.py",
                    "regge_quantum_constraint_transfer.py", "regge_coalescing_kernel.py",
                    "regge_internal_marginal.py", "regge_pachner_constraints.py",
                    "regge_pachner_creation.py", "regge_pachner_transport.py", "regge_tent_transfer.py")
    return {
        "status":"[산출]", "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in dependencies},
        "scope":"기존 실제 공동 제약의 공급 에너지 적합성과 매끄러운 국소 흐름",
        "formulas":{
            "multipliers":"u={F,H}/a, v=-{c,H}/a",
            "configuration_normal_form":"H=H0(z)+a^2*k near a_e!=0",
            "square_flow":"e_dot=a_y, y_dot=-a_e, z_dot=0",
            "shifted_momentum":"Pi_dot=beta*G_z*y_dot-H_z",
            "rank_invariance":"smooth i_X omega_C=dH implies Phi_t^*omega_C=omega_C",
            "fold_a_e":"-6*sqrt(2)*beta at height=1,e=2,y=sqrt(8/3)",
        },
        "fold_certificate":exact_fold_certificate(),
        "flat_points":points, "deformed_points":deformed, "configuration_trajectories":curves,
        "unfinished":[
            "공급 에너지와 시간의 실제 미시 작용 유도",
            "계수 변화의 이산 정준 관계와 전체 양자 커널에서의 에너지 전달",
            "물리 내적·편극·장치 준비와 환류·0D 보충",
            "공통 계량·3+1 중력·전자기·약력·강력·암흑부문·허블 텐션",
        ],
    }


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"points":len(report["flat_points"])+len(report["deformed_points"]),
                      "trajectories":report["configuration_trajectories"]}, ensure_ascii=False))

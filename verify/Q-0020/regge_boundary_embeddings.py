"""실제 레게 경계에서 진폭 일관성의 위상 자유와 상태 변형을 대조한다.

두 조건부측도와 같은 경계의 원뿔 작용은 공급 입력이다. 작용을 상쇄하는
삽입은 물리적 진공의 유도가 아니며, 미분 대조를 중력 제약으로 해석하지 않는다.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import regge_boundary_pushforward as boundary
from regge_boundary_pushforward import base


def fiber_data(domain, f, *, order=128):
    """전체 허용 절단에서 기존 네 단체 작용과 두 조건부 구적을 구한다."""
    if not np.isfinite(f) or f <= 0:
        raise ValueError("경계 길이는 양의 유한 실수여야 한다")
    left, right = domain.fiber(f, 1)
    if not np.isfinite(left+right) or right <= left:
        raise ValueError("양의 폭을 가진 허용 절단이 필요하다")
    nodes, weights = base.rule(order)
    e = left+(right-left)*(nodes+1)/2
    action = base.OLD_ACTION(domain.lengths(e, f))
    conditional = {
        "length": weights/2,
        "squared": weights*e/(right+left),
    }
    return {"e": e, "action": action, "weights": conditional,
            "bounds": [float(left), float(right)]}


def phase_insertion(fine_action, coarse_action, beta):
    """선형 진폭 규약에서 상위 목표와 세밀한 작용의 차이를 곱한다."""
    return np.exp(1j*np.asarray(beta)*(np.asarray(coarse_action)-np.asarray(fine_action)))


def compare_fiber(domain, f, beta_values=(0, 1, 5, 20), *, order=256):
    data = fiber_data(domain, f, order=order)
    betas = boundary.phase_values(beta_values)
    action = data["action"]
    fine_phase = np.exp(1j*action[:, None]*betas)
    cones = [boundary.cone_completion(domain, f, c) for c in (1, 4)]
    result = {}
    for name, weights in data["weights"].items():
        kernel = weights @ fine_phase
        rows = []
        for cone in cones:
            target = np.exp(1j*cone["action"]*betas)
            j = phase_insertion(action[:, None], cone["action"], betas)
            amplitude = weights @ (fine_phase*j)
            distance = weights @ abs(j-1)**2
            formula = 2-2*(target*kernel.conj()).real
            # 같은 끝점을 갖는 두 원뿔 경로의 망원경 합일 뿐 새 측도족이 아니다.
            first_stage = np.exp(-1j*cone["action"]*betas)
            combined = j*first_stage
            direct = np.exp(-1j*action[:, None]*betas)
            # 베타 1에서 정한 삽입을 그대로 두고 베타 0의 P_0(v)=적분(v)에 넣는다.
            frozen = phase_insertion(action, cone["action"], 1.0)
            rows.append({
                "clearance": cone["clearance"], "coarse_action": cone["action"],
                "boundary_preserved": cone["boundary_preserved"],
                "minimum_gram_eigenvalue": cone["minimum_gram_eigenvalue"],
                "amplitude_error": float(np.max(abs(amplitude-target))),
                "norm_error": float(np.max(abs(weights @ abs(j)**2-1))),
                "distance_squared": distance.tolist(),
                "distance_formula_error": float(np.max(abs(distance-formula))),
                "composition_error": float(np.max(abs(combined-direct))),
                "natural_amplitude_error": abs(kernel-target).tolist(),
                "opposite_phase_error": abs(weights @ (fine_phase*j.conj())-target).tolist(),
                "frozen_beta1_error_at_beta0": float(abs(weights @ frozen-1)),
            })
        optimal_phase = np.exp(1j*np.angle(kernel))
        optimal_j = optimal_phase[None, :]*fine_phase.conj()
        measured_minimum = weights @ abs(optimal_j-1)**2
        predicted_minimum = 2*(1-abs(kernel))
        result[name] = {
            "kernel": boundary.complex_list(kernel),
            "minimum_distance_squared": measured_minimum.tolist(),
            "minimum_formula_error": float(np.max(abs(measured_minimum-predicted_minimum))),
            "cones": rows,
        }
    return {"f": f, "bounds": data["bounds"], "betas": betas.tolist(), "results": result}


def derivative_check(domain, f, *, beta=5.0, step=2e-5):
    """지지 내부의 국소 교환자를 직접 차분하고 원래 작용 기울기와 비교한다."""
    if not np.isfinite(beta) or not np.isfinite(step) or step <= 0:
        raise ValueError("유한 위상 계수와 양의 차분 간격이 필요하다")
    data = fiber_data(domain, f, order=16)
    left, right = data["bounds"]
    e = left+.43*(right-left)
    if not left < e-step < e+step < right:
        raise ValueError("차분점은 모두 같은 열린 절단 안에 있어야 한다")
    coarse = boundary.cone_completion(domain, f, 1)["action"]
    ids = base.WHOLE.indices(base.OLD.edges)
    geometry = base.OLD.evaluate(domain.lengths(e, f)[ids])
    gradient = float(geometry["gradient"][base.OLD.edges.index((0, 1))])

    def wave(x):
        action = float(base.OLD_ACTION(domain.lengths(x, f)))
        return phase_insertion(action, coarse, beta)

    first = (wave(e+step)-wave(e-step))/(2*step)
    second = (wave(e+step/2)-wave(e-step/2))/step
    actual = -1j*(4*second-first)/3
    expected = -beta*gradient*wave(e)
    # 제곱길이 측도의 형식 대칭항도 [D,J]에서는 정확히 상쇄된다.
    symmetric_commutator = actual-1j*wave(e)/(2*e)-wave(e)*(-1j/(2*e))
    return {"e": e, "f": f, "beta": beta, "old_action_gradient": gradient,
            "commutator_magnitude": float(abs(expected)),
            "derivative_error": float(abs(actual-expected)),
            "squared_measure_commutator_error": float(abs(symmetric_commutator-expected))}


def run():
    previous_path = Path(boundary.__file__).with_suffix(".json")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous["source_sha256"] != hashlib.sha256(Path(boundary.__file__).read_bytes()).hexdigest():
        raise ValueError("경계 전달 산출물과 현재 소스의 해시가 다르다")
    for name, digest in previous["dependencies"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError("경계 전달의 의존 파일이 변경되었다: "+name)
    cases = []
    for case in previous["cases"]:
        domain = base.Domain(case["boundary"])
        samples = []
        for f in (.7, .9, 1.1):
            sample = compare_fiber(domain, f, order=256)
            middle = compare_fiber(domain, f, order=128)
            lower = compare_fiber(domain, f, order=64)
            for name in ("length", "squared"):
                def difference(a, b):
                    kernels = np.array(a["kernel"])-b["kernel"]
                    return max(
                        float(np.max(np.linalg.norm(kernels, axis=1))),
                        float(np.max(abs(np.array(a["minimum_distance_squared"])
                                         -b["minimum_distance_squared"]))),
                        *(float(np.max(abs(np.array(x["distance_squared"])-y["distance_squared"])))
                          for x, y in zip(a["cones"], b["cones"])),
                    )
                sample["results"][name]["initial_64_128_error"] = difference(
                    middle["results"][name], lower["results"][name])
                sample["results"][name]["refined_128_256_error"] = difference(
                    sample["results"][name], middle["results"][name])
            sample["derivative"] = derivative_check(domain, f)
            samples.append(sample)
        cases.append({"boundary": case["boundary"], "samples": samples})
    names = ("regge_boundary_pushforward.py", "regge_boundary_pushforward.json",
             "regge_two_edge_composition.py", "regge_pachner_constraints.py",
             "regge_pachner_creation.py", "regge_tent_transfer.py")
    return {"status": "[산출]", "scope": "공급 조건부측도에서 실제 두 원뿔 목표의 삽입 자유와 강제 상태 변형",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in names},
            "cases": cases,
            "boundaries": [
                "OLD4 작용과 중간 경계 f를 사용한다. 최종 일곱 단체의 전체 작용으로 바꾸지 않는다",
                "f 표본의 조건부 검산이다. 모든 경계의 엄밀 구적 오차 인증이나 독립 세분화 측도족이 아니다",
                "조건부 Cauchy–Schwarz와 위상 망원경 합은 표준 수학이며 CE 신규 정리가 아니다",
                "독립 미분 대조는 콤팩트 지지 국소식이고 전체 중력 제약이나 자기수반 도메인을 유도하지 않는다"],
            "unfinished": ["물리 진공·측도·삽입의 독립 고정과 실제 제약의 양립",
                           "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "scope", "source_sha256")}, ensure_ascii=True))

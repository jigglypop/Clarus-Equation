"""외부 원천에 대한 실제 레게 정적 응답과 전체 유한 적분을 대조한다.

원천 j는 공급 공리다. 정적 민감도·복소 진폭을 물리 시간의 지연 응답이나
확률 평균으로 해석하지 않는다. 원래 초기 경계 운동량 준비를 함께 유지한다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import regge_coalescing_kernel as geometry
import regge_postconstraint_projection as projected
import regge_constraint_reduction as reduced

C2 = 24*math.sqrt(3)
CENTER = -5*math.sqrt(3)/144


def source_roots(source):
    """합류점 양쪽의 국소 정상 가지. 음의 원천은 부호 정리로 거부한다."""
    if not np.isfinite(source) or source <= 0:
        raise ValueError("두 국소 실근은 양의 유한 원천에서 검사한다")
    function = lambda e: float(geometry.gradient(e, 1.))-source
    return np.array([brentq(function, 1.9, 2., xtol=5e-15),
                     brentq(function, 2., 2.09, xtol=5e-15)])


def stationary_response(source):
    roots = source_roots(source)
    exact_response = np.array([1/geometry.hessian(e, 1.) for e in roots])
    step = source*1e-3
    coarse = (source_roots(source+step)-source_roots(source-step))/(2*step)
    fine = (source_roots(source+step/2)-source_roots(source-step/2))/step
    response = (4*fine-coarse)/3
    prediction = 2 + np.array([-1,1])*math.sqrt(source/C2) + CENTER*source
    bracket = -projected.symmetric_g_e(roots, 1.)
    direct, first_constraint, second_constraint = [], [], []
    for edge in roots:
        q = reduced.flat_lengths(1.)
        q[reduced.E] = edge
        data = reduced.moves.actions(q)
        direct.append(abs(data["final"]["gradient"][reduced.E]-source))
        old_p = np.zeros(reduced.N)
        old_p[reduced.moves.OLD_IDS] = data["old"]["gradient"]
        middle_p = old_p+data["first"]["gradient"]
        first_constraint.append(abs(middle_p[reduced.Y]-data["first"]["gradient"][reduced.Y]))
        second_constraint.append(abs(middle_p[reduced.E]+data["second"]["gradient"][reduced.E]-source))
    return {
        "source":source, "roots":roots.tolist(), "susceptibility":exact_response.tolist(),
        "source_derivative_relative_error":float(np.max(np.abs((response-exact_response)/exact_response))),
        "direct_regge_gradient_error":max(direct), "c_residual":max(first_constraint),
        "source_F_residual":max(second_constraint),
        "branch_formula_error":float(np.max(np.abs(roots-prediction))),
        "scaled_branch_remainder":float(np.max(np.abs(roots-prediction))/source**1.5),
        "center_shift_per_source":float((roots.mean()-2)/source),
        "center_coefficient_error":float(abs((roots.mean()-2)/source-CENTER)),
        "a_over_sqrt_source":(bracket/math.sqrt(source)).tolist(),
        "a_squared_over_source":(bracket**2/source).tolist(),
        "a_squared_ratio_error":float(np.max(np.abs(bracket**2/source-math.sqrt(3)))),
    }


def source_kernel(beta, source, kind, order=256):
    """고정 구간의 원래 진동 부호를 유지한 K와 원천 미분 0,1,2차."""
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(source):
        raise ValueError("beta는 음이 아닌 유한 수, 원천은 유한 실수여야 한다")
    angles, weights = geometry.angular_rule(order)
    upper = geometry.limit(1.)
    edge = upper*np.sin(angles)
    weights = weights*geometry.density(edge, 1., kind)*upper*np.cos(angles)
    phase = np.exp(1j*beta*(geometry.action(edge, 1.)-source*edge))
    return np.array([np.dot(weights, (-1j*beta*edge)**n*phase) for n in range(3)])


def kernel_check(beta, source, kind, order):
    exact = source_kernel(beta, source, kind, order)
    coarse = source_kernel(beta, source, kind, order//2)
    step = 1e-4/max(1., beta)
    def differences(delta):
        plus = source_kernel(beta, source+delta, kind, order)[0]
        minus = source_kernel(beta, source-delta, kind, order)[0]
        return np.array([(plus-minus)/(2*delta), (plus-2*exact[0]+minus)/delta**2])
    # 이계 차분은 반올림 오차가 커지는 지나치게 작은 간격을 피한다.
    second_step = 5e-3/max(1., beta)
    first = (4*differences(step/2)[0]-differences(step)[0])/3
    second = (4*differences(second_step/2)[1]-differences(second_step)[1])/3
    return {
        "beta":beta, "source":source, "measure":kind, "order":order,
        "kernel":[float(exact[0].real),float(exact[0].imag)],
        "derivatives":[[float(x.real),float(x.imag)] for x in exact[1:]],
        "quadrature_difference":float(np.max(np.abs(exact-coarse))),
        "first_derivative_error":float(abs(first-exact[1])),
        "second_derivative_error":float(abs(second-exact[2])),
        "moment_bound_ratios":[float(abs(x)/max(1.,(beta*geometry.limit(1.))**n))
                               for n,x in enumerate(exact)],
    }


def run():
    branches = [stationary_response(j) for j in (1e-2,1e-3,1e-4,1e-5,1e-6)]
    kernels = []
    for beta in (0.,1.,5.,20.):
        for source in (-.01,0.,.01):
            for kind in ("length","squared"):
                kernels.append(kernel_check(beta,source,kind,512))
    dependencies = ("regge_coalescing_kernel.py", "regge_postconstraint_projection.py",
                    "regge_constraint_reduction.py", "regge_quantum_constraint_transfer.py",
                    "regge_internal_marginal.py", "regge_pachner_constraints.py",
                    "regge_pachner_creation.py", "regge_pachner_transport.py", "regge_tent_transfer.py")
    return {
        "status":"[산출]", "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in dependencies},
        "scope":"공급 원천 s_j=beta(R-j*e)의 정적 정상점·전체 유한 진폭 응답",
        "formulas":{
            "stationary":"R_e=j, c=0, F_j=F-beta*j=0 with p_old=beta*dR_old",
            "two_branches":"2 +/- sqrt(j/(24*sqrt(3))) - 5*sqrt(3)*j/144 + O(j**1.5)",
            "susceptibility":"1/R_ee on a regular stationary branch",
            "a_scaling":"a_+/-/(beta*sqrt(j)) -> -/+ 3**0.25",
            "a_squared_scaling":"a**2/(beta**2*j) -> sqrt(3)",
            "negative_source":"R_e>0 on (0,L) except e=2, so j<0 has no real stationary point",
            "entire_kernel":"K_beta(j) entire in complex j; |K|<=1 for real j",
            "derivative_bound":"|d_j**n K| <= (beta*L)**n for real j",
        },
        "stationary_branches":branches, "kernel_cases":kernels,
        "unfinished":[
            "원천 j의 미시 작용과 에너지 공급·반작용",
            "실제 물리 시간·인과적 지연 응답·물리 내적",
            "공통 계량·3+1 중력·모든 힘·암흑부문·허블 텐션",
        ],
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"branches":result["stationary_branches"],
                      "kernel_maxima":{key:max(row[key] for row in result["kernel_cases"]) for key in
                                      ("quadrature_difference","first_derivative_error","second_derivative_error")}},
                     ensure_ascii=False))

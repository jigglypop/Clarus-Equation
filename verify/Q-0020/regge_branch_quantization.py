"""실제 두 정준 가지의 전체 커널 보존과 접합 선택의 비유일성을 검산한다.

고정 경계에서의 좌표변환과 i∂P 후보의 양자 도메인을 구별한다.
공유 기하·내적·작용은 공급 조건이며 실제 축약의 편극은 미유도다.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import regge_postconstraint_projection as projection

full = projection.full


def branch_data(h=1., beta=5.):
    """내부 최소점이 있는 두 가지의 길이와 운동량 구간."""
    full.limit(h)
    if not np.isfinite(beta) or beta <= 0 or h*h <= 7/9:
        raise ValueError("두 가지 사상은 beta>0, h^2>7/9에서 정의한다")
    critical = math.sqrt(2*(h*h+1))
    upper = full.limit(h)
    pc = beta*float(projection.symmetric_g(critical, h))
    ends = beta*np.array([projection.symmetric_g(0., h), projection.symmetric_g(upper, h)])
    return {"h": h, "beta": beta, "critical": critical, "upper": upper,
            "pc": pc, "ends": ends, "widths": ends-pc}


def inverse_momentum(momentum, branch, data):
    if branch not in (0, 1):
        raise ValueError("가지 번호는 0 또는 1이다")
    if not data["pc"] < momentum < data["ends"][branch]:
        raise ValueError("운동량은 해당 가지의 열린 구간 안에 있어야 한다")
    bracket = (0., data["critical"]) if branch == 0 else (data["critical"], data["upper"])
    return brentq(lambda e: data["beta"]*float(projection.symmetric_g(e, data["h"]))-momentum,
                  *bracket, xtol=1e-13)


def branch_rule(data, branch, order=192):
    """P-Pc=폭*u^2로 적분 가능한 임계점 특이성을 제거한다."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit, weight = (nodes+1)/2, weights/2
    width = data["widths"][branch]
    momentum = data["pc"]+width*unit**2
    edge = np.array([inverse_momentum(p, branch, data) for p in momentum])
    jacobian = np.abs(data["beta"]*projection.symmetric_g_e(edge, data["h"]))
    return edge, weight*2*width*unit/jacobian


def coarea_check(h=1., kind="length", phase_scale=5., order=192, beta=5.):
    data = branch_data(h, beta)
    values, masses, norms, omitted = [], [], [], []
    for branch in (0, 1):
        edge, de = branch_rule(data, branch, order)
        density = full.density(edge, h, kind)
        phase = np.exp(1j*phase_scale*full.action(edge, h))
        jacobian = np.abs(beta*projection.symmetric_g_e(edge, h))
        values.append(complex(np.dot(de, density*phase)))
        masses.append(float(np.dot(de, density)))
        norms.append(float(np.dot(de, density*(1+.3*edge/data["upper"])**2)))
        omitted.append(complex(np.dot(de, density*phase*jacobian)))
    edge, weight = projection.transfer.sample_rule((0., data["upper"]), 512)
    density = full.density(edge, h, kind)
    direct = complex(np.dot(weight, density*np.exp(1j*phase_scale*full.action(edge, h))))
    direct_norm = float(np.dot(weight, density*(1+.3*edge/data["upper"])**2))
    return {"h": h, "kind": kind, "phase_scale": phase_scale, "map_beta": beta,
            "branch_values": [[z.real,z.imag] for z in values], "branch_masses": masses,
            "sum": [sum(values).real,sum(values).imag], "direct": [direct.real,direct.imag],
            "kernel_error": abs(sum(values)-direct), "normalization_error": abs(sum(masses)-1),
            "norm_error": abs(sum(norms)-direct_norm),
            "drop_right_error": abs(values[0]-direct),
            "omit_jacobian_error": abs(sum(omitted)-direct)}


def boundary_check(beta=5.):
    """실제 구간에서 항등·교환 접합의 경계형식과 스펙트럼을 대조한다."""
    data = branch_data(beta=beta)
    lengths = data["widths"]
    identity = np.eye(2, dtype=complex)
    swap = np.array([[0.,1.],[1.,0.]], dtype=complex)
    lower_f = np.array([1.+.2j, -.4+.7j])
    lower_g = np.array([.3-.8j, 1.1+.1j])
    rows = []
    for name, unitary in (("identity",identity), ("swap",swap)):
        upper_f, upper_g = unitary@lower_f, unitary@lower_g
        boundary = 1j*(np.vdot(upper_f,upper_g)-np.vdot(lower_f,lower_g))
        if name == "identity":
            modes = [(2*math.pi/lengths[j], np.eye(2,dtype=complex)[:,j]) for j in (0,1)]
        else:
            frequency = 2*math.pi/sum(lengths)
            modes = [(frequency,np.array([1.,np.exp(-1j*frequency*lengths[0])]))]
        residuals = [np.linalg.norm(np.exp(-1j*y*lengths)*a-unitary@a) for y,a in modes]
        rows.append({"junction":name, "boundary_form_error":float(abs(boundary)),
                     "zero_multiplicity":int(2-np.linalg.matrix_rank(identity-unitary)),
                     "positive_frequencies":[float(y) for y,_ in modes],
                     "spectral_boundary_error":float(max(residuals))})
    # 두 아래끝을 같은 값으로 잇고 위끝을 각각 0으로 두면 -2i가 남는다.
    local_lower = np.ones(2, dtype=complex)
    local_defect = -1j*np.vdot(local_lower,local_lower)
    return {"pc":data["pc"], "upper_momenta":data["ends"].tolist(), "widths":lengths.tolist(),
            "extensions":rows, "local_only_defect":[local_defect.real,local_defect.imag],
            "deficiency_indices":[2,2],
            "scope":"고정 두 구간의 i∂P 후보; 실제 28차원 편극·물리 시간 진화 아님"}


def critical_density_check(beta=5.):
    data = branch_data(beta=beta)
    a = 3*math.sqrt(2)*beta
    coefficient = 1/(data["upper"]*math.sqrt(a))
    rows = []
    for delta in (1e-3, 2.5e-4, 6.25e-5):
        edge = [inverse_momentum(data["pc"]+delta,j,data) for j in (0,1)]
        jac = np.abs(beta*projection.symmetric_g_e(np.array(edge),1.))
        density = float(np.sum(1/(data["upper"]*jac)))
        rows.append({"delta":delta,"density":density,
                     "scaled_density":density*math.sqrt(delta),
                     "ratio":density*math.sqrt(delta)/coefficient})
    return {"coefficient":coefficient,"rows":rows}


def g_second(edge, h=1.):
    upper = full.limit(h)
    r = math.sqrt(upper*upper-edge*edge)
    denominator = 9*r*r+8
    constant = 6*h*h-2
    term = constant/denominator-1/3
    return 4*math.sqrt(2)*(edge*term/r**3+18*constant*edge/(r*denominator**2))


def inherited_constraint_check(beta=5., step=2e-5):
    """부모의 위상·밀도를 제거한 F와 정확한 도메인상을 함께 전달한다."""
    data = branch_data(beta=beta)
    rows = []
    for branch in (0,1):
        for fraction in (.2,.6,.85):
            momentum = data["pc"]+fraction*data["widths"][branch]
            edge = inverse_momentum(momentum,branch,data)
            velocity = beta*float(projection.symmetric_g_e(edge,1.))
            velocity_derivative = beta*g_second(edge)/velocity
            for mode in (0,1):
                eigenvalue = 2*math.pi*mode/data["upper"]
                def state(p):
                    e = inverse_momentum(p,branch,data)
                    jac = abs(beta*float(projection.symmetric_g_e(e,1.)))
                    return np.exp(1j*eigenvalue*e)/math.sqrt(data["upper"]*jac)
                delta = min(step,.01*data["widths"][branch])
                coarse = (state(momentum+delta)-state(momentum-delta))/(2*delta)
                fine = (state(momentum+delta/2)-state(momentum-delta/2))/delta
                derivative = (4*fine-coarse)/3
                value = state(momentum)
                operated = -1j*(velocity*derivative+.5*velocity_derivative*value)
                omitted = -1j*velocity*derivative
                rows.append({"branch":branch,"fraction":fraction,"mode":mode,
                             "error":float(abs(operated-eigenvalue*value)),
                             "omit_connection_error":float(abs(omitted-eigenvalue*value))})
    frequency = 2*math.pi/data["upper"]
    return {"rows":rows,"first_positive_eigenvalue":frequency,
            "outer_trace_error":float(abs(np.exp(1j*frequency*data["upper"])-1)),
            "critical_jump_negative_control":2.,
            "scope":"가중 자취 접합을 포함한 부모 주기 H1(e) 도메인의 정확한 단위상"}


def domain_check(beta=5.):
    """원래 균일 상태의 변환상에 대해 임계점 양쪽의 미분 발산을 계산한다."""
    data = branch_data(beta=beta)
    critical, upper = data["critical"],data["upper"]
    a = 3*math.sqrt(2)*beta
    coefficient = 1/(24*upper*math.sqrt(a))

    def integrand(edge):
        # |d_P(1/sqrt(L*kappa))|^2 dP = kappa_e^2/(4L*kappa^4) de.
        gee = g_second(edge)
        kappa = abs(beta*float(projection.symmetric_g_e(edge,1.)))
        return (beta*gee)**2/(4*upper*kappa**4)

    outer = [inverse_momentum(data["pc"]+.01,j,data) for j in (0,1)]
    rows = []
    for epsilon in (1e-4, 2.5e-5, 6.25e-6):
        inner = [inverse_momentum(data["pc"]+epsilon,j,data) for j in (0,1)]
        values = [quad(integrand,outer[0],inner[0],epsabs=1e-7,epsrel=1e-9)[0],
                  quad(integrand,inner[1],outer[1],epsabs=1e-7,epsrel=1e-9)[0]]
        scaled = sum(values)*epsilon**1.5
        rows.append({"epsilon":epsilon,"derivative_integral":sum(values),
                     "scaled":scaled,"ratio":scaled/coefficient})
    return {"coefficient":coefficient,"rows":rows,
            "scope":"L2 상태는 보존되지만 i∂P의 모든 H1 도메인 밖이다"}


def run():
    coarse,medium,refined = [],[],[]
    for h in (.9,1.,1.1):
        for kind in ("length","squared"):
            for phase_scale in (0.,5.,20.):
                coarse.append(coarea_check(h,kind,phase_scale,96))
                medium.append(coarea_check(h,kind,phase_scale,192))
                refined.append(coarea_check(h,kind,phase_scale,384))
    previous = json.loads(Path(projection.__file__).with_suffix(".json").read_text(encoding="utf-8"))
    deps = {Path(projection.__file__).name:hashlib.sha256(Path(projection.__file__).read_bytes()).hexdigest(),
            **previous["dependencies"]}
    for name,sha in deps.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha:
            raise ValueError("선행 소스 해시 불일치: "+name)
    return {"status":"[산출]","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies":deps,"coarea":refined,
            "initial_quadrature_difference":max(abs(complex(*x["sum"])-complex(*y["sum"]))
                                                for x,y in zip(coarse,medium)),
            "quadrature_difference":max(abs(complex(*x["sum"])-complex(*y["sum"]))
                                        for x,y in zip(medium,refined)),
            "boundary":boundary_check(),"critical_density":critical_density_check(),"domain":domain_check(),
            "inherited_constraint":inherited_constraint_check(),
            "scope":"실제 두 정준 가지의 보존 사상과 고정 경계 연산자 후보의 판정",
            "unfinished":["실제 전체 축약의 편극·움직이는 경계·물리 작용과 내적",
                          "공통 계량·3+1 중력·암흑부문·허블 텐션"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"kernel_error":max(x["kernel_error"] for x in report["coarea"]),
                      "quadrature_difference":report["quadrature_difference"],
                      "norm_error":max(x["norm_error"] for x in report["coarea"]),
                      "inherited_constraint_error":max(x["error"] for x in report["inherited_constraint"]["rows"]),
                      "boundary":report["boundary"],"critical_density":report["critical_density"],
                      "domain":report["domain"]},ensure_ascii=False))

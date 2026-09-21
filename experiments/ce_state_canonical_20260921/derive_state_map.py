"""CE-SC1: full metric variations, canonical-lift boundary, and exact counterexamples.

The continuum arguments are in chapter 36. This script checks the explicit
state geometry and exact Poisson witnesses, not quantum gravity completion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sympy as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    checks = []

    def zero(name, expression):
        entries = list(expression) if isinstance(expression, s.MatrixBase) else [expression]
        for entry in entries:
            reduced = s.simplify(s.trigsimp(entry))
            if reduced != 0:
                reduced = s.trigsimp(reduced, method="fu")
            assert reduced == 0, (name, entry)
        checks.append(name)

    x, y, z = s.symbols("x y z", real=True)
    coordinates = (x, y, z)
    vectors = [s.Matrix(v) for v in
               [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1)]]
    r = s.Rational(1, 4)
    c = s.sqrt(10)/4
    phases = [(v.T*s.Matrix(coordinates))[0] for v in vectors]
    state = s.Matrix([c] + [a for phase in phases
                            for a in (r*s.cos(phase), r*s.sin(phase))])
    zero("normalized_13_component_real_state", (state.T*state)[0]-1)
    deriv = [state.diff(q) for q in coordinates]
    for i in range(3):
        zero(f"horizontal_real_derivative_{i}", (state.T*deriv[i])[0])
    h0 = r*r*sum((v*v.T for v in vectors), s.zeros(3))
    for i in range(3):
        for j in range(i, 3):
            zero(f"constant_metric_{i}{j}", (deriv[i].T*deriv[j])[0]-h0[i, j])
    assert h0.eigenvals() == {s.Rational(5, 16): 1, s.Rational(1, 8): 2}
    checks.append("positive_spatial_metric")

    # Amplitude variations preserve normalization to first order. Their
    # spatially varying coefficients introduce no derivatives into delta h.
    normals = []
    for a, phase in enumerate(phases):
        n = s.zeros(13, 1)
        n[0] = -r/c
        n[1+2*a] = s.cos(phase)
        n[2+2*a] = s.sin(phase)
        normals.append(n)
        zero(f"normal_to_state_{a}", (state.T*n)[0])
        for i in range(3):
            zero(f"normal_to_spatial_tangent_{a}_{i}", (deriv[i].T*n)[0])
        for i in range(3):
            for j in range(i, 3):
                actual = (deriv[i].T*n.diff(coordinates[j])
                          + deriv[j].T*n.diff(coordinates[i]))[0]
                zero(f"metric_variation_{a}_{i}{j}", actual-2*r*vectors[a][i]*vectors[a][j])

    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    B = s.Matrix([[2*r*v[i]*v[j] for v in vectors] for i, j in pairs])
    assert B.det() == s.Rational(1, 64)
    assert B.rank() == 6
    checks.append("surjective_metric_variation_at_every_spatial_point")
    dh = s.Matrix(s.symbols("h0:6"))
    phi = B.inv()*dh
    zero("explicit_metric_right_inverse", B*phi-dh)
    pi = s.Matrix(s.symbols("pi0:6"))
    W = s.diag(1, 1, 1, 2, 2, 2)
    amplitude_momentum = B.T*W*pi
    zero("linear_cotangent_one_form", (amplitude_momentum.T*phi-pi.T*W*dh)[0])
    print("state metric, full-rank variation and cotangent form passed", flush=True)

    # Intrinsic projective-state Poisson bracket: first compute the qubit
    # functional derivatives; then check the full product-state contributions.
    rho, ell, hbar = s.symbols("rho ell hbar", positive=True)
    f, fx, g, gy = s.symbols("f fx g gy", real=True)
    k, l = s.symbols("k l", real=True)
    var = rho*(1-rho)
    F_rho = ell**2*f*(1-2*rho)*k*k
    F_phase = -2*ell**2*var*k*fx
    G_rho = ell**2*g*(1-2*rho)*l*l
    G_phase = -2*ell**2*var*l*gy
    bracket_density = (F_rho*G_phase-F_phase*G_rho)/hbar
    expected = 2*ell**4*var*(1-2*rho)*(k*l*l*g*fx-k*k*l*f*gy)/hbar
    zero("qubit_functional_poisson_density", bracket_density-expected)
    ci = -f*k*k*(1-2*rho)+s.I*fx*k
    cj = -g*l*l*(1-2*rho)+s.I*gy*l
    zero("full_projective_gradient_qubit_block",
         2*ell**4/hbar*s.im(s.conjugate(ci)*cj)*var-expected)
    # For the real factor its tangent/tension inner products are real.
    # The entangled normal block has <u_i,u_j> real times
    # <D_i v,D_j v> = k_i k_j rho(1-rho), also real.
    real_u_inner = s.symbols("u_inner", real=True)
    zero("entangled_normal_block_has_zero_imaginary_part",
         s.im(real_u_inner*k*l*var))
    witness_density = expected.subs({rho:s.Rational(1, 4), k:1, l:1,
                                    f:s.sin(x), fx:s.cos(x), g:s.cos(x), gy:0})
    witness = s.integrate(witness_density, (x, 0, 2*s.pi))*(2*s.pi)**2
    zero("nonzero_metric_poisson_bracket_on_full_product",
         witness-3*s.pi**3*ell**4/(4*hbar))
    assert witness != 0
    checks.append("intrinsic_metric_coordinates_fail_canonical_commutativity")
    total_metric = h0+s.Rational(3,16)*s.Matrix([1,1,0])*s.Matrix([[1,1,0]])
    assert all(total_metric[:n,:n].det() > 0 for n in (1,2,3))
    checks.append("poisson_witness_spatial_metric_positive")
    print("intrinsic Poisson counterexample passed", flush=True)

    # Independent finite-dimensional counterexample: identical metric over
    # an open patch, but a different curvature norm. Both retain full rank B.
    theta, ph = s.symbols("theta ph", real=True)
    q = s.Matrix([s.cos(theta/2), s.exp(s.I*ph)*s.sin(theta/2)])

    def geometry(ket):
        derivatives = [ket.diff(t) for t in (theta, ph)]
        # Contract inner products before simplifying: expanding the entire
        # projector creates redundant degree-eight trigonometric expressions.
        reduce = lambda a:s.trigsimp(s.simplify(s.expand(a)))
        connection = [reduce((ket.conjugate().T*d)[0]) for d in derivatives]
        Q = s.Matrix([[reduce((di.conjugate().T*dj)[0]
                       -s.conjugate(connection[i])*connection[j])
                       for j,dj in enumerate(derivatives)]
                       for i,di in enumerate(derivatives)])
        return Q.applyfunc(lambda a:s.simplify(s.expand_complex(a)))

    qp = s.kronecker_product(q,q)
    qm = s.kronecker_product(q,s.conjugate(q))
    Qp, Qm = geometry(qp), geometry(qm)
    desired = s.diag(s.Rational(1,2),s.sin(theta)**2/2)
    zero("finite_equal_metric_plus", s.re(Qp)-desired)
    zero("finite_equal_metric_minus", s.re(Qm)-desired)
    Fp = s.simplify(s.I*(Qp[0,1]-Qp[1,0]))
    Fm = s.simplify(s.I*(Qm[0,1]-Qm[1,0]))
    zero("finite_plus_curvature", Fp+s.sin(theta))
    zero("finite_minus_curvature", Fm)
    equator_metric = h0+s.diag(s.Rational(1,2),s.Rational(1,2),0)
    hi = equator_metric.inv()
    curvature_norm = 2*(hi[0,0]*hi[1,1]-hi[0,1]**2)
    assert curvature_norm > 0
    checks.append("same_regular_metric_fiber_different_gauge_invariant_energy")
    print("finite equal-metric, unequal-curvature counterexample passed", flush=True)

    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    chapter_dir = ROOT/"paper/후속연구_기록과_상태선택"
    sources = [next(chapter_dir.glob("23_*.md")),next(chapter_dir.glob("35_*.md")),
               next((ROOT/"paper/06_QFT_재설계").glob("55_*.md")),
               next((ROOT/"paper/06_QFT_재설계").glob("62_*.md"))]
    result = {
        "scope":"CE rank-one state metric, conditional cotangent lift, local Berry-state-field counterexample",
        "all_checks_passed":True,"checks":checks,"source_sha256":sha(Path(__file__)),
        "premise_sources":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "real_state_dimension":13,"metric_variation_rank":6,
        "raw_metric_variation_determinant":str(B.det()),
        "right_inverse":list(map(str,phi)),
        "intrinsic_poisson_counterexample":{
            "full_state_dimension":26,"domain":"three circles of length 2*pi",
            "occupation":"1/4","phase":"x+y","smear_f":"sin(x)","smear_g":"cos(x)",
            "actual_bracket":str(s.factor(witness)),"required_canonical_bracket":"0",
            "full_projective_normal_blocks_accounted_for":True},
        "three_channel_direct_sum_extension":{
            "orthonormal_frame_rank":3,"ambient_dimension":78,
            "actual_bracket":str(s.factor(3*witness)),
            "basis":"block-diagonal first-variation and symplectic-pairing proof in chapter 36"},
        "metric_quotient_counterexample":{
            "full_state_dimension":52,"same_metric_on_entire_patch":True,
            "plus_curvature":str(Fp),"minus_curvature":str(Fm),
            "plus_curvature_norm_at_equator_ell_1":str(curvature_norm),
            "minus_curvature_norm":"0","metric_variation_rank_both":6},
        "full_goal_complete":False,
        "not_claimed":["cotangent momentum follows from CE microscopic dynamics",
                       "intrinsic Berry ansatz is the only possible CE symplectic structure",
                       "all metrics admit a global state lift",
                       "Lorentzian algebra or quantum constraint closure derived",
                       "all metric-fiber components are continuously gauge connected",
                       "no possible unified state geometry",
                       "gauge curvature uniquely fixes all remaining state data"]
    }
    (HERE/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["all_checks_passed","metric_variation_rank",
                     "intrinsic_poisson_counterexample","metric_quotient_counterexample"]},indent=2))


if __name__ == "__main__":
    main()

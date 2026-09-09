"""Gauge mass-renormalization audit of the common-Casimir candidate.

Only the one-loop gauge contribution to the scalar mass beta function is
tested. The dimensionless flow time is a diagnostic, not a fitted physical
coupling or a new scale choice for observations. Quartics, Yukawas, thresholds
and finite pole matching are not computed here.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp


def algebra_checks(fields):
    values=[sy.Rational(row['casimir']) for row in fields]
    dimensions=[row['dimension'] for row in fields]
    x=sy.symbols('x')
    polynomial=sy.factor(sy.prod(x-c for c in values))
    vandermonde=sy.Matrix([[c**n for n in range(6)] for c in values])
    assert vandermonde.rank()==6
    assert sy.Matrix([[1,c,c*c] for c in values]).rank()==3
    C=sy.diag(*values);I=sy.eye(6)
    projectors=[]
    for c in values:
        P=I
        for other in values:
            if other!=c: P=P*(C-other*I)/(c-other)
        assert P*P==P and sy.trace(P)==1
        projectors.append(P)
    assert sum(projectors,sy.zeros(6))==I
    for i,P in enumerate(projectors):
        for j,Q in enumerate(projectors):
            assert P*Q==(P if i==j else sy.zeros(6))
    # A common beta_epsilon cannot equal -c_R*epsilon for both c=0 and c>0.
    assert min(values)==0 and max(values)>0
    # Check standard hypercharge convention against the Higgs doublet gauge term.
    gy,g2=sy.symbols('g_Y g_2')
    higgs=sy.expand(-6*(gy**2*sy.Rational(1,4)+g2**2*sy.Rational(3,4)))
    assert higgs==-sy.Rational(3,2)*gy**2-sy.Rational(9,2)*g2**2
    # Constant-coupling, aligned-flow heavy observable: running alone has a
    # spurious O(tau) change; its mass-running leading log cancels that change.
    tau,c,F0=sy.symbols('tau c F0')
    naive=F0*sy.exp(tau*c)
    first_order=naive*(1-tau*c)
    assert sy.diff(first_order,tau).subs(tau,0)==0
    assert sy.diff(naive*sy.exp(-tau*c),tau)==0
    # Along a single complex scalar background phi=rho/sqrt(2), the gauge
    # logarithm independently fixes the g^4 source for V=lambda*|phi|^4.
    rho,g,mu=sy.symbols('rho g mu',positive=True)
    vg=3*g**4*rho**4/(64*sy.pi**2)*sy.log(g**2*rho**2/mu**2)
    beta_lambda=6*g**4/(16*sy.pi**2)
    assert sy.simplify(mu*sy.diff(vg,mu)+beta_lambda*rho**4/4)==0
    return dict(casimir_minimal_polynomial=str(polynomial),
                central_polynomial_dimension=6,mass_ansatz_tangent_rank=3,
                higgs_gauge_beta_coefficient=str(higgs),
                unit_charge_quartic_gauge_source=str(16*sy.pi**2*beta_lambda),
                sum_dimension_times_C=str(sum(d*c for d,c in zip(dimensions,values))),
                first_order_matched_observable_series=str(sy.series(first_order,tau,0,3)),
                scope='Exact finite-dimensional algebra; aligned gauge-flow example, not a complete beta system.')


def flow_and_score_checks(previous):
    fields=previous['casimir']['fields']
    charged=[row for row in fields if row['name']!='nc']
    c=np.array([float(sy.Rational(row['casimir'])) for row in charged])
    dimension=np.array([row['dimension'] for row in charged])
    reference=previous['prediction']
    masses=np.array([p['mass_squared'] for p in reference['parts']])
    amplitudes=np.array([p['muon_EM'] for p in reference['parts']])
    epsilon=.35*(.014414e-9)**2
    gap=385e-12;sigma=np.hypot(145,620)*1e-12
    mu0=float(amplitudes.sum())
    score0=reference['null_frozen_background_R14_after']
    r0=(mu0-gap)/sigma
    # Recover the fixed BAO chi-square from the already verified score.
    bao_chi2=14*score0**2-r0*r0
    def score(mu): return float(np.sqrt((bao_chi2+((mu-gap)/sigma)**2)/14))
    rows=[]
    for tau in [-.04,-.02,0.,.02,.04]:
        factors=np.exp(-tau*c)
        if tau:
            # Integrate dimensionless ratios, not tiny absolute epsilons.
            sol=solve_ivp(lambda t,y:-np.r_[c,c]*y,(0.,tau),np.ones(10),
                          method='DOP853',rtol=2e-12,atol=2e-14,max_step=.002)
            assert sol.success
            ode_error=float(np.max(abs(sol.y[:,-1]-np.r_[factors,factors])))
            assert ode_error<2e-12
        else: ode_error=0.
        ratio_error=float(np.max(abs((epsilon*factors)/(masses*factors)-epsilon/masses)))
        assert ratio_error<1e-40
        # Values at tau=0 are the prior finite-mass integral; transport only
        # its leading 1/s dependence, isolating the mass-running contribution.
        naive_parts=amplitudes/factors
        matched_parts=naive_parts*factors
        first_order_parts=naive_parts*(1-tau*c)
        naive=float(naive_parts.sum());matched=float(matched_parts.sum())
        first_order=float(first_order_parts.sum())
        assert abs(matched-mu0)<1e-28
        naive_score=score(naive);matched_score=score(matched)
        stable_change=(((naive-gap)/sigma)**2-r0*r0)/(14*(naive_score+score0))
        rows.append(dict(tau=tau,epsilon_R_over_epsilon_D=dict(zip([r['name'] for r in charged],factors.tolist())),
                         independent_flow_ode_error=ode_error,
                         within_representation_epsilon_over_s_error=ratio_error,
                         running_mean_squared_mass_GeV2=float(np.dot(dimension,masses*factors)/15),
                         naive_mass_only_EM=naive,mass_running_LL_matched_EM=matched,
                         first_order_log_matched_EM=first_order,
                         naive_R14=naive_score,naive_delta_R14=stable_change,
                         mass_running_LL_matched_R14=matched_score,
                         matched_delta_R14=matched_score-score0))
    assert rows[0]['naive_delta_R14']>0 and rows[-1]['naive_delta_R14']<0
    assert all(abs(row['matched_delta_R14'])<2e-15 for row in rows)
    # O(tau^2) remainder after the O(tau) matching term is included.
    e_small=abs(rows[3]['first_order_log_matched_EM']/mu0-1)
    e_large=abs(rows[4]['first_order_log_matched_EM']/mu0-1)
    assert 3.9<e_large/e_small<4.2
    return dict(reference_null_R14=score0,reference_EM=mu0,rows=rows,
                first_order_matching_quadratic_remainder_ratio=e_large/e_small,
                actual_additional_rmse_improvement=None,
                scope='A renormalization-scale diagnostic at fixed physics. Compensates only the gauge mass-running leading logs; not a new muon prediction or a complete three-loop result.')


if __name__=='__main__':
    source=Path(__file__).with_name('ce_residual_mass_closure.json')
    previous=json.loads(source.read_text())
    result=dict(algebra=algebra_checks(previous['casimir']['fields']),
                scale_diagnostic=flow_and_score_checks(previous),
                convention='16*pi^2*dM_R^2/dln(mu)|gauge=-6*sum_a(g_a^2*C2_a(R))*M_R^2',
                diagnostic_alignment='g_a^2=gbar^2/T_a; tau=6*integral(gbar^2*dln(mu))/(16*pi^2). Alignment is not claimed to persist under physical gauge running.',
                required_new_structures=['C^2 in diagonal mass counterterms',
                                         'C times the cyclic mass operator',
                                         'gauge-generated scalar quartics and their matching conditions'],
                fitted_parameters=[],full_joint_rmse=None,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

"""Common cyclic shift under Schur reduction, and a gauge-Casimir candidate.

The Casimir penalty is an explicit new structural hypothesis. Absolute scales,
the matter representation and the cosmological state remain supplied inputs.
Compare the existing 14-entry BAO+EM diagnostic on frozen backgrounds, with
the same charged mean-square mass and no optimization of representation masses.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import sympy as sy
from scipy.linalg import block_diag

from common_spectrum_muon import exact_em, ALPHA, Q2, M_MU
from ce_obs32_profiled_matter import ce


def schur_checks():
    z, x, a, b, d = sy.symbols('z x a b d', real=True)
    K = sy.Matrix([[z+a+x, b], [b, z+d+x]])
    effective = z+a+x-b*b/(z+d+x)
    assert sy.simplify(K.det()-(z+d+x)*effective) == 0
    M0 = a+x-b*b/(d+x)
    Z = 1+b*b/(d+x)**2
    assert sy.simplify(effective.subs(z,0)-M0) == 0
    assert sy.simplify(sy.diff(effective,z).subs(z,0)-Z) == 0
    assert sy.simplify(sy.diff(effective,x)-sy.diff(effective,z)) == 0
    # A fixed test, no physical scale fitting. Add the cycle on both P and Q.
    T = np.array([[2., .4], [.4, 5.]])
    S = np.roll(np.eye(3, dtype=complex),1,axis=0)
    rows=[]
    for theta in [0., .7, np.pi]:
        C = .2*(np.exp(1j*theta/3)*S+np.exp(-1j*theta/3)*S.conj().T)
        full = np.kron(T,np.eye(3))+np.kron(np.eye(2),C)
        expected = np.sort((np.linalg.eigvalsh(T)[:,None]+np.linalg.eigvalsh(C)).ravel())
        error = float(np.max(abs(np.linalg.eigvalsh(full)-expected)))
        assert error < 1e-13
        rows.append(dict(theta=theta, full_pole_error=error))
    # Opposite choice: add x only to P. Slopes differ and sum to one.
    xx=sy.symbols('xx', real=True)
    low = (7+xx-sy.sqrt((3-xx)**2+sy.Rational(16,25)))/2
    high = (7+xx+sy.sqrt((3-xx)**2+sy.Rational(16,25)))/2
    slopes=[float(sy.diff(y,xx).subs(xx,0)) for y in [low,high]]
    assert abs(sum(slopes)-1)<1e-14 and abs(slopes[0]-slopes[1])>.9
    # K0 is a mass-squared matrix, not a Hamiltonian energy matrix.
    lam=sy.symbols('lam',positive=True)
    rank_one=lam*sy.ones(2)
    assert rank_one.det()==0 and rank_one.eigenvals()=={2*lam:1,sy.Integer(0):1}
    kinetic_checks=[]
    for delta in [1e-2,1e-3,1e-4]:
        exact=float(np.linalg.eigvalsh([[.4+delta,2.],[2.,10.]])[0])
        canonical=delta/1.04
        assert abs(exact-canonical)<abs(exact-delta)/100
        kinetic_checks.append(dict(delta=delta,exact_light_mass_squared=exact,
                                   canonically_normalized_mass_squared=canonical,
                                   unnormalized_mass_squared=delta))
    return dict(effective_kernel=str(effective), zero_momentum_mass=str(M0),
                kinetic_coefficient=str(Z), shared_shift_identity='dKeff/dx=dKeff/dz',
                pole_checks=rows, visible_only_shift_pole_slopes=slopes,
                low_energy_kinetic_checks=kinetic_checks,
                scope='Constant backgrounds and canonical full kinetic term. Mixing is allowed only within equivalent gauge representations.')


def generators_and_casimir():
    # Same one-family representation used in the previous charge derivation.
    sigma=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    color=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        x=np.zeros((3,3),complex);x[i,j]=x[j,i]=.5;color.append(x)
        y=np.zeros((3,3),complex);y[i,j]=-.5j;y[j,i]=.5j;color.append(y)
    color.extend([np.diag([1.,-1.,0.])/2,np.diag([1.,1.,-2.])/(2*np.sqrt(3))])
    fields=[('Q',3,2,sy.Rational(1,6)),('uc',3,1,sy.Rational(-2,3)),
            ('dc',3,1,sy.Rational(1,3)),('L',1,2,sy.Rational(-1,2)),
            ('ec',1,1,sy.Integer(1)),('nc',1,1,sy.Integer(0))]
    Gens=[block_diag(*[float(y)*np.eye(c*w) for _,c,w,y in fields])]
    for s in sigma:
        Gens.append(block_diag(*[np.kron(np.eye(c),s/2) if w==2 else np.zeros((c,c)) for _,c,w,y in fields]))
    for g in color:
        blocks=[]
        for name,c,w,_ in fields:
            blocks.append(np.kron(-g.conj() if name in ['uc','dc'] else g,np.eye(w)) if c==3 else np.zeros((w,w)))
        Gens.append(block_diag(*blocks))
    metric=np.array([[np.trace(x@y).real for y in Gens] for x in Gens])
    inverse=np.linalg.inv(metric)
    C=sum(inverse[a,b]*(x@y) for a,x in enumerate(Gens) for b,y in enumerate(Gens))
    assert np.max(abs(metric-np.diag([10/3]+[2]*11)))<1e-14
    rows=[];offset=0
    for name,c,w,y in fields:
        casimir=y*y/sy.Rational(10,3)+(sy.Rational(3,8) if w==2 else 0)+(sy.Rational(2,3) if c==3 else 0)
        charge2=c*sum((y+t)**2 for t in ([sy.Rational(-1,2),sy.Rational(1,2)] if w==2 else [0]))
        block=C[offset:offset+c*w,offset:offset+c*w]
        assert np.max(abs(block-float(casimir)*np.eye(c*w)))<1e-14
        rows.append(dict(name=name,dimension=c*w,casimir=str(casimir),charge_squared_sum=str(charge2)))
        offset+=c*w
    assert abs(np.trace(C)-12)<1e-14
    assert max(np.max(abs(C@g-g@C)) for g in Gens)<1e-14
    # Casimir is invariant under a change of basis in the gauge algebra.
    rng=np.random.default_rng(37);R=np.eye(12)+.05*rng.normal(size=(12,12))
    new=[sum(R[a,b]*Gens[b] for b in range(12)) for a in range(12)]
    invnew=np.linalg.inv(R@metric@R.T)
    Cnew=sum(invnew[a,b]*(x@y) for a,x in enumerate(new) for b,y in enumerate(new))
    assert np.max(abs(C-Cnew))<1e-13
    W=sum(sy.Rational(row['charge_squared_sum'])/sy.Rational(row['casimir']) for row in rows if row['name']!='nc')
    assert W==sy.Rational(65,7)
    return dict(fields=rows,dimension=16,casimir_trace=float(np.trace(C).real),
                zero_eigenvalue_multiplicity=int(np.sum(abs(np.linalg.eigvalsh(C))<1e-12)),
                algebra_basis_change_error=float(np.max(abs(C-Cnew))),
                inverse_mass_charge_index=str(W))


def prediction_check(casimir):
    oldpath=Path(__file__).with_name('ce_symmetric_bao_ruler.json')
    old=json.loads(oldpath.read_text())
    ref_mass=1000.  # GeV; inherited, not optimized here
    neutral_s=(.014414e-9)**2
    epsilon=.35*neutral_s
    # Preserve mean diagonal mass squared over all 15 gauge-nonsinglet components.
    scale_squared=15*(ref_mass**2-neutral_s)/12
    parts=[]
    for field in casimir['fields']:
        if field['name']=='nc': continue
        c=float(sy.Rational(field['casimir']))
        charge2=float(sy.Rational(field['charge_squared_sum']))
        s=neutral_s+scale_squared*c
        value=exact_em(np.sqrt(s),epsilon/s,0.)*charge2/Q2
        other=exact_em(np.sqrt(s),epsilon/s,0.,representation='spectral')*charge2/Q2
        assert abs(value/other-1)<2e-7
        parts.append(dict(name=field['name'],mass_GeV=float(np.sqrt(s)),mass_squared=s,
                          dimension=field['dimension'],charge_squared_sum=charge2,
                          muon_EM=value, spectral_EM=other))
    assert abs(sum(x['dimension']*x['mass_squared'] for x in parts)/(15*ref_mass**2)-1)<1e-14
    mu=sum(x['muon_EM'] for x in parts)
    reference=exact_em(ref_mass,epsilon/ref_mass**2,0.)
    assert abs(reference-old['muon_EM_component'])<1e-27
    leading_ratio=sy.Rational(4,5)*sy.Rational(65,7)/sy.Rational(16,3)
    assert leading_ratio==sy.Rational(39,28)
    assert abs(mu/reference-float(leading_ratio))<1e-7
    # Direct covariance calculation; previous backgrounds and acoustic ruler fixed.
    _,y,_,cov,*_=ce.load_data();chol=np.linalg.cholesky(cov)
    sigma=np.hypot(145,620)*1e-12;gap=385e-12
    rmu=(mu-gap)/sigma;rmu0=(reference-gap)/sigma
    statepath=Path(__file__).with_name('ce_symmetric_small_f_stability.json')
    states=json.loads(statepath.read_text())
    final_theta={(row['r'],case['seed']):case['final_theta'] for row in states['rows'] for case in row['cases']}
    rows=[]
    for row in old['rows']:
        case_s=(row['sqrt_s_eV']*1e-9)**2
        case_epsilon=row['r']*case_s
        theta=final_theta[(row['r'],row['theta_initial'])]
        case_scale=15*(ref_mass**2-case_s)/12
        case_mu=sum(exact_em(np.sqrt(case_s+case_scale*float(sy.Rational(field['casimir']))),
                                  case_epsilon/(case_s+case_scale*float(sy.Rational(field['casimir']))),
                                  theta)*float(sy.Rational(field['charge_squared_sum']))/Q2
                    for field in casimir['fields'] if field['name']!='nc')
        assert abs(case_mu-mu)<1e-27
        case_rmu=(case_mu-gap)/sigma
        residual=np.linalg.solve(chol,np.array(row['prediction'])-y)
        before=float(np.sqrt((residual@residual+rmu0*rmu0)/14))
        after=float(np.sqrt((residual@residual+case_rmu*case_rmu)/14))
        assert abs(before-row['partial_rmse_14'])<1e-12
        # Avoid cancellation when reporting the very small score change.
        difference=(case_rmu*case_rmu-rmu0*rmu0)/(14*(after+before))
        assert difference<0 and abs((after-before)-difference)<1e-15
        rows.append(dict(r_D=row['r'],theta_initial=row['theta_initial'],
                         theta_final=theta,common_epsilon_GeV2=case_epsilon,muon_EM=case_mu,
                         previous_R14=before,frozen_background_R14=after,delta_R14=difference))
    null_before=old['baseline']['partial_rmse_14']
    null_after=float(np.sqrt((old['baseline']['bao_chi2']+rmu*rmu)/14))
    # Leading charged relative vacuum response at theta=0; no large-mass cancellation.
    import mpmath as mp
    with mp.workdps(70):
        rd=mp.mpf('.35');sD=mp.mpf('.014414e-9')**2;ep=rd*sD
        def raw(t):
            x=[sD+2*ep*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
            return sum(v*v*(mp.log(v)-mp.mpf('1.5')) for v in x)/(32*mp.pi**2)
        neutral_top=raw(0)-raw(mp.pi)
        heavy_top=ep**3/(8*mp.pi**2)*sum(mp.mpf(p['dimension'])/mp.mpf(str(p['mass_squared'])) for p in parts)
        vacuum_ratio=float(heavy_top/neutral_top)
    return dict(parts=parts, Lambda_squared_GeV2=scale_squared,
                inherited_mean_squared_mass_GeV2=ref_mass**2,
                previous_muon_EM=reference,new_muon_EM=mu,
                heavy_limit_enhancement=str(leading_ratio),
                muon_standardized_residual_before=rmu0,muon_standardized_residual_after=rmu,
                rows=rows,null_R14_before=null_before,null_frozen_background_R14_after=null_after,
                charged_to_neutral_relative_vacuum_top_ratio_leading=vacuum_ratio,
                calibration='Fixed the previous 1 TeV RMS diagonal mass over 15 nonsinglet components; no mass was optimized against observations.',
                interpretation='Conditional change of scalar representation masses only. The neutral backgrounds and BAO ruler are held fixed; no new cosmological state or full joint likelihood.',
                source_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                state_source_sha256=hashlib.sha256(statepath.read_bytes()).hexdigest())


if __name__=='__main__':
    casimir=generators_and_casimir()
    result=dict(schur=schur_checks(),casimir=casimir,prediction=prediction_check(casimir),
                assumptions=['one-family scalar representation plus its singlet identified as the neutral sector',
                             'quadratic gauge-relation penalty with trace metric of that representation',
                             'same cyclic operator acts on every retained and eliminated copy',
                             'neutral scale, epsilon, heavy RMS scale and cold states remain inputs'],
                fitted_parameters=[],full_joint_rmse=None)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

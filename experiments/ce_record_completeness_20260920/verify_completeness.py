#!/usr/bin/env python3
"""CE-RC1: record completeness, hidden metric response, independent-selection audit.
Synthetic records only. Reconstructing an input Hamiltonian is NOT a prediction
of its natural coefficients. No network access or repository writes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import sympy as sy
from scipy.linalg import eigh
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import as1_reference as old

class Checks:
    def __init__(self): self.rows = []
    def add(self, name, condition, **data):
        ok = bool(condition)
        self.rows.append({'name': name, 'passed': ok, **data})
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok: raise AssertionError((name, data))

def records(h):
    energies, vectors = eigh(h)
    psi = vectors[:, 0]
    if psi.sum() < 0: psi = -psi
    if np.min(psi) <= 0: raise ValueError('Positive-ground-state hypothesis failed')
    amplitudes = psi[:, None] * vectors
    moment = (amplitudes * (energies-energies[0])) @ amplitudes.T
    p = psi**2
    recovered = moment / np.sqrt(p[:,None]*p[None,:])
    return energies, p, moment, recovered

def graph_parent(p, edge_weights, nx=5, ny=6):
    n=nx*ny; lap=np.zeros((n,n)); k=0
    for i in range(nx):
        for j in range(ny):
            a=i*ny+j
            for di,dj in [(1,0),(0,1)]:
                if i+di<nx and j+dj<ny:
                    b=(i+di)*ny+j+dj; w=edge_weights[k]; k+=1
                    lap[a,a]+=w; lap[b,b]+=w; lap[a,b]-=w; lap[b,a]-=w
    return lap / np.sqrt(p[:,None]*p[None,:])

def dipole(shape, group, w=(3.,.84,1.07)):
    # Multiplication by one Cartesian coordinate maps an angular singlet to ell=1.
    # Radial Laguerre phases match AS1's positive off-diagonal convention.
    axis=group+1; n=shape[axis]; dim=(2,4)[group]
    d=np.diag(np.sqrt((np.arange(n)+dim/2)/(w[axis]*dim)))
    d+=np.diag(np.sqrt(np.arange(1,n)/(w[axis]*dim)),1)
    mats=[np.eye(a) for a in shape]; mats[axis]=d
    return old.kron(*mats)

def run():
    ck=Checks(); out={'name':'CE-RC1','scope':'Real local scalar operators and inherited finite CE model; synthetic records; no natural-constant selection'}
    # I. Fully resolved discrete position/time records, no polynomial dictionary.
    xx,yy=np.meshgrid(np.linspace(-1.3,1.3,5),np.linspace(-1.2,1.2,6),indexing='ij')
    p=np.exp(-.6*xx.ravel()**2-.5*yy.ravel()**2-.15*(xx*yy).ravel()**2); p/=p.sum()
    ne=4*6+5*5; rng=np.random.default_rng(7319)
    conductances=[(.5+rng.random(ne))*.03,(.6+rng.random(ne))*.027]
    parents=[graph_parent(p,w) for w in conductances]; rr=[]
    for k,h in enumerate(parents):
        ev,pv,m,hr=records(h+(.8+2*k)*np.eye(len(h)))
        error=float(np.linalg.norm(hr-h,2))
        ck.add('complete_record_reconstruction_'+str(k),error<1e-10,error=error)
        ck.add('positive_unique_graph_ground_'+str(k),ev[1]-ev[0]>.01 and max(abs(pv-p))<1e-12,gap=float(ev[1]-ev[0]))
        ck.add('record_gram_and_ground_null_'+str(k),np.linalg.eigvalsh(m)[0]>-1e-12 and np.linalg.norm(m@np.ones(len(p)))<1e-11)
        # Independent matrix-exponential prediction from the reconstructed generator.
        a=np.sin(xx.ravel())+.2*yy.ravel()**2; v=a*np.sqrt(p)
        exact=np.vdot(v,expm_multiply(-.37j*h,v)); pred=np.vdot(v,expm_multiply(-.37j*hr,v))
        ck.add('new_observable_time_response_'+str(k),abs(exact-pred)<1e-11,error=float(abs(exact-pred)))
        rr.append({'ground_probability_max_error':float(max(abs(pv-p))),'gap':float(ev[1]-ev[0]),'reconstruction_operator_error':error,'time_response':[float(pred.real),float(pred.imag)]})
    ck.add('same_density_different_dynamics',abs(rr[0]['gap']-rr[1]['gap'])>.001)
    # Arbitrary mixing of two edge sets stays a valid positive parent: no unique self-selection.
    loop=[]
    for a in [0.,.2,.5,.8,1.]:
        h=(1-a)*parents[0]+a*parents[1]; ev,pv,m,hr=records(h)
        loop.append({'mix':a,'self_consistency_error':float(np.linalg.norm(hr-h,2)),'gap':float(ev[1]-ev[0])})
    ck.add('self_consistency_is_identity_not_selection',max(x['self_consistency_error'] for x in loop)<1e-10 and np.ptp([x['gap'] for x in loop])>.001)
    ev,pv,mm,hs=records(parents[0]+13.7*np.eye(len(p)))
    ck.add('energy_origin_remains_invisible',np.linalg.norm(hs-parents[0],2)<1e-10)
    # Deleting cross records loses the off-diagonal generator.
    bad=np.diag(np.diag(mm))/p[:,None]
    ck.add('equal_time_or_diagonal_records_not_complete',np.linalg.norm(bad-parents[0],2)>.1)
    out['finite_complete_records']=rr; out['self_consistency_family']=loop
    # II. Local metric / potential identities in an independent exact symbolic calculation.
    x,y=sy.symbols('x y',real=True); coord=sy.Matrix([x,y]); rho2=x*x+y*y
    eta=sy.Rational(3,10); G=sy.eye(2)+2*eta*(rho2*sy.eye(2)-coord*coord.T)
    grad=lambda f:sy.Matrix([sy.diff(f,z) for z in coord])
    div=lambda v:sum(sy.diff(v[i],coord[i]) for i in range(2))
    ell=-(x*x+sy.Rational(13,10)*y*y+sy.Rational(1,5)*x*x*y*y)/2
    V=sy.expand((div(G*grad(ell))+(grad(ell).T*G*grad(ell))[0])/2)
    H=lambda f:sy.expand(-div(G*grad(f))/2+V*f)
    f=x*x+y; g=x*y+y*y; z=1+x-y+x*y
    comm=sy.expand(f*H(g*z)-f*g*H(z)-H(g*f*z)+g*H(f*z))
    ck.add('exact_variable_metric_double_commutator',sy.expand(comm-(grad(f).T*G*grad(g))[0]*z)==0)
    ck.add('positive_radial_and_tangent_metric',sy.simplify(G.det()-(1+2*eta*rho2))==0 and G.trace()==2+2*eta*rho2)
    w=1+x+2*y*y
    def E(a,b): return (grad(a).T*G*grad(b))[0]/2
    loc=[]
    for i in range(2):
        for j in range(2):
            val=sy.expand(E(coord[i],w*coord[j])+E(coord[j],w*coord[i])-E(coord[i]*coord[j],w)-w*G[i,j])
            loc.append(val)
    ck.add('localized_record_recovers_every_metric_component',all(v==0 for v in loc))
    psi=sy.exp(ell)
    ck.add('positive_density_inverse_schrodinger_identity',sy.simplify(-div(G*grad(psi))/2+V*psi)==0)
    # General ground-state transform: no omitted residual/geometry energy.
    lhs=sy.simplify((-div(G*grad(psi*f))/2+V*psi*f)/psi)
    rhs=-div(G*grad(f))/2-(grad(ell).T*G*grad(f))[0]
    ck.add('ground_transform_quadratic_form_identity',sy.simplify(lhs-rhs)==0)
    B=rho2*sy.eye(2)-coord*coord.T
    radial=x*x+y*y+(x*x+y*y)**2
    ck.add('angular_metric_invisible_to_radial_functions',sy.simplify(B*grad(radial))==sy.zeros(2,1))
    # The coefficient class really contains position-dependent kinetic energy.
    L=lambda h:x*sy.diff(h,y)-y*sy.diff(h,x)
    ck.add('angular_casimir_equals_positive_metric_extension',sy.expand(-div(B*grad(f))+L(L(f)))==0)
    out['continuum_symbolic']={'metric':str(G),'log_wavefunction':str(ell),'potential':str(V),'identities':'exact SymPy rational arithmetic'}
    # III. Inherited CE interacting ground; nonradial probes reveal hidden eta's.
    angular=[]
    for shape in [(12,6,6),(24,12,12),(28,14,14)]:
        ops=old.operators(shape); ee,vv,h=old.ground(ops); v=vv[:,0]; e0=float(ee[0]); row={'shape':list(shape),'groups':[]}
        for group,eta0 in enumerate([.37,.19]):
            dim=(2,4)[group]; o=ops[4+group]; av=float(v@(o@v)); v0=o@v-av*v
            kinetic=float(v0@(h@v0-e0*v0))/av
            el=[0,0]; el[group]=1
            op1=old.operators(shape,ell=tuple(el)); e1,p1,h1=old.ground(op1)
            D=dipole(shape,group); d=D@v; m0=float(d@d)
            h1shift=h1+eta0*(dim-1)*sp.eye(h1.shape[0],format='csr')
            m1=float(d@(h1shift@d-e0*d))
            etahat=(m1-kinetic/2)/((dim-1)*m0)
            raw=float(d@(h1@d-e0*d)); gap=float(e1[0]-e0)
            row['groups'].append({'dimension':dim,'input_eta':eta0,'kinetic_from_radial_record':kinetic,'dipole_m0':m0,'dipole_m1':m1,'recovered_eta':etahat,'eta_error':abs(etahat-eta0),'zero_eta_sumrule_error':abs(raw-.5),'lowest_gap_without_eta':gap,'lowest_gap_with_eta':gap+eta0*(dim-1)})
            ck.add('dipole_norm_'+str(shape)+'_'+str(group),abs(m0-2*av/dim)<1e-11,error=abs(m0-2*av/dim))
        angular.append(row)
    ck.add('angular_eta_recovery_at_converged_basis',max(g['eta_error'] for g in angular[-1]['groups'])<1e-9,error=max(g['eta_error'] for g in angular[-1]['groups']))
    ck.add('dipole_sumrule_converges',max(g['zero_eta_sumrule_error'] for g in angular[-1]['groups'])<1e-9)
    # Complete daughter-spectrum sum is independent of quadratic-form computation.
    shape=(6,4,4); op=old.operators(shape); es,ps,h=old.ground(op); v=ps[:,0]
    op1=old.operators(shape,ell=(1,0)); es1,ps1,h1=old.ground(op1); ed,vd=eigh(h1.toarray()); d=dipole(shape,0)@v
    spectral=float(np.sum((ed-es[0])*(vd.T@d)**2)); direct=float(d@(h1@d-es[0]*d))
    ck.add('independent_dipole_spectral_sum',abs(spectral-direct)<1e-11,error=abs(spectral-direct))
    out['angular_completion']=angular; out['angular_independent_spectral_error']=abs(spectral-direct)
    # IV. Different admissible CE parameters all satisfy forward/inverse consistency.
    family=[]
    for s,eps,k,lam,u0 in [(.5,.15,1.,10.,.5),(.65,.1,.6,8.,.4),(.8,.2,1.2,12.,.6)]:
        c=np.array([1.,1.,1.,-2*lam*u0,s-2*eps,s+eps,lam,0.,0.,k,k,0.])
        shape=(24,12,12); oo=old.operators(shape); ev,vec,hh=old.ground(oo,c); v=vec[:,0]
        big=tuple(i+2 for i in shape); be=old.embed(v,shape,big)
        rec=old.fixed_kinetic_selection(old.covariance(old.operators(big),be))
        delta=float(max(abs(rec['coefficients']-c)))
        family.append({'input':{'s0':s,'epsilon':eps,'kappa':k,'lambda':lam,'u0':u0},'mean_u':float(v@(oo[3]@v)),'energy_with_common_potential_convention':float(ev[0]+lam*u0*u0),'reconstructed_max_error':delta,'reconstructed':old.physical_values(rec['coefficients'])})
        ck.add('admissible_CE_family_'+str(k),s>2*eps and k>0 and lam>0 and delta<2e-7,error=delta)
    ck.add('CE_selfconsistency_does_not_select_unique_input',np.ptp([r['mean_u'] for r in family])>.02)
    out['CE_nonunique_selection_family']=family
    # V. Independent maximal-symmetry candidate, not reverse inference.
    u,l,h,a,b=sy.symbols('u l h a b',real=True)
    inv=u+l+h; symV=sy.expand(a*inv+b*inv**2)
    sym_coeff=[symV.coeff(u,1).subs({l:0,h:0}),symV.coeff(l,1).subs({u:0,h:0}),symV.coeff(h,1).subs({u:0,l:0}),symV.coeff(u,2),symV.coeff(l,2),symV.coeff(h,2),sy.diff(symV,u,l),sy.diff(symV,u,h),sy.diff(symV,l,h)]
    ck.add('maximal_O7_symmetry_fixes_ratios_not_scale',sym_coeff==[a,a,a,b,b,b,2*b,2*b,2*b])
    ck.add('maximal_symmetry_incompatible_with_original_nonzero_split',sy.simplify((sym_coeff[2]-sym_coeff[1])/3)==0 and old.DEFAULT[4]!=old.DEFAULT[5])
    out['independent_symmetry_candidate']={'symmetry':'O(7) on inherited real coordinates; an added candidate axiom, not standard-model gauge symmetry','potential':'a*(u+O_L+O_H)+b*(u+O_L+O_H)**2, b>0','coefficient_relations':[str(v) for v in sym_coeff],'epsilon':0,'portal':'2*b','remaining_free':'a,b and kinetic/energy normalization','verdict':'Does not preserve the original nonzero-splitting branch; not adopted.'}
    out['checks']=ck.rows; out['number_of_checks']=len(ck.rows); out['all_passed']=all(r['passed'] for r in ck.rows)
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--output',default='results.json'); opt=a.parse_args()
    res=run(); path=Path(opt.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS TOTAL',res['number_of_checks'])

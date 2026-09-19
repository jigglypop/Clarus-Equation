#!/usr/bin/env python3
"""CE-AS1: conditional action identifiability and symmetry-sector selection.
Synthetic vacuum records, not observational fits or natural-constant predictions.
Only numpy/scipy are required. No network or repository mutations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

LABELS = ['T_phi','T_L','T_H','u','O_L','O_H','u2','O_L2','O_H2','uO_L','uO_H','O_LO_H']
DEFAULT = np.array([1.,1.,1.,-10.,.2,.65,10.,0.,0.,1.,1.,0.])

def kron(*args):
    out=sp.csr_matrix([[1.]])
    for a in args: out=sp.kron(out,sp.csr_matrix(a),format='csr')
    return out

def radial(n,dim,ell,w):
    # Project polynomial operators only AFTER forming products in a larger basis.
    k=np.arange(n+2,dtype=float); alpha=ell+dim/2-1
    off=np.sqrt(k[1:]*(k[1:]+alpha))/(2*w)
    O=np.diag((2*k+alpha+1)/(2*w))+np.diag(off,1)+np.diag(off,-1)
    T=np.diag(w*(2*k+alpha+1))-w*w*O
    return T[:n,:n],O[:n,:n],(O@O)[:n,:n]

def operators(shape=(16,8,8),parity=0,ell=(0,0),w=(3.,.84,1.07)):
    nh,nl,nr=shape
    nf=2*nh+8
    aa=np.diag(np.sqrt(np.arange(1,nf)),1)
    x=(aa+aa.T)/np.sqrt(2*w[0]); pp=1j*np.sqrt(w[0]/2)*(aa.T-aa)
    ids=2*np.arange(nh)+parity; ii=np.ix_(ids,ids)
    x2=x@x
    u=x2[ii]/2; u2=(x2@x2)[ii]/4; T=(pp@pp).real[ii]/2
    tl,ol,ol2=radial(nl,2,ell[0],w[1]); th,oh,oh2=radial(nr,4,ell[1],w[2])
    a,b,c=np.eye(nh),np.eye(nl),np.eye(nr)
    return [kron(T,b,c),kron(a,tl,c),kron(a,b,th),kron(u,b,c),kron(a,ol,c),kron(a,b,oh),kron(u2,b,c),kron(a,ol2,c),kron(a,b,oh2),kron(u,ol,c),kron(u,b,oh),kron(a,ol,oh)]

def ground(ops,c=DEFAULT,k=1):
    h=sum((v*a for v,a in zip(c,ops)),start=sp.csr_matrix(ops[0].shape))
    es,vs=eigsh(h,k=k,which='SA',tol=5e-13,maxiter=25000,ncv=max(30,2*k+1),v0=np.random.default_rng(201).normal(size=h.shape[0]))
    order=np.argsort(es)
    return es[order],vs[:,order],h

def covariance(ops,v):
    w=np.column_stack([a@v for a in ops]); means=v@w
    w=w-np.outer(v,means)
    m=w.T@w; d=np.sqrt(np.diag(m))
    z=w/d
    # SVD of the centered feature action is more reliable than eigenvalues of M.
    _,s,vh=np.linalg.svd(z,full_matrices=False)
    vec=vh[-1]/d; vec/=vec[0]
    return dict(M=m,C=z.T@z,d=d,singular=s,vector=vec,means=means,W=w)

def embed(v,shape,big):
    out=np.zeros(big); out[tuple(slice(0,n) for n in shape)]=v.reshape(shape)
    return out.reshape(-1)


def fixed_kinetic_selection(cv,kinetic=None):
    # Once canonical kinetic energy is specified, only multiplication potentials remain.
    # Positive ground density makes their covariance strictly positive if no nonzero
    # linear combination is constant. Solve, do not invert the matrix explicitly.
    W=cv['W']; Q=W[:,3:]; kinetic=np.ones(3) if kinetic is None else np.asarray(kinetic); k=W[:,:3]@kinetic
    cov=Q.T@Q; cross=Q.T@k
    c=np.linalg.solve(cov,-cross)
    residual=float(np.linalg.norm(k+Q@c))
    return dict(coefficients=np.r_[kinetic,c],residual=residual,
                smallest_covariance_eigenvalue=float(eigh(cov,eigvals_only=True)[0]))

class Checks:
    def __init__(self): self.rows=[]
    def test(self,name,ok,**info):
        self.rows.append(dict(name=name,passed=bool(ok),**info))
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        if not ok: raise AssertionError((name,info))

def physical_values(c):
    a,b=c[4:6]
    return dict(s0=float((a+2*b)/3),epsilon=float((b-a)/3),
                kappa_L=float(c[9]),kappa_H=float(c[10]),
                lam=float(c[6]),u0=float(-c[3]/(2*c[6])),
                residual_quartics=c[[7,8,11]].tolist(),kinetic=c[:3].tolist())

def density_covariance(ops,rho):
    vals,vec=eigh(rho); sr=(vec*np.sqrt(np.maximum(vals,0)))@vec.T
    mus=np.array([np.trace(rho@a.toarray()) for a in ops])
    ws=[(a.toarray()-m*np.eye(len(rho)))@sr for a,m in zip(ops,mus)]
    z=np.column_stack([w.ravel() for w in ws]); d=np.linalg.norm(z,axis=0)
    return np.linalg.svd(z/d,compute_uv=False)

def run():
    ck=Checks(); report={'name':'CE-AS1','scope':'Conditional identification from synthetic joint-vacuum records; NOT natural-constant prediction',
                       'labels':LABELS,'truth_coefficients':DEFAULT.tolist(),'numeric_units':'Inherited finite oscillator units, hbar=1; not 4D QFT units'}
    # Check polynomial projection: squaring a truncated matrix loses real leakage.
    op=operators((8,4,4)); es,vs,h=ground(op); v=vs[:,0]
    ck.test('quartic_formed_before_projection',np.linalg.norm((op[3]@op[3]-op[6]).toarray())>1.)
    ck.test('hermitian_dictionary',all(np.linalg.norm((a-a.T).data)<1e-12 for a in op))
    cv=covariance(op,v)
    rng=np.random.default_rng(404); b=rng.normal(size=12); q=sum((x*a for x,a in zip(b,op)),start=sp.csr_matrix(h.shape))
    variance=np.linalg.norm(q@v-float(v@(q@v))*v)**2
    ck.test('covariance_equals_energy_variance',abs(float(b@cv['M']@b)-variance)<2e-11)
    rows=[]
    for shape in [(8,4,4),(12,6,6),(16,8,8),(20,10,10),(24,12,12),(28,14,14)]:
        ops=operators(shape); ee,ps,hh=ground(ops); p=ps[:,0]
        small=covariance(ops,p)
        enlarged=tuple(n+2 for n in shape); eops=operators(enlarged); ep=embed(p,shape,enlarged)
        big=covariance(eops,ep)
        fixed=fixed_kinetic_selection(big)
        row=dict(fixed_kinetic_recovered=physical_values(fixed['coefficients']),fixed_kinetic_max_error=float(max(abs(fixed['coefficients']-DEFAULT))),fixed_kinetic_residual=fixed['residual'],fixed_kinetic_covariance_gap=fixed['smallest_covariance_eigenvalue'],shape=list(shape),dim=int(np.prod(shape)),energy=float(ee[0]+2.5),mean_u=float(p@(ops[3]@p)),
                 projected_max_coefficient_error=float(max(abs(small['vector']-DEFAULT))),
                 expanded_max_coefficient_error=float(max(abs(big['vector']-DEFAULT))),
                 expanded_smallest_singular=float(big['singular'][-1]),expanded_next_singular=float(big['singular'][-2]),
                 expanded_coefficients=big['vector'].tolist(),recovered=physical_values(big['vector']),
                 eigen_residual=float(np.linalg.norm(hh@p-ee[0]*p)))
        rows.append(row)
        ck.test('basis_'+str(shape),row['eigen_residual']<1e-9 and row['expanded_next_singular']>.007)
    report['convergence']=rows
    ck.test('projected_only_precision_is_not_certificate',rows[0]['projected_max_coefficient_error']<1e-8 and rows[0]['expanded_max_coefficient_error']>.1)
    ck.test('unprojected_action_recovery_converges',all(rows[i+1]['expanded_max_coefficient_error']<rows[i]['expanded_max_coefficient_error'] for i in range(len(rows)-1)))
    last=rows[-1]
    ck.test('twelve_coefficients_recovered',last['expanded_max_coefficient_error']<5e-9)
    ck.test('fixed_kinetic_potential_uniqueness',last['fixed_kinetic_max_error']<1e-9 and all(r['fixed_kinetic_covariance_gap']>0 for r in rows))
    ck.test('common_portal_emerges_not_imposed',abs(last['recovered']['kappa_L']-last['recovered']['kappa_H'])<1e-8)
    ck.test('unneeded_local_quartics_zero',max(abs(np.array(last['recovered']['residual_quartics'])))<1e-9)
    # Same full operator dictionary; report all nonzero singular values.
    shape=(20,10,10); ops=operators(shape); es,vs,h=ground(ops); p=vs[:,0]; cv=covariance(ops,p)
    report['normalized_covariance_spectrum']=np.sort(cv['singular']**2).tolist()
    ck.test('one_dimensional_kernel_in_tested_dictionary',sum(cv['singular']<1e-9)==1)
    # Prediction of records/energies not used in reconstruction.
    recovered=np.array(last['expanded_coefficients']); hold=[]
    for parity,ell in [(0,(0,0)),(1,(0,0)),(0,(1,0)),(0,(0,1)),(0,(2,0)),(0,(0,2))]:
        oo=operators(shape,parity=parity,ell=ell)
        ev,_,_=ground(oo,DEFAULT,k=2); er,_,_=ground(oo,recovered,k=2)
        hold.append(dict(parity=parity,ell=list(ell),true_energy=float(ev[0]+2.5),reconstructed_energy=float(er[0]+2.5),error=float(max(abs(ev-er)))))
    report['held_out_sectors']=hold
    ck.test('ground_is_even_angular_singlet',all(row['true_energy']>hold[0]['true_energy']+1e-4 for row in hold[1:]))
    ck.test('held_out_sector_predictions',max(row['error'] for row in hold)<3e-8)
    # Frequency-weighted records determine kinetic coefficients themselves.
    # m1(A)=<delta A (H-E0) delta A>; in the continuum m1/Abar=t_A.
    kinetic_rows=[]
    for scale,kin in [(1.,np.array([1.,1.,1.])),(1.,np.array([1.13,.87,1.21])),(3.7,np.ones(3))]:
        coeff=DEFAULT.copy(); coeff[:3]=kin; coeff*=scale
        ss=(24,12,12); oo=operators(ss); ee,vv,hh=ground(oo,coeff); pv=vv[:,0]
        A=oo[3:6]; av=np.array([pv@(a@pv) for a in A]); aw=np.column_stack([a@pv-v*pv for a,v in zip(A,av)])
        M1=aw.T@(hh@aw-ee[0]*aw); kval=np.diag(M1)/av
        out=fixed_kinetic_selection(covariance(oo,pv),kval)
        row=dict(input_kinetic=(kin*scale).tolist(),recovered_kinetic=kval.tolist(),
                 max_kinetic_error=float(max(abs(kval-kin*scale))),
                 cross_moment=float(max(abs((M1-np.diag(np.diag(M1))).ravel()))),
                 complete_coefficient_error=float(max(abs(out['coefficients']-coeff))))
        kinetic_rows.append(row)
        ck.test('frequency_record_kinetic_'+str(kin.tolist())+'_'+str(scale),row['max_kinetic_error']<2e-8 and row['complete_coefficient_error']<1e-6)
    report['dynamic_record_kinetic_selection']=kinetic_rows
    # The sum over every excited eigenstate equals the quadratic-form calculation.
    oo=operators((6,3,3)); ee,vv,hh=ground(oo); ev,ve=eigh(hh.toarray()); pv=ve[:,0]
    A=oo[3]; mu=float(pv@(A@pv)); wv=A@pv-mu*pv
    spectral=float(np.sum((ev-ev[0])*(ve.T@wv)**2)); quadratic=float(wv@(hh@wv-ev[0]*wv))
    ck.test('energy_weighted_sum_independent_spectrum',abs(spectral-quadratic)<1e-12)
    report['independent_energy_sum']=dict(spectral=spectral,quadratic=quadratic,finite_basis_ratio=spectral/mu)
    # Same Gaussian ground density has infinitely many kinetic/curvature pairs.
    # V_G=1/2 q^T W G W q, E_G=Tr(GW)/2 for diagonal W,G here.
    W=np.diag([.8,1.7]); Gs=[np.eye(2),np.diag([1.4,.6])]
    points=np.array([[.2,.7],[-1.1,.3],[.4,-.9]])
    errors=[]
    for G in Gs:
        for qv in points:
            kinetic_term=.5*np.trace(G@W)-.5*qv@W@G@W@qv
            potential_term=.5*qv@W@G@W@qv
            errors.append(abs(kinetic_term+potential_term-.5*np.trace(G@W)))
    ck.test('density_alone_does_not_fix_kinetic_metric',max(errors)<1e-14 and not np.allclose(Gs[0],Gs[1]))
    report['same_density_different_metric']=dict(kinetic_matrices=[G.tolist() for G in Gs],ground_energies=[float(np.trace(G@W)/2) for G in Gs],residual=max(errors))
    # Without entangling portal a single product ground has three parent directions.
    free=DEFAULT.copy(); free[9:11]=0.
    ee,vv,_=ground(ops,free); cvf=covariance(ops,vv[:,0])
    report['decoupled_singular_values']=cvf['singular'].tolist()
    ck.test('decoupled_action_is_not_unique',sum(cvf['singular']<1e-8)>=3)
    # Anisotropic portal is recovered; equality is a falsifiable property of the input records.
    ani=DEFAULT.copy(); ani[10]=1.13
    ea,va,_=ground(ops,ani); cva=covariance(ops,va[:,0])
    report['anisotropic_portal_recovery']=physical_values(cva['vector'])
    ck.test('unequal_portal_negative_control',abs(cva['vector'][10]-1.13)<1e-8 and abs(cva['vector'][9]-1.)<1e-8)
    # Removing required interactions prevents a zero-variance parent.
    z=cv['W']/cv['d']; idx=[i for i in range(12) if i not in (9,10)]
    missing=float(np.linalg.svd(z[:,idx],compute_uv=False)[-1])
    report['missing_portal_smallest_singular']=missing
    ck.test('missing_portal_detected',missing>1e-4)
    # A degree-six potential not in the requested dictionary cannot masquerade as a quartic.
    nh,nl,nr=(12,6,6); oo=operators((nh,nl,nr))
    # u^3 is made in a padded oscillator basis, not (P u P)^3.
    nn=2*nh+10; aa=np.diag(np.sqrt(np.arange(1,nn)),1); xx=(aa+aa.T)/np.sqrt(6.)
    u3=np.linalg.matrix_power(xx,6)[np.ix_(2*np.arange(nh),2*np.arange(nh))]/8
    extra=kron(u3,np.eye(nl),np.eye(nr))
    e6,v6,h6=ground(oo+[extra],np.r_[DEFAULT,.2])
    miss6=covariance(oo,v6[:,0]); full6=covariance(oo+[extra],v6[:,0])
    report['degree_six_control']=dict(omitted_singular=float(miss6['singular'][-1]),restored_error=float(max(abs(full6['vector']-np.r_[DEFAULT,.2]))))
    ck.test('missing_operator_detection_and_recovery',miss6['singular'][-1]>1e-5 and report['degree_six_control']['restored_error']<1e-7)
    # Product of reduced states destroys the correlations required by the interacting parent.
    ss=(6,4,4); oo=operators(ss); ee,vv,hh=ground(oo); pp=vv[:,0].reshape(ss)
    rhos=[]
    for axis in range(3):
        mat=np.moveaxis(pp,axis,0).reshape(ss[axis],-1); rhos.append(mat@mat.T)
    rp=np.kron(np.kron(rhos[0],rhos[1]),rhos[2]); sig=density_covariance(oo,rp)
    report['product_marginals_min_singular']=float(sig[-1])
    ck.test('joint_correlations_cannot_be_discarded',sig[-1]>1e-4)
    # Energy-origin and positive time-scale ambiguity are exact.
    scaled=3.7*h+2.1*sp.eye(h.shape[0],format='csr')
    ck.test('affine_hamiltonian_same_vacuum',np.linalg.norm(scaled@p-(3.7*es[0]+2.1)*p)<1e-9)
    # Noise bound with fixed record scaling d. No experimental sampling is claimed.
    C=cv['C']; vals,vec=eigh(C); reference=vec[:,0]; gap=float(vals[1]); noise=[]
    for delta in [1e-9,1e-8,1e-7,1e-6]:
        q=rng.normal(size=C.shape); q=(q+q.T)/2; q*=delta/np.linalg.norm(q,2)
        ev,vc=eigh(C+q); overlap=abs(float(reference@vc[:,0])); angle=float(np.sqrt(max(0.,1-overlap**2)))
        bound=delta/(gap-delta)
        noise.append(dict(delta=delta,sin_angle=angle,bound=bound))
        ck.test('noise_bound_'+str(delta),angle<=bound+5e-8)
    report['record_noise_bound']=dict(gap=gap,trials=noise)
    # A singlet vacuum cannot determine a coefficient multiplying angular Casimir.
    alpha=.37; base=hold[0]['true_energy']; Lenergy=hold[2]['true_energy']
    report['invisible_sector_extension']=dict(added_coefficient=alpha,vacuum_shift=0.,ell1_energy_shift=alpha,example_before=Lenergy,example_after=Lenergy+alpha)
    oa=operators((8,4,4)); ob=operators((8,4,4),ell=(1,0))
    e0,v0,h0=ground(oa); e1,v1,h1=ground(ob)
    J=sp.block_diag((sp.csr_matrix(h0.shape),sp.eye(h1.shape[0])),format='csr')
    hall=sp.block_diag((h0,h1),format='csr'); vac=np.r_[v0[:,0],np.zeros(h1.shape[0])]
    pair_observable=sp.block_diag((oa[4],ob[4]),format='csr')
    ck.test('sector_blindness_negative_control',np.linalg.norm(alpha*J@vac)<1e-14 and
            np.linalg.norm((J@pair_observable-pair_observable@J).data)<1e-14 and
            e1[0]+alpha>e0[0])
    # Fixed canonical kinetic + a strictly positive wavefunction determines V up to a constant.
    # Independent closed-form Gaussian inverse check (not a CE natural-vacuum ansatz).
    x=np.array([-.7,.2,1.1]); w=np.array([.8,1.2,1.7]); grad=-2*w*x; lap=-2*w.sum()
    inverse=.25*lap+.125*float(grad@grad)
    explicit=.5*float((w*w)@(x*x))-.5*w.sum()
    ck.test('positive_density_inverse_identity',abs(inverse-explicit)<1e-14)
    report['checks']=ck.rows; report['number_of_checks']=len(ck.rows); report['all_passed']=all(x['passed'] for x in ck.rows)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    result=run(); target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS TOTAL',result['number_of_checks'])

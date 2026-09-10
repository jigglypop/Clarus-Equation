"""CE-GF2: variational rank audit and an eight-scalar Berry representation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ce_common_isotropic_frame import coherent


def tensor(items):
    out=np.array([1.+0j])
    for item in items:out=np.kron(out,item)
    return out


def berry(cutoff):
    q=np.array([.1,-.2,.3,.4]);p=np.array([.2,.1,-.3,.2])
    values=[];dq=[];dp=[];tails=[];connection=[]
    for qa,pa in zip(q,p):
        alpha=(qa+1j*pa)/np.sqrt(2)
        v,dqa,tail=coherent(alpha,1/np.sqrt(2),cutoff)
        _,dpa,_=coherent(alpha,1j/np.sqrt(2),cutoff)
        values.append(v);dq.append(dqa);dp.append(dpa);tails.append(tail)
        connection.append([float((1j*np.vdot(v,dqa)).real+pa/2),
                           float((1j*np.vdot(v,dpa)).real+qa/2)])
    expected=np.column_stack([p,np.zeros(4)])
    error=float(np.max(abs(np.array(connection)-expected)))
    phase=np.exp(-.5j*np.dot(p,q));full=phase*tensor(values)
    normalization=float(abs(np.vdot(full,full)-1))
    direct=[]
    for a in range(4):
        pair=[]
        for derivatives,phase_derivative in [(dq,-.5j*p[a]),(dp,-.5j*q[a])]:
            pieces=values.copy();pieces[a]=derivatives[a]
            dv=phase*tensor(pieces)+phase_derivative*full
            pair.append(float((1j*np.vdot(full,dv)).real))
        direct.append(pair)
    factorization=float(np.max(abs(np.array(direct)-connection)))
    assert normalization<1e-12 and factorization<1e-12
    if cutoff==16:assert error<1e-10
    return {'cutoff':cutoff,'dimension':len(full),'connection':connection,
            'connection_max_error':error,'normalization_error':normalization,
            'full_vs_factorized_error':factorization,'product_tail_union_bound':sum(tails)}


def main():
    spurious=[]
    for x in (0.,math.pi/6,math.pi/2):
        # At y=z=0; orthogonality also holds for arbitrary y,z.
        r_jac=np.array([[0,1,0,0],[0,0,math.sin(x),-math.cos(x)],
                        [0,0,0,0],[0,0,0,0]],float)
        maxwell=np.array([0,0,math.cos(x),math.sin(x)])
        contractions=r_jac@maxwell
        assert np.max(abs(contractions))<1e-12
        assert abs(np.linalg.norm(maxwell)-1)<1e-12
        spurious.append({'x':x,'maxwell_residual':maxwell.tolist(),
                         'scalar_stationarity_contractions':contractions.tolist()})
    # q is an invertible coframe, independent of p; delta p spans delta A even at p=0.
    coframes=[np.eye(4),np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]],float)]
    variation=[]
    target=np.array([.2,-.3,.4,-.1])
    for j in coframes:
        delta_p=np.linalg.solve(j,target)
        error=float(np.linalg.norm(j@delta_p-target));assert error<1e-12
        variation.append({'determinant':float(np.linalg.det(j)),
                          'vacuum_variation_rank':int(np.linalg.matrix_rank(j)),
                          'variation_reconstruction_error':error})
    # q=x, p=(0,0,x,t) gives A=x dy+t dz and the previous parallel E,B example.
    dp=np.zeros((4,4),int);dp[1,2]=1;dp[0,3]=1
    field=dp-dp.T
    expected=np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]])
    assert np.array_equal(field,expected)
    rows=[berry(cutoff) for cutoff in (4,8,16)]
    here=Path(__file__).resolve()
    sources=[here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py']
    out={'candidate':'CE-GF2','four_scalar_spurious_solutions':spurious,
         'eight_scalar_variations':variation,'parallel_field':field.tolist(),'berry':rows,
         'source_sha256':{s:hashlib.sha256(here.with_name(s).read_bytes()).hexdigest() for s in sources},
         'limits':['Maxwell action and Lorentzian metric supplied, not induced',
                   'Four-scalar variational reduction fails on degenerate fields',
                   'Eight-scalar result is local with invertible coframe',
                   'Original common metric and quantum scalar corrections not yet reconciled']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()

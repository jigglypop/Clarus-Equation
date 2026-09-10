"""CE-GF1: exact rank/Pfaffian obstruction for three-coordinate gauge pullbacks."""
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path


def rank(matrix):
    a=[[Fraction(v) for v in row] for row in matrix];pivot=0
    for col in range(len(a[0])):
        found=next((r for r in range(pivot,len(a)) if a[r][col]),None)
        if found is None:continue
        a[pivot],a[found]=a[found],a[pivot]
        scale=a[pivot][col];a[pivot]=[v/scale for v in a[pivot]]
        for row in range(len(a)):
            if row!=pivot:
                scale=a[row][col]
                a[row]=[v-scale*w for v,w in zip(a[row],a[pivot])]
        pivot+=1
        if pivot==len(a):break
    return pivot


def pullback(j,f):
    return [[sum(j[mu][a]*f[a][b]*j[nu][b]
                 for a in range(len(f)) for b in range(len(f)))
             for nu in range(4)] for mu in range(4)]


def pfaffian(f):return f[0][1]*f[2][3]-f[0][2]*f[1][3]+f[0][3]*f[1][2]


def epsilon_contraction(f):
    total=0
    for p in itertools.permutations(range(4)):
        sign=(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))
        total+=sign*f[p[0]][p[1]]*f[p[2]][p[3]]
    return total


def main():
    jacobians=[[[1,0,0],[0,1,0],[0,0,1],[1,2,3]],
               [[2,-1,3],[1,4,0],[-2,1,5],[3,-2,1]],
               [[1,2,3],[2,4,6],[0,0,0],[-1,-2,-3]],
               [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]
    targets=[[[0,-1,0],[1,0,0],[0,0,0]],
             [[0,2,-3],[-2,0,5],[3,-5,0]],
             [[0,0,1],[0,0,-2],[-1,2,0]]]
    rows=[]
    for i,j in enumerate(jacobians):
        for k,f in enumerate(targets):
            full=pullback(j,f);pf=pfaffian(full);rr=rank(full);epsilon=epsilon_contraction(full)
            assert pf==0 and epsilon==0 and rr<=2
            rows.append({'jacobian':i,'target':k,'matrix':full,'pfaffian':pf,'rank':rr,
                         'levi_civita_contraction':epsilon})
    target4=[[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]
    assert pfaffian(target4)==1 and rank(target4)==4 and epsilon_contraction(target4)==8
    # Constant coefficients imply every derivative in both local Maxwell equations is zero.
    derivatives=[[[0]*4 for _ in range(4)] for _ in range(4)]
    bianchi=[derivatives[a][b][c]+derivatives[b][c][a]+derivatives[c][a][b]
             for a,b,c in itertools.combinations(range(4),3)]
    metric=[-1,1,1,1]
    maxwell=[sum(metric[mu]*metric[nu]*derivatives[mu][mu][nu] for mu in range(4)) for nu in range(4)]
    assert bianchi==[0]*4 and maxwell==[0]*4
    # At a constant q background, scaling dq by t scales F by t^2 and F^2 by t^4.
    base=pullback(jacobians[0],targets[0]);base_norm=sum(v*v for row in base for v in row)
    scaling=[]
    for t in (1,2,3,4):
        scaled=pullback([[t*v for v in row] for row in jacobians[0]],targets[0])
        norm=sum(v*v for row in scaled for v in row)
        assert norm==t**4*base_norm
        scaling.append({'amplitude':t,'euclidean_F_squared':norm})
    here=Path(__file__).resolve()
    out={'candidate':'CE-GF1','script_sha256':hashlib.sha256(here.read_bytes()).hexdigest(),
         'pullbacks':rows,'counterexample':{'F':target4,'pfaffian':1,'rank':4,
                                           'levi_civita_contraction':8,'bianchi':bianchi,'maxwell':maxwell},
         'four_coordinate_identity_pullback':pullback([[int(i==j) for j in range(4)] for i in range(4)],target4),
         'quartic_scaling':scaling,
         'limits':['Applies to smooth three-coordinate pullback connections at fixed other parameters',
                   'Does not rule out three force projections with additional underlying state variables',
                   'No observational fitting or Maxwell dynamics of an extended model claimed']}
    assert out['four_coordinate_identity_pullback']==target4
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':len(rows),'all_pullback_pfaffians_zero':True,
                     'counterexample':out['counterexample'],'quartic_scaling':scaling},indent=2))


if __name__=='__main__':main()

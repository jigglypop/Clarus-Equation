"""Outward interval proof for the six-coordinate JS3 B-branch minimum.

The analytic derivative and infinite-tail bounds are stated in chapter 32.
This does not certify a preferred mass hierarchy or a UV completion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from mpmath import iv

from derive_dynamic_mass import DynamicModel, BASE_SOURCE


def lo(x):
    return float(np.nextafter(float(x.a), -np.inf))


def hi(x):
    return float(np.nextafter(float(x.b), np.inf))


def ab(x):
    return max(abs(lo(x)),abs(hi(x)))


def main():
    folder=Path(__file__).resolve().parent
    data=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert data['source_sha256']==sha(folder/'derive_dynamic_mass.py')
    assert data['base_source_sha256']==sha(BASE_SOURCE)
    assert data['new_inputs']['sextic_eta']==12.
    iv.dps=45
    point=data['values']['selected_point']
    u,r,a0,q,aq,sigma=[iv.mpf(str(v)) for v in point]
    assert point[2]==.5 and point[4]==-point[3]
    y=iv.mpf('.18')*u*iv.exp(2*r)
    constant=3/(64*iv.pi**6)
    p=constant*iv.exp(-6*r)
    zero=iv.mpf(0)
    w0=wy=wyy=sn=sr=srr=zero
    wa,way=[zero]*3,[zero]*3
    waa=[[zero for _ in range(3)] for _ in range(3)]
    nmax,nmass=8192,256
    for n in range(1,nmax+1):
        wn,weight=2*iv.pi*n,iv.mpf(1)/n**5
        ct,st=iv.cos(wn*q),iv.sin(wn*q)
        d=1 if n%2==0 else -1
        trace=d+2*ct
        f=fy=fyy=zero
        if n<=nmass:
            z=wn*iv.sqrt(y)
            ez=iv.exp(-z)
            f=ez*(1+z+z*z/3)
            fy=-wn**2*(1+z)*ez/6
            fyy=wn**4*ez/12
            zn=wn*sigma*iv.exp(r)
            en=iv.exp(-zn)
            sn+=weight*en*(1+zn+zn*zn/3)
            sr+=weight*(-en*zn*zn*(1+zn)/3)
            srr+=weight*en*(zn**4-2*zn**3-2*zn**2)/3
        w0+=weight*(trace*trace-1-2*f*trace)
        wy+=-2*weight*fy*trace
        wyy+=-2*weight*fyy*trace
        ga=-2*wn*st*(trace-f)*weight
        gya=2*fy*wn*st*weight
        wa[1]+=ga
        wa[2]-=ga
        way[1]+=gya
        way[2]-=gya
        h0=2*wn*wn*(1-(trace-f)*d)*weight
        h1=2*wn*wn*(1-(trace-f)*ct)*weight
        h01=2*wn*wn*d*ct*weight
        h12=2*wn*wn*(2*ct*ct-1)*weight
        local=[[h0,h01,h01],[h01,h1,h12],[h01,h12,h1]]
        for i in range(3):
            for j in range(3):
                waa[i][j]+=local[i][j]
    er,dy=iv.exp(-r),y/u
    h,hp=10*(u-iv.mpf('.5'))**2,20*(u-iv.mpf('.5'))
    v=12*constant*er*sigma**6
    full=w0+4*sn
    gradient=[er*hp+p*dy*wy,-er*h-v+p*(-6*full+2*y*wy+4*sr)]
    gradient += [p*z for z in wa]+[(6*v+4*p*sr)/sigma]
    matrix=[[zero for _ in range(6)] for _ in range(6)]
    matrix[0][0]=20*er+p*dy*dy*wyy
    matrix[0][1]=matrix[1][0]=-er*hp+p*dy*(-4*wy+2*y*wyy)
    matrix[1][1]=er*h+v+p*(36*full-20*y*wy+4*y*y*wyy-48*sr+4*srr)
    matrix[1][5]=matrix[5][1]=(-6*v+4*p*(srr-6*sr))/sigma
    matrix[5][5]=(30*v+4*p*(srr-sr))/sigma**2
    for i in range(3):
        matrix[0][i+2]=matrix[i+2][0]=p*dy*way[i]
        matrix[1][i+2]=matrix[i+2][1]=p*(-6*wa[i]+2*y*way[i])
        for j in range(3):
            matrix[i+2][j+2]=p*waa[i][j]
    _,_,numeric,_=DynamicModel(eta=12.,nmax=nmax).evaluate(point)
    _,basis=np.linalg.eigh(numeric)
    basis=[[iv.mpf(str(basis[i,j])) for j in range(6)] for i in range(6)]
    gram=[[sum(basis[k][i]*basis[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
    rotated=[[sum(basis[k][i]*matrix[k][l]*basis[l][j] for k in range(6) for l in range(6))
              for j in range(6)] for i in range(6)]
    lower=lambda a:min(lo(a[i][i])-sum(ab(a[i][j]) for j in range(6) if i!=j) for i in range(6))
    upper=lambda a:max(hi(a[i][i])+sum(ab(a[i][j]) for j in range(6) if i!=j) for i in range(6))
    assert lower(gram)>.99
    center=lower(rotated)/(upper(gram)+1e-14)-1e-12
    gt=hi(p*iv.sqrt((iv.mpf(12)/nmax**4)**2+3*(4*iv.pi/nmax**3)**2))+1e-40
    ht=hi(p*(iv.mpf(72)/nmax**4+48*iv.sqrt(3)*iv.pi/nmax**3+12*iv.pi**2/nmax**2))+1e-40
    gnorm=np.nextafter(np.sqrt(sum(ab(v)**2 for v in gradient)),np.inf)+gt+1e-23
    radius,lipschitz=1e-8,3000.
    convexity=center-ht-lipschitz*radius-1e-14
    assert .49<point[0]-radius<point[0]+radius<.51
    assert -1.08<point[1]-radius<point[1]+radius< -1.04
    assert 1.35<point[5]-radius<point[5]+radius<1.40
    assert convexity>0 and gnorm<convexity*radius
    result={'claim':'strict six-variable local minimum of JS3 selected-determinant approximation, excluding the new scalar own one-loop winding',
            'branch':'B: a=(1/2,q,-q), two light modes and one heavy mode',
            'center':point,'ball_radius':radius,'interval_decimal_precision':iv.dps,
            'massless_winding_cutoff':nmax,'massive_winding_cutoff':nmass,
            'interval_gradient':[str(v) for v in gradient],
            'interval_Hessian':[[str(v) for v in row] for row in matrix],
            'center_Hessian_lower':center,'Hessian_tail_bound':ht,'gradient_tail_bound':gt,
            'total_gradient_upper':float(gnorm),'third_derivative_bound':lipschitz,
            'box_u':[.49,.51],'box_r':[-1.08,-1.04],'box_sigma':[1.35,1.40],
            'full_ball_Hessian_lower':convexity,
            'boundary_outward_margin':float(convexity*radius-gnorm),
            'distance_to_unique_stationary_point_upper':float(np.nextafter(gnorm/convexity,np.inf)),
            'certified':True,'full_goal_complete':False,
            'limitations':['not global uniqueness','not desired light-one/heavy-two ordering',
                           'no UV or all-loop completion','remaining sextic and Yukawa inputs',
                           'new scalar own same-order winding is excluded'],
            'source_sha256':sha(Path(__file__)),'model_source_sha256':data['source_sha256'],
            'base_source_sha256':data['base_source_sha256'],
            'input_results_sha256':sha(folder/'results.json')}
    (folder/'minimum_certificate.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['certified','full_ball_Hessian_lower',
                      'total_gradient_upper','distance_to_unique_stationary_point_upper']},indent=2))


if __name__=='__main__':
    main()

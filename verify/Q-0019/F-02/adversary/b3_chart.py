"""B3: 비선형 재매개화 (차트 의존). s = l^p (p=2 제곱길이, p=1/2), 넓이 변수 A_t.
질문: 정성 판정 (std<1, [3,2]>1) 이 다른 자연 차트에서 유지되는가."""
import json, math, itertools, numpy as np, indep as I
pts=I.regular_points(2.0); cells=I.refine([tuple(I.BV)],pts)
b0=np.full(10,math.sqrt(2.0)); hc=I.richardson(I.coarse_action,b0)
cx=I.build(cells,pts,hc)
def clus(v,tol=1e-4):
    out=[]
    for x in np.sort(np.asarray(v).real):
        if out and abs(out[-1][0]-x)<tol*max(1,abs(x)): out[-1].append(float(x))
        else: out.append([float(x)])
    return [{"lambda2":float(np.mean(c)),"mult":len(c)} for c in out]
res={"baseline":clus(I.pencil(cx["N"],cx["M"]))}
# 차트 s=l^p: l = s^(1/p). Hessian_s = (dl/ds)^T H_l (dl/ds) + diag(g_l * d2l/ds2)
def chart_spec(p):
    hess_s=[]
    for c,l,k in zip(cells,cx["lens"],cx["kap"]):
        H=I.richardson(lambda v,k=k,c=c: I.cell_action(c,v,k), l)
        # gradient
        g=np.zeros(10); h=2e-3
        for i in range(10):
            a=l.copy();a[i]+=h; b=l.copy();b[i]-=h
            g[i]=(I.cell_action(c,a,k)-I.cell_action(c,b,k))/(2*h)
        s=l**p
        dlds=(1.0/p)*s**(1.0/p-1.0)
        d2lds2=(1.0/p)*(1.0/p-1.0)*s**(1.0/p-2.0)
        hess_s.append(np.diag(dlds)@H@np.diag(dlds)+np.diag(g*d2lds2))
    # coarse
    Hc=hc.copy()
    gc=np.zeros(10); h=2e-3
    for i in range(10):
        a=b0.copy();a[i]+=h; b=b0.copy();b[i]-=h
        gc[i]=(I.coarse_action(a)-I.coarse_action(b))/(2*h)
    sc=b0**p; dc=(1/p)*sc**(1/p-1); d2c=(1/p)*(1/p-1)*sc**(1/p-2)
    Hc_s=np.diag(dc)@Hc@np.diag(dc)+np.diag(gc*d2c)
    # mismatch 좌표: 차트 s 의 편차. T_a 는 metric -> l; s 편차 = p l^{p-1} dl
    N=np.zeros((50,50))
    for a,(t,H) in enumerate(zip(cx["T"],hess_s)):
        # metric -> s: ds = p l^{p-1} dl = p l^{p-1} T dm
        S=np.diag(p*cx["lens"][a]**(p-1))@t
        N[10*a:10*a+10,10*a:10*a+10]=S.T@H@S
    # L in s-chart: coarse s 편차 = p b0^{p-1} * (평균 dl)
    Ls=np.diag(p*b0**(p-1))@cx["L"]
    M=Ls.T@Hc_s@Ls
    return clus(I.pencil(N,M)), [int(np.sum(np.linalg.eigvalsh(Hc_s)>1e-9)),int(np.sum(np.linalg.eigvalsh(Hc_s)<-1e-9))]
for p in (2.0, 0.5, 3.0, 1.0):
    c,s=chart_spec(p)
    res[f"p={p}"]={"clusters":c,"Hc_chart_signature":s}
# 넓이 차트: 10 삼각형 넓이 (4-simplex 는 삼각형 10개, 길이 10개 - 국소 전단사)
def area_chart():
    hess_a=[]; jac=[]
    for c,l,k in zip(cells,cx["lens"],cx["kap"]):
        # A(l) 야코비안
        def areas(v):
            return np.array([I.area(tuple(c[i] for i in tl), c, v) for tl in I.TRIS])
        h=2e-3
        Jm=np.zeros((10,10))
        for i in range(10):
            a=l.copy();a[i]+=h; b=l.copy();b[i]-=h
            Jm[:,i]=(areas(a)-areas(b))/(2*h)
        Ji=np.linalg.inv(Jm)   # dl/dA
        H=I.richardson(lambda v,k=k,c=c: I.cell_action(c,v,k), l)
        g=np.zeros(10)
        for i in range(10):
            a=l.copy();a[i]+=h; b=l.copy();b[i]-=h
            g[i]=(I.cell_action(c,a,k)-I.cell_action(c,b,k))/(2*h)
        # 2계 항: d2l/dA2 는 무시하지 않고 FD 로 (l(A) 를 직접 못 쓰므로 근사: 
        # 정확 항 = Ji^T H Ji + sum_m g_m * d2 l_m/dA2 ; d2l/dA2 = -Ji (d2A/dl2 contracted) Ji
        # d2A/dl2 를 삼각형별로 FD
        d2A=[]
        for n in range(10):
            HH=np.zeros((10,10))
            for i in range(10):
                for j in range(i,10):
                    if i==j:
                        a=l.copy();a[i]+=h;b=l.copy();b[i]-=h
                        HH[i,i]=(areas(a)[n]-2*areas(l)[n]+areas(b)[n])/h**2
                    else:
                        aa=l.copy();aa[i]+=h;aa[j]+=h
                        bb=l.copy();bb[i]+=h;bb[j]-=h
                        cc=l.copy();cc[i]-=h;cc[j]+=h
                        dd=l.copy();dd[i]-=h;dd[j]-=h
                        HH[i,j]=HH[j,i]=(areas(aa)[n]-areas(bb)[n]-areas(cc)[n]+areas(dd)[n])/(4*h**2)
            d2A.append(HH)
        gA = Ji.T@g   # dS/dA
        corr=np.zeros((10,10))
        for n in range(10):
            corr += -gA[n]*(Ji.T@d2A[n]@Ji)
        hess_a.append(Ji.T@H@Ji+corr)
        jac.append(Jm)
    # coarse 도 마찬가지 (경계 10 삼각형)
    def careas(v):
        return np.array([I.area(tuple(I.BV[i] for i in tl), I.BV, v) for tl in I.TRIS])
    h=2e-3; Jc=np.zeros((10,10))
    for i in range(10):
        a=b0.copy();a[i]+=h;b=b0.copy();b[i]-=h
        Jc[:,i]=(careas(a)-careas(b))/(2*h)
    Jci=np.linalg.inv(Jc)
    gc=np.zeros(10)
    for i in range(10):
        a=b0.copy();a[i]+=h;b=b0.copy();b[i]-=h
        gc[i]=(I.coarse_action(a)-I.coarse_action(b))/(2*h)
    d2Ac=[]
    for n in range(10):
        HH=np.zeros((10,10))
        for i in range(10):
            for j in range(i,10):
                if i==j:
                    a=b0.copy();a[i]+=h;b=b0.copy();b[i]-=h
                    HH[i,i]=(careas(a)[n]-2*careas(b0)[n]+careas(b)[n])/h**2
                else:
                    aa=b0.copy();aa[i]+=h;aa[j]+=h
                    bb=b0.copy();bb[i]+=h;bb[j]-=h
                    ccx=b0.copy();ccx[i]-=h;ccx[j]+=h
                    dd=b0.copy();dd[i]-=h;dd[j]-=h
                    HH[i,j]=HH[j,i]=(careas(aa)[n]-careas(bb)[n]-careas(ccx)[n]+careas(dd)[n])/(4*h**2)
        d2Ac.append(HH)
    gAc=Jci.T@gc
    corrc=np.zeros((10,10))
    for n in range(10): corrc += -gAc[n]*(Jci.T@d2Ac[n]@Jci)
    Hc_A=Jci.T@hc@Jci+corrc
    N=np.zeros((50,50))
    for a,(t,H) in enumerate(zip(cx["T"],hess_a)):
        S=jac[a]@t     # metric -> A
        N[10*a:10*a+10,10*a:10*a+10]=S.T@H@S
    LA=Jc@cx["L"]
    M=LA.T@Hc_A@LA
    return clus(I.pencil(N,M)), [int(np.sum(np.linalg.eigvalsh(Hc_A)>1e-9)),int(np.sum(np.linalg.eigvalsh(Hc_A)<-1e-9))]
ca,sa=area_chart()
res["area_chart"]={"clusters":ca,"Hc_chart_signature":sa}
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b3_chart.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))

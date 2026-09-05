"""완전 독립 재구현: 4-simplex Regge 작용, 이면각, 등분할 kappa, 펜슬.
카드 스크립트의 모듈(regge_one_to_five_*)을 전혀 쓰지 않는다."""
import math, itertools
import numpy as np

BV = (0,1,2,3,4)
EDGES = tuple(itertools.combinations(BV,2))   # 10

def regular_points(sq=2.0):
    # 정규 4-simplex: 제곱 변 sq, 무게중심 원점 (표준 구성)
    P = np.eye(5)
    P = P - P.mean(axis=0)
    # 현재 제곱 변 = 2 -> 스케일
    cur = np.sum((P[0]-P[1])**2)
    P = P*math.sqrt(sq/cur)
    # 4차원으로 (5x5 rank4)
    U,S,Vt = np.linalg.svd(P)
    Q = (U[:,:4]*S[:4])
    Q = Q - Q.mean(axis=0)
    return {i:Q[i] for i in range(5)}

def points_from_sq(sq):
    """제곱 변 10개(EDGES 순)에서 좌표 복원 (Cayley-Menger/Gram)."""
    d = np.zeros((5,5))
    for k,(i,j) in enumerate(EDGES):
        d[i,j]=d[j,i]=sq[k]
    G = np.array([[0.5*(d[0,a]+d[0,b]-d[a,b]) for b in range(1,5)] for a in range(1,5)])
    w,V = np.linalg.eigh(G)
    X = V*np.sqrt(np.maximum(w,0))
    pts = np.vstack((np.zeros(4), X))
    pts = pts - pts.mean(axis=0)
    return {i:pts[i] for i in range(5)}

def cm_volume_sq(sqd, n):
    """n-simplex (n+1 점) 의 Cayley-Menger 부피 제곱. sqd: (n+1)x(n+1) 제곱거리."""
    m = n+1
    B = np.ones((m+1,m+1)); B[0,0]=0.0
    B[1:,1:] = sqd
    fac = ((-1)**(n+1))/(2**n * math.factorial(n)**2)
    return fac*np.linalg.det(B)

def edge_map(cell):
    return {tuple(sorted(e)): k for k,e in enumerate(itertools.combinations(cell,2))}

def sqd_of(sub, cell, lens):
    em = edge_map(cell)
    m = len(sub)
    D = np.zeros((m,m))
    for a in range(m):
        for b in range(a+1,m):
            k = em[tuple(sorted((sub[a],sub[b])))]
            D[a,b]=D[b,a]=lens[k]**2
    return D

def area(tri, cell, lens):
    return math.sqrt(max(cm_volume_sq(sqd_of(tri,cell,lens),2),0.0))

def dihedral(tri, cell, lens):
    """4-simplex 안 삼각형 hinge 의 이면각: sin theta = (4/3)*V4*A2/(V3a*V3b) 대신
    표준식 cos theta = ... 여기서는 부피 공식으로: 
    theta = arcsin( (4/3) * V4 * A_tri / (V3_1 * V3_2) ) 를 arccos 로 안정화."""
    rest = [v for v in cell if v not in tri]
    t1 = tuple(tri)+ (rest[0],)
    t2 = tuple(tri)+ (rest[1],)
    V4 = math.sqrt(max(cm_volume_sq(sqd_of(cell,cell,lens),4),0.0))
    A  = area(tri,cell,lens)
    V31 = math.sqrt(max(cm_volume_sq(sqd_of(t1,cell,lens),3),0.0))
    V32 = math.sqrt(max(cm_volume_sq(sqd_of(t2,cell,lens),3),0.0))
    s = (4.0/3.0)*V4*A/(V31*V32)
    s = min(1.0,max(-1.0,s))
    # 예각/둔각 판별: 법선 내적으로. 좌표를 써서 cos 계산
    X = embed(cell, lens)
    idx = {v:n for n,v in enumerate(cell)}
    # hinge 평면의 직교여공간(2차원) 안에서 두 반평면 사이 각
    o = X[idx[tri[0]]]
    T = np.array([X[idx[tri[1]]]-o, X[idx[tri[2]]]-o])
    # 직교 사영
    Qb,_ = np.linalg.qr(T.T)
    def perp(v):
        w = v-o
        return w - Qb@(Qb.T@w)
    u1 = perp(X[idx[rest[0]]]); u2 = perp(X[idx[rest[1]]])
    c = float(u1@u2/(np.linalg.norm(u1)*np.linalg.norm(u2)))
    return math.acos(min(1.0,max(-1.0,c)))

def embed(cell, lens):
    D = sqd_of(cell,cell,lens)
    G = np.array([[0.5*(D[0,a]+D[0,b]-D[a,b]) for b in range(1,len(cell))] for a in range(1,len(cell))])
    w,V = np.linalg.eigh(G)
    X = V*np.sqrt(np.maximum(w,0))
    return np.vstack((np.zeros(X.shape[1]), X))

TRIS = tuple(itertools.combinations(range(5),3))

def cell_action(cell, lens, kap):
    """S_a = sum_t A_t (kappa_t - theta_t)."""
    tot = 0.0
    for n,tl in enumerate(TRIS):
        tri = tuple(cell[i] for i in tl)
        tot += area(tri,cell,lens)*(kap[n]-dihedral(tri,cell,lens))
    return tot

def refine(cells, pts):
    out=[]
    for cell in cells:
        lab = max(pts)+1
        pts[lab]=np.mean([pts[v] for v in cell],axis=0)
        for om in cell:
            out.append((lab,)+tuple(v for v in cell if v!=om))
    return out

def kappas(cells, bverts=BV):
    cnt={}
    for c in cells:
        for t in itertools.combinations(c,3):
            k=tuple(sorted(t)); cnt[k]=cnt.get(k,0)+1
    out=[]
    for c in cells:
        ks=[]
        for t in itertools.combinations(c,3):
            k=tuple(sorted(t))
            tot = math.pi if all(v in bverts for v in k) else 2*math.pi
            ks.append(tot/cnt[k])
        out.append(np.asarray(ks))
    return out

def hess_fd(f, x, h):
    n=len(x); H=np.zeros((n,n)); 
    f0=f(x)
    for i in range(n):
        for j in range(i,n):
            if i==j:
                xp=x.copy(); xp[i]+=h; xm=x.copy(); xm[i]-=h
                H[i,i]=(f(xp)-2*f0+f(xm))/h**2
            else:
                a=x.copy(); a[i]+=h; a[j]+=h
                b=x.copy(); b[i]+=h; b[j]-=h
                c=x.copy(); c[i]-=h; c[j]+=h
                d=x.copy(); d[i]-=h; d[j]-=h
                H[i,j]=H[j,i]=(f(a)-f(b)-f(c)+f(d))/(4*h**2)
    return H

def richardson(f,x,h=2e-3):
    return (4*hess_fd(f,x,h/2)-hess_fd(f,x,h))/3.0

def coarse_action(bl, kap_c=None):
    """coarse 단일 4-simplex 작용: 경계 변 10, kappa = pi (경계 hinge, 1 cell)."""
    cell=BV
    kap = np.full(10, math.pi)
    return cell_action(cell, bl, kap)

def sym_basis():
    B=[]
    for i in range(4):
        m=np.zeros((4,4)); m[i,i]=1.0; B.append(m)
    for i in range(4):
        for j in range(i+1,4):
            m=np.zeros((4,4)); m[i,j]=m[j,i]=1/math.sqrt(2); B.append(m)
    return B
BASIS = sym_basis()

def build(cells, pts, hc, chart="length"):
    n=len(cells)
    kap = kappas(cells)
    lens = [np.asarray([np.linalg.norm(pts[i]-pts[j]) for i,j in itertools.combinations(c,2)]) for c in cells]
    hess = [richardson(lambda v,k=k: cell_action(c,v,k), l) for c,l,k in zip(cells,lens,kap)]
    T=[]
    for c,l in zip(cells,lens):
        t=np.zeros((10,10))
        for r,(i,j) in enumerate(itertools.combinations(c,2)):
            u=pts[i]-pts[j]
            t[r]=[float(u@b@u)/l[r] for b in BASIS]
        T.append(t)
    dim=10*n
    N=np.zeros((dim,dim)); Nl=np.zeros((dim,dim))
    for a,(t,h) in enumerate(zip(T,hess)):
        N[10*a:10*a+10,10*a:10*a+10]=t.T@h@t
        Nl[10*a:10*a+10,10*a:10*a+10]=h
    Ll=np.zeros((10,dim))
    for k,(i,j) in enumerate(EDGES):
        own=[]
        for a,c in enumerate(cells):
            if i in c and j in c:
                em=edge_map(c); own.append((a,em[tuple(sorted((i,j)))]))
        for a,r in own: Ll[k,10*a+r]=1.0/len(own)
    P=np.zeros((dim,dim))
    for a,t in enumerate(T): P[10*a:10*a+10,10*a:10*a+10]=t
    L=Ll@P
    return dict(N=N,Nl=Nl,L=L,Ll=Ll,M=L.T@hc@L,Ml=Ll.T@hc@Ll,hess=hess,T=T,lens=lens,kap=kap,cells=cells)

def pencil(N,M,tol=1e-6):
    ev=np.linalg.eigvals(np.linalg.solve(N,M))
    ev=ev[np.abs(ev)>tol]
    return ev[np.argsort(ev.real)]

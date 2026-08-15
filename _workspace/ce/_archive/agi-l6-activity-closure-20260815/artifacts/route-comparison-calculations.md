# L6-H1 연산자 대수 (상세)

이 파일은 `12-routes.md`의 수치만 적는다. 정리 지위가 아니다.
한 스텝 유리는 `verify_l6_one_step.py` 두 경로가 같다.
재현: `python artifacts/route_l6_operators.py` → `route_l6_operators.txt`.

명목값: $\lambda=5/2$, $\rho=1/5$, $\delta=1/10$, $\theta_D=3/4$,
$\kappa=1/4$, $r(3/4)=81/16$. $q=3/4$는 $F_{1/4}$의 표지 고정점
($s=\mu$ 상쇄). $q'$는 관측량이 아니다.

등록쌍: $P_{\star}=(1/2,49/99,3/4)$, $P_{\circ}=(7/15,49/99,3/4)$.
둘 다 $U_0=\operatorname{int}(B_c)$ 내부. $1-b=50/99$.

## 한 스텝 $u=1$

전개: $1+r(1-m)-\lambda(1-b)=57/16-(81/16)m+(5/2)b$.

$$
\widetilde m_{\star}=\frac{7187}{6336},\qquad
m'_{\star}=\frac{7187}{12672},\qquad
b'_{\star}=\frac{491}{990}.
$$

$$
\widetilde m_{\circ}=\frac{16891}{14850},\qquad
m'_{\circ}=\frac{16891}{29700},\qquad
b'_{\circ}=\frac{133}{270}.
$$

둘 다 $\widetilde m>3/4$이므로 분열. $d_{\star}=d_{\circ}=1$.

경계 차이는 성장항 없이

$$
b'_{\star}-b'_{\circ}
=
\rho(1-b)\Bigl(\frac12-\frac7{15}\Bigr)
=
\frac15\cdot\frac{50}{99}\cdot\frac1{30}
=
\frac1{297}.
$$

질량 차이 $m'_{\star}-m'_{\circ}=-1487/950400$.

비트 예측기: $\sigma_{\star}=\sigma_{\circ}=1$이므로 공역은 한 점.
참 $(m',b')$가 다르면 그 한 점이 둘을 동시에 맞출 수 없다.

## $T=32$ 점유, $u=1$

인용 $O$-$E1$: $U_0$의 모든 $(m,b)$에 대해 $q_0=3/4$, $u=1$이면
점유 참. 두 점은 그 상자 안. 새 헐 없음.

부동 증인 (헐 아님): 둘 다 $o_{32}=1$, 분열 $32$, $t=0,\ldots,31$에서
$R_0$을 안 빠짐. 종점 근사 $m_{32}\approx 0.572$, $b_{32}\approx 0.534$
($Z_+$ 근방). 종점 좌표는 갈리나 점유 비트는 같다.

## 재귀 $u_t=m_t$ ($\sigma=1$)

$t=0$ 드라이브: $1/2$ 대 $7/15$. 한 스텝 (정확, 비분열):

$$
P_{\star}\mapsto\Bigl(\frac{6355}{12672},\frac{491}{990}\Bigr),\qquad
\widetilde m_{\star}=\frac{6355}{12672}<3/4.
$$

$$
P_{\circ}\mapsto\Bigl(\frac{34559}{74250},\frac{133}{270}\Bigr),\qquad
\widetilde m_{\circ}=\frac{34559}{74250}<3/4.
$$

$b'$는 $u=1$과 같다 (경계가 $u$에 안 붙음). $m'$는 비분열이라
$\widetilde m$ 자체.

$T=32$ 부동 (포락 아님):

- $P_{\circ}$: $m$이 단조 감소. $t=11$에서 $\widetilde m<0$,
  $m\leftarrow 0$ 흡수. $o_{32}=0$. $b_{32}\approx 0.045$.
- $P_{\star}$: $m$이 증가. $t=12$에서 $m>3/5$라 $R_0$ 탈출.
  분열 $0$. $m_{32}\approx 0.710$, $b_{32}\approx 0.584$, $o_{32}=0$.

점유 읽기는 $0$ 대 $0$. 활동 읽기는 소멸 대 고질량.

## 재귀 $u_t=\mathbf 1[(m_t,b_t)\in R_0]$

시작 점유 $1$. $u=1$ 궤적이 $R_0$을 안 빠지면 $u_t\equiv 1$.
이 쌍의 부동 증인은 붕괴: 종점·점유가 상수 $u=1$과 같다.

질량 구간만 보는 $u_t=\mathbf 1[m_t\in[2/5,3/5]]$도 이 증인에서는
같다.

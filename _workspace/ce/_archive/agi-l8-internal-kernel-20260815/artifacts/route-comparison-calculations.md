# L8-H1 맵 대수 (상세)

이 파일은 `12-routes.md`의 수치만 적는다. 정리 지위가 아니다.
재현: `python artifacts/route_l8_maps.py` → `route_l8_maps.txt`.

명목값: $\lambda=5/2$, $\rho=1/5$, $\delta=1/10$, $\theta_D=3/4$,
$\kappa=1/4$, $r(3/4)=81/16$. $W=I$. 역할 $(S,A)=(L,R)$.
등록 플럭스 $E=e^{(2)}=(0,1)$. 한 스텝, 세척 없음.
비트 $\sigma=I=1$은 이 한 스텝에서 유지 (`L8-D2`).

등록 점 (평가 전 이름, `L8-D4`):

$$
P_{\star}=\Bigl(\tfrac12,\tfrac{49}{99},\tfrac34\Bigr),\qquad
P_{\circ}=\Bigl(\tfrac{7}{15},\tfrac{49}{99},\tfrac34\Bigr).
$$

$$
H_{\star}=(0,e^{(2)},P_{\star},P_{\star},1,1),\qquad
H_{\circ}=(0,e^{(2)},P_{\circ},P_{\circ},1,1).
$$

드라이브: $u^{\mathrm{A}}=I\,u_I(e^{(2)})=1$, $u^{\mathrm{S}}=u_I^{\mathrm{S}}(e^{(2)})=0$.

## 현재 비트

$U_0\subset R_0$. 두 점의 $(m,b)$는 $R_0$ 안에 있다. 따라서

$$
o^{\mathrm{A}}(H_{\star})=o^{\mathrm{A}}(H_{\circ})=1.
$$

$\sigma$와 $I$도 $S$에서 $1$. $o^{\mathrm{S}}$도 $1$ (같은 큐브).

## 액션 한 스텝 $u=1$ (인용 $L6$-$E1$, 재유도 없음)

$$
(m'^{\mathrm{A}},b'^{\mathrm{A}})_{\star}
=
\Bigl(\frac{7187}{12672},\frac{491}{990}\Bigr),\qquad
(m'^{\mathrm{A}},b'^{\mathrm{A}})_{\circ}
=
\Bigl(\frac{16891}{29700},\frac{133}{270}\Bigr).
$$

표지 맵은 $q=3/4$ 고정점. $q'=3/4$ 양쪽. 스크립트 검산만.

## 센서 한 스텝 $u=0$ (비교 숫자, 새 헐 아님)

$b=49/99$에서

$$
1-\lambda(1-b)=1-\frac52\cdot\frac{50}{99}=-\frac{26}{99}<0.
$$

따라서 $\widetilde m=[m(-26/99)]_+=0$, $m'^{\mathrm{S}}=0$ 양쪽.
경계 갱신은 $u$에 독립:

$$
b'^{\mathrm{S}}_{\star}=\frac{491}{990},\qquad
b'^{\mathrm{S}}_{\circ}=\frac{133}{270}.
$$

액션 $b'$와 같다. $\Delta b'=1/297$.

## $\Phi$ 튜플

$$
\Phi(H_{\star})
=
\bigl(1,e^{(2)},(0,491/990,3/4),(7187/12672,491/990,3/4),1,1\bigr),
$$

$$
\Phi(H_{\circ})
=
\bigl(1,e^{(2)},(0,133/270,3/4),(16891/29700,133/270,3/4),1,1\bigr).
$$

$\Phi(H_{\star})\neq\Phi(H_{\circ})$ (액션 $(m',b')$와 센서 $b'$).
공통 슬롯: $t'=1$, $E'=e^{(2)}$, $q'=3/4$, $\sigma'=I'=1$, $m'^{\mathrm{S}}=0$.

## 공역 표

| 맵 | 공역 | $H_{\star}$ 값 | $H_{\circ}$ 값 | $\Phi$와 등식 |
|---|---|---|---|---|
| $K=\Phi$ | $H$ 공간 | $\Phi(H_{\star})$ | $\Phi(H_{\circ})$ | 구성으로 성립 |
| $K_{\mathrm{bit}}=o^{\mathrm{A}}$ | $\{0,1\}$ | $1$ | $1$ | 종류 불일치 |
| $K_{\mathrm{act}}$ | $[0,1]^2$ | $L6$-$E1$ $\star$ | $L6$-$E1$ $\circ$ | 종류 불일치 (슬롯 부족) |
| $K_3$ | $H\times[0,1]^3$ | 셋째 큐브 추가 | 셋째 큐브 추가 | 공역이 $H$가 아님 |

비트 값 $1$을 $H$에 넣는 임베딩은 계약이 인가하지 않는다.
인가된 임베딩이 없어도 등식 $1=\Phi(H)$는 성립할 수 없다.

$S\to\{0,1\}$ 맵은 $4$개. 상수 $0$, 상수 $1$, 두 비상수.
모두 오른쪽이 비트가 아니라서 같은 종류 시험에서 죽는다.
look-elsewhere $4$, 연속 조정 $0$.

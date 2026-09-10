# 55. 확장 상태 매장의 Einstein 변분 조건 — CE-GR6

## 55.1 계산 전 등록

**[확장 계량 후보의 필요조건]** 2026-09-10. CE-LC1의 네 상태 좌표만으로 만든
계량은 가역 영역에서 평탄하다. 더 많은 실수 좌표 $Y^A(x)$를 가진 한 시간
부호의 평탄한 주변 공간에서 $g_{\mu\nu}=\eta_{AB}\partial_\mu Y^A\partial_\nu Y^B$
를 쓰는 확장을 검사한다. $Y$를 양자 상태에서 유도한 결과나 주변 차원을 실제
시공간 차원으로 해석한 주장은 아직 없다.

- 우선 가장 유리한 대조로 $S[g]=\frac12\int\sqrt{-g}(R-2\Lambda)$를
  공급한다. Planck 계수는 단위 1, 물질은 0이다. 이 Einstein–Hilbert 항의
  미시적 기원·UV 계수는 이번에 유도하지 않는다.
- 이 작용을 계량 대신 모든 $Y^A(x)$에 대해 변분한다. 단순한 계량 좌표변환이
  되는 조건과 추가 해가 남는 조건을 같은 변분에서 구한다.
- 5차원 대조 A는 $Y=(t,n^1,n^2,n^3,n^4)$, $n\in S^3$, $\Lambda=1$인
  단위 원통이다. 내재 곡률·Einstein 잔차와 매장 변분 잔차를 검사한다.
- 5차원 대조 B는 $\Lambda=0$,
  $a(\chi)=\sin^3\chi$, $Y^0(\chi)=\frac32(\chi-\sin\chi\cos\chi)$,
  $Y^i=a(\chi)n^i$이며 $\chi=\pi/4,\pi/3,\pi/2$를 고정한다.
  proper time, Gauss 곡률과 FRW 곡률, 두 종류의 운동방정식을 독립 대조한다.
- 충분조건 대조는 독립 정상 변분이 대칭 계량 10성분을 모두 만들 수 있는
  full-rank 매장이다. $v_a=e_\mu$ 네 개와 $e_\mu+e_\nu$ 여섯 개,
  $r=0.1$, $H=\eta_4-r^2\sum_av_av_a^{\mathsf T}$를 고정한다.
  $L^{\mathsf T}\eta_4L=H$인 $L$을 구성하고 24차원 배경
  $Y=(Lx,r\cos(v_a\cdot x),r\sin(v_a\cdot x))$를 검사한다.
- 검산점은 $x=(0,0,0,0),(0.1,-0.2,0.3,-0.1)$이다. 가정한 주변 signature,
  유도 계량·정상 직교성·두 번째 기본형식의 rank·Gauss 곡률을 확인한다.
  10개의 대칭 계량 기저 모두를 정상 변분으로 복원한다.
- 배경의 함수형만 변화시키는 제한된 ansatz가 아니라 $Y^A(x)$의 모든 독립
  변분을 허용하는 후보임을 고정한다. 정상 rank는 유리수와 행렬 rank로 대조한다.
  나머지 대수의 허용오차는 $10^{-10}$, 피팅 없음.
- Einstein 변분의 조건부 동등성을 양자 측도·숨은 성분·네 힘의 공통 작용 또는
  두 graviton 편광의 양자 검증으로 승격하지 않는다.

## 55.2 좌표를 늘려도 변분은 자동으로 같아지지 않는다

**[정의·조건부 유도]** 주변 계량은 $\eta_{AB}$, 시공간 signature는
$(-,+,+,+)$로 둔다. 매장 사상 $Y:M^4\to\mathbb R^{1,N-1}$의 유도 계량과
두 번째 기본형식은

$$
g_{\mu\nu}=\partial_\mu Y\cdot\partial_\nu Y,\qquad
b^A_{\mu\nu}=\partial_\mu\partial_\nu Y^A-
\Gamma^\lambda_{\mu\nu}\partial_\lambda Y^A,\qquad
b_{\mu\nu}\cdot\partial_\lambda Y=0.
$$

물질을 포함한 일반식에서
$\mathcal E^{\mu\nu}=M^2(G^{\mu\nu}+\Lambda g^{\mu\nu})-T^{\mu\nu}$라 쓰면,
경계 안에 지지되는 변분에 대해

$$
\delta S=-\frac12\int\sqrt{-g}\,\mathcal E^{\mu\nu}\delta g_{\mu\nu},
\qquad
\delta g_{\mu\nu}=2\partial_{(\mu}Y_A\partial_{\nu)}\delta Y^A.
$$

부분적분 뒤 매장 운동방정식은

$$
\nabla_\mu(\mathcal E^{\mu\nu}\partial_\nu Y_A)=0.
$$

물질 운동방정식과 공변 보존을 사용하면 $\nabla_\mu\mathcal E^{\mu\nu}=0$이고,
따라서

$$
\boxed{\mathcal E^{\mu\nu}b^A_{\mu\nu}=0}
\tag{55.1}
$$

을 얻는다. 진공에서는 Bianchi 항등식만으로 이 단계가 성립한다. 작용이 $Y$에
오직 $g(Y)$를 통해 의존한다는 조건이 필요하다. 같은 $Y$가 별도의 Berry 연결,
질량 또는 기록 항에도 직접 들어가면 그 항의 변분도 남으므로 (55.1)을 그대로
적용하지 않는다.

이는 기존 Regge–Teitelboim 매장 중력의 방정식이며, Einstein 방정식보다 넓은
해집합을 가진다는 사실도 알려져 있다. 이번 연구는 이 알려진 경계를 상태 계량
후보에 적용하고 명시적 반례와 충분조건을 검산한다.
[Sheykin–Paston, 2014](https://arxiv.org/abs/1402.1121),
[Paston–Zaitseva, 2021, §1–3](https://arxiv.org/html/2111.04188).

## 55.3 비평탄 계량이지만 Einstein 해가 아닌 정적 반례

**[반례 A]** $Y=(t,n)$, $n\cdot n=1$인 5차원 원통을 쓴다.
기준 길이를 단위 1로 쓰면

$$
ds^2=-dt^2+d\Omega_3^2,\qquad R=6,\qquad
b_{00}=0,\qquad b_{ij}=-g_{ij}.
$$

법선은 $(0,n)$인 단위 공간꼴 벡터다. 직교 frame에서

$$
G_{\hat\mu\hat\nu}=\operatorname{diag}(3,-1,-1,-1),\qquad
\Lambda=1\quad\Longrightarrow\quad
\mathcal E_{\hat\mu\hat\nu}=\operatorname{diag}(2,0,0,0).
$$

따라서 $\mathcal E^{\mu\nu}b_{\mu\nu}=0$이지만 $\mathcal E\ne0$다.
원통 반지름만 변화시키는 제한 변분의 결과가 아니다. 모든 $Y^A$ 변분에서
접선 방정식은 보존 항등식이고, 독립 법선 방향의 방정식도 0이므로 전체 매장
방정식을 만족한다. 그러나 공급한 진공 Einstein 방정식에는 실패한다.

길이 단위 $a_0$를 복원하면 $R=6/a_0^2$, $\Lambda=1/a_0^2$,
$\mathcal E_{\hat0\hat0}=2/a_0^2$다. 이 값은 관측을 맞춘 상수가 아니라
두 운동방정식의 차이를 확인하기 위해 사전 지정한 대조다.

## 55.4 우주상수가 0이어도 남는 동적 반례

**[반례 B·유도]** proper time $t$에서

$$
Y=(F(t),a(t)n),\qquad \dot F=f=\sqrt{1+\dot a^2},\qquad
ds^2=-dt^2+a^2d\Omega_3^2
$$

로 둔다. 단위 법선 $N=(\dot a,fn)$에서

$$
b_{00}=\frac{\ddot a}{f},\qquad b_{ij}=-\frac f a g_{ij},
\qquad G_{00}=\frac{3f^2}{a^2},\qquad
G_{ij}=-\left(\frac{2\ddot a}{a}+\frac{f^2}{a^2}\right)g_{ij}.
$$

$\Lambda=0$인 매장 방정식은

$$
G^{\mu\nu}b_{\mu\nu}
=\frac{3f}{a^3}(3a\ddot a+\dot a^2+1)=0,
\qquad (1+\dot a^2)a^{2/3}=C
\tag{55.2}
$$

로 줄어든다. 등록한 $C=1$의 해는 $0<\chi<\pi$에서

$$
a=\sin^3\chi,\qquad
t=2-3\cos\chi+\cos^3\chi,\qquad
F=\frac32(\chi-\sin\chi\cos\chi).
$$

$dt/d\chi=3\sin^3\chi>0$이고 직접 매장 미분으로도
$g_{\chi\chi}=-9\sin^6\chi$를 얻는다. 따라서

$$
\dot a=\cot\chi,\qquad
\ddot a=-\frac{1}{3\sin^5\chi},\qquad
3a\ddot a+\dot a^2+1=0.
$$

반면 내재 곡률과 Einstein 잔차는

$$
R=\frac4{\sin^8\chi},\qquad
G_{\hat\mu\hat\nu}=
\frac1{\sin^8\chi}\operatorname{diag}\left(3,-\frac13,-\frac13,-\frac13\right)
\ne0.
$$

| 등록 지점 $\chi$ | $R$ | $\max\lvert\mathcal E_{\hat\mu\hat\nu}\rvert$ | $\lvert\mathcal E^{\mu\nu}b_{\mu\nu}\rvert$ |
|---|---:|---:|---:|
| $\pi/4$ | 64 | 48 | $4.974\times10^{-14}$ |
| $\pi/3$ | 12.6419753 | 9.48148148 | $3.997\times10^{-15}$ |
| $\pi/2$ | 4 | 3 | $2.220\times10^{-16}$ |

잔차를 형식적 유체로 옮겨 쓰면 $\rho=3/\sin^8\chi$,
$p=-1/(3\sin^8\chi)$, $p/\rho=-1/9$이고
$\dot\rho+3(\dot a/a)(\rho+p)=0$이다. 이는 진공 Einstein 극한에 남는 추가
해를 설명하는 표기다. 실제 물질이나 암흑부문의 검증된 예측으로 해석하지 않는다.
끝점의 특이점을 건너는 해도 주장하지 않는다.

## 55.5 Einstein 변분을 되찾는 충분조건

**[조건부 정리]** 비퇴화 계량의 각 점에서 법선 두 번째 기본형식들이
$\operatorname{Sym}^2(T^*M)$의 10성분을 모두 생성한다고 하자.
그러면 (55.1)은 $\mathcal E^{\mu\nu}=0$과 동등하다.

증명은 쌍대 선형사상으로 충분하다. 접선과 법선 변분을
$\delta Y=\xi^\mu\partial_\mu Y+\phi^aN_a$로 나누면

$$
\delta g_{\mu\nu}=2\nabla_{(\mu}\xi_{\nu)}-2\phi^a b_{a\mu\nu}.
\tag{55.3}
$$

법선 방향의 rank가 10이면 접선 변분 없이도 임의의 대칭 $\delta g$를 만들 수
있다. 그 모든 변분에 대해 작용이 정지하려면 $\mathcal E=0$이어야 한다.
동일하게, 10차원 대칭 공간을 생성하는 모든 $b_a$에 수축하여 0이 되는
$\mathcal E$는 0뿐이다. 역방향은 즉시 성립한다.

이 점별 충분조건은 $N-4\ge10$, 즉 $N\ge14$인 주변 공간을 요구한다.
이는 모든 GR 재표현에 필요한 최소 물리 차원이라는 주장이 아니다.
추가 초기 제약이나 다른 변수·변분 규칙을 쓰는 방식은 별도다. 여기서 쓰는 것은
기존 free embedding 조건이다.
[Paston–Zaitseva, 2021, §3](https://arxiv.org/html/2111.04188).

**[명시적 충분조건 대조]** 이 rank가 Lorentzian 계량과 양립함을 보이기 위해
$r=0.1$, $v_a=e_\mu$ 네 개와 $e_\mu+e_\nu$ 여섯 개를 쓴다. 좌표·삼각함수
위상은 선택한 기준 단위에서 무차원이다. $H=\eta_4-r^2\sum_av_av_a^{\mathsf T}$는
음의 고유값 하나와 양의 고유값 세 개를 가진다. 실제로
$\|r^2\sum_av_av_a^{\mathsf T}\|=0.07<1$이므로 부호가 바뀌지 않는다.
따라서 $L^{\mathsf T}\eta_4L=H$인 가역 실수 행렬 $L$을 골라

$$
Y(x)=\left(Lx,\{r\cos(v_a\cdot x),r\sin(v_a\cdot x)\}_{a=1}^{10}\right)
\in\mathbb R^{1,23}
\tag{55.4}
$$

로 둘 수 있다. $Lx$가 가역이므로 주기 함수에 의한 자기 교차도 없다.
유도 계량은 모든 $x$에서

$$
g=H+r^2\sum_av_av_a^{\mathsf T}=\eta_4.
$$

각 원의 방사 단위 벡터 $N_a$는 서로 직교하며 모든 접선에 수직이다.
계량이 상수이므로 $\Gamma=0$이고

$$
b_{a\mu\nu}=-r(v_a)_\mu(v_a)_\nu.
$$

네 대각 dyad와 여섯 쌍 dyad는 모든 대칭 4×4 행렬을 생성한다. 특히 임의의
$h_{\mu\nu}$에 대해

$$
\phi_{\mu\nu}=\frac{h_{\mu\nu}}{2r}\quad(\mu<\nu),\qquad
\phi_\mu=\frac{h_{\mu\mu}-\sum_{\nu\ne\mu}h_{\mu\nu}}{2r}
$$

로 택하면 $-2\sum_a\phi_a b_a=h$다. $\phi(x)$가 위치에 의존해도
(55.3)의 법선 항에는 $\partial\phi$가 나타나지 않는다.

각 $b_a$는 rank 1이므로 Gauss 식

$$
R_{\mu\nu\rho\sigma}
=\sum_a(b_{a\mu\rho}b_{a\nu\sigma}-b_{a\mu\sigma}b_{a\nu\rho})
$$

에서 각 항이 0이다. 따라서 내재적으로 평탄하면서도 정상 변분의 rank는 10일 수
있다. 평평한 초평면 매장의 $b=0$과 구별되는 대조다. 24차원은 계산하기 쉬운
충분조건 예시이며 최소 차원으로 주장하지 않는다. 일반 곡률 시공간의 전역 매장
존재를 이 한 구성으로 증명한 것도 아니다.

## 55.6 독립 계산과 연구 판정

**[계산 검산]** [실행 코드](../../verify/ce_embedding_einstein_variation.py)와
[산출 JSON](../../verify/ce_embedding_einstein_variation.json)에 다음을 보존했다.

- 두 5차원 반례에서 매장 미분의 법선 투영으로 $b$를 얻고, Gauss 곡률에서
  계산한 Einstein 텐서를 FRW 식과 대조했다. 우주상수 0의 해에서는 proper-time
  변환·일차 적분·잔차의 공변 보존도 검사했다.
- 24차원 구성의 두 등록점에서 주변 부호, 계량과 그 미분, 법선 직교성,
  두 번째 기본형식과 Gauss 곡률을 확인했다. rank는 정수 dyad 행렬의 유리수
  소거와 부동소수점 행렬 rank로 각각 10을 얻었다.
- 10개 계량 기저 각각을 정상 변분으로 복원했다. $-2\phi^ab_a$ 계산과
  실제 $Y+\epsilon\sum_a\phi^aN_a$의 접선을 미분한 계산이 모두 일치했다.
  코드 해시·Python/NumPy 판본을 JSON에 기록했다.

최대 대수 항등식 잔차는 $5.685\times10^{-14}$로 등록 허용오차
$10^{-10}$ 안이다. Einstein 잔차가 크다는 반례 판정은 수치 오차에 의존하지
않으며 위의 닫힌 식으로도 확인된다. 관측 자료·피팅은 사용하지 않았다.

**[기각·조건부 생존]** ‘확장 상태 좌표가 비평탄 계량을 만들고 EH 항을 가지므로
자동으로 GR이다’라는 판본은 두 반례로 기각한다. 작용의 $Y$ 의존이 계량만을
통하고 full normal rank 10이 유지되는 영역에서는 고전 Einstein 변분을 복원하는
판본이 조건부로 생존한다. rank가 보존되는 동역학과 허용 영역의 전역성은 아직
별도로 입증해야 한다.

**[미완성]** (55.4)의 주변 법선은 20차원이고 일차 계량 변분의 핵은 10차원이다.
계량을 바꾸지 않는 이 변분을 양자 경로적분에서 어떤 중복성으로 나눌지, 유도
측도·Jacobian과 제약이 무엇인지는 계산하지 않았다. 고전 방정식의 동등성만으로
두 graviton 편광의 양자 이론이나 전체 확률 보존이 증명되지는 않는다.

또 $Y$, 주변 Lorentz 부호, EH 작용과 그 계수는 입력이다. 이를 비관측 성분의
동역학에서 얻는 유도, 같은 작용의 내부 세 힘·기록과의 결합, 전체 UV·공동 RMSE가
남아 있다. 따라서 OBJ-01·OBJ-04의 필요 검증 하나를 구체화한 결과이며 네 목표의
달성 선언이 아니다.

관련 선행 문서: [상태 원뿔과 단일 좌표 계량](54_양의_상태_원뿔과_Lorentz_확률_보존.md),
[곡률 계량의 부호·작용 경계](53_곡률_계량의_부호와_자기_운동항의_위상화.md),
[무한 보완공간의 곡률 응답](42_무한_보완공간의_곡률_응답_검사.md).

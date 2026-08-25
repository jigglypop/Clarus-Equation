# Research contract — self-nonidentity kinetic dark sector

Status: IN_PROGRESS

Current-Route: R2-k4-ADM-completion

Claim-Ceiling: R0의 균일 배경 IR EFT 계산은 완료됐지만, 비선택 경로·0차원
기록에서 $\Pi_{\rm fold}$를 산출하는 미시 다리는 아직 미완성이다. R1은
명시적 Gaussian 저장소를 포함하는 조건부 열린계 완성을 검증 중이다. R2는
고정계량 성장량을 게이지 의존 진단으로 내리고, 유한 관측구간의 clock+GR
단일-clock $Q_s,c_s^2$, super-Hubble pump 및 두-미분 strong-coupling
hierarchy를 조건부로 통과시켰다. $k^4$ 항이 strong coupling 전에 켜질 수
있는 계수창도 찾았지만, 다성분 ADM·bath pole·정확한 $k^4$ 완성·관측 폐쇄는
아직 열려 있다.

PREDECESSOR:
../self-measurement-dark-energy-closure-20260825
../_archive/dimensionless-self-measurement-time-20260825
../_archive/zero-dimensional-fold-memory-field-20260825
../_archive/measurement-record-one-way-compatibility-20260825
../_archive/quantum-neighbor-bootstrap-dark-sector-20260825

## 1. 질문과 실패한 선행식

검증할 새 명제는 다음이다.

> “흐름”을 장의 절대값이 아니라, 한 국소 상태가 직전 상태와 같지 않은
> 변화율의 반복으로 정의한다. 0차원 측정 사건의 미래향 순서를 이 흐름의
> 경계자료로 보내고, 비선택 경로의 잔여를 한 유효 장의 암흑에너지 및
> 암흑물질 readout으로 만들 수 있는가?

선행식

$$
V(\Theta)=\rho_*e^{-\Theta}
$$

은

$$
(\Theta,\rho_*)\mapsto(\Theta+\Delta,\rho_*e^\Delta)
$$

에서 정확히 불변이므로, 자유 $\rho_*$ shooting 아래 operational
$\Theta$ 원점을 식별하지 못한다. 이번 run은 이 식을 살리지 않고 교체한다.

## 2. operational 정의

0차원 사건의 순서열을 $z_0\prec z_1\prec\cdots$라 쓰고
$z_{n+1}\ne z_n$을 자기-비동일성의 한 step으로 정의한다. 연속 coarse-graining
clock $T(x)$에 대해

$$
X:=-\frac12g^{\mu\nu}\nabla_\mu T\nabla_\nu T
$$

를 변화율 불변량으로 쓴다. $T$는 시간 차원, $X$는 무차원이다. 미래향
timelike branch는

$$
T|_{\Sigma_*}=0,\qquad
n^\mu\nabla_\mu T|_{\Sigma_*}>0,\qquad
J|_{\Sigma_*}:=P_Xn^\mu\nabla_\mu T|_{\Sigma_*}>0,\qquad
\delta|_{\Sigma_*}>0
$$

로 고정한다. 이는 시간 화살의 동역학적 증명이 아니라 0D 사건 순서를
4D Cauchy 자료로 보내는 one-way matching 공리다. 진짜 비가역 retarded
법칙은 별도 환경 influence kernel 없이는 주장하지 않는다.

homogeneous branch에서는 retained fold inventory를

$$
\boxed{
a_i^3J_i=\Pi_{\rm fold}
=\mathcal R_\Pi[\mu_F,C_{\rm self};\rho_\infty,\Gamma,L_c]>0
}
$$

로 정의한다. $\Pi_{\rm fold}$는 두 번째 벌크장이 아니라 초기 canonical
current 자료다. 이 map과 그 차원 있는 normalization은 미시적으로 유도되지 않은
matching 공리이며, 같은 재고를 별도 에너지로 다시 더하지 않는다.

고정 dephasing 선행 모형의 무차원 측정 깊이와 연결할 때는

$$
\theta=\Gamma T,\qquad
c(\theta)=1-e^{-\theta}=1-e^{-\Gamma T}
$$

를 쓴다. $\Gamma$는 정보 좌표를 물리 clock에 연결하는 독립 변환 scale이다.

## 3. 비교할 두 action

### A. shift-symmetric kinetic condensate

$$
\boxed{
P_A(X)=M^4\left[
-1+\frac{\kappa}{2}\left(\frac{X}{X_*}-1\right)^2
\right]}
$$

### B. anchored soft measurement clock

$$
\boxed{
P_B(T,X)=\rho_\infty\left[
\frac{\kappa}{2}\left(\frac{X}{X_*}-1\right)^2
-\left(1-e^{-\Gamma T}\right)
\right]}
$$

$\rho_\infty$는 에너지밀도, $\kappa$와 $X_*$는 무차원,
$\Gamma$는 역시간이다. $T=0$은 $\Sigma_*$가 고정하는 물리적 기록 anchor다.
B는 kinetic condensate를 soft clock constraint로 쓰며, 증가하는
$V(T)=\rho_\infty(1-e^{-\Gamma T})$를 같은 장의 누적 opportunity readout으로
쓴다.

## 4. 필수 유도

각 후보에서 다음을 직접 유도한다.

1. Euler--Lagrange 식, stress tensor와 on-shell 공변 보존.
2. 평탄 FLRW의 $\rho$, $p$, current 식.
3. $X=X_*\ne0$에서 또는 그 점근에서 $w=-1$이 가능한지.
4. $\delta=X/X_*-1>0$의 작은 편차가 $a^{-3}$ dust와 작은 sound speed를
   주는지.
5. ghost 조건 $P_X+2XP_{XX}>0$, gradient 조건 $P_X\ge0$,
   $c_s^2=P_X/(P_X+2XP_{XX})$.
6. $T\mapsto T+\Delta$가 amplitude 재정의로 흡수되는지.
7. 초기 current가 양의 branch를 전 우주시간에 유지하는 조건.
8. early-time radiation-like 전이, caustic 또는 strong-coupling이
   암흑물질 주장을 깨는 domain.

## 5. route gate

다음 우선순위로 채택한다.

1. B와 $\Pi_{\rm fold}$ 경계 matching이 한 벌크장, 공변 보존, anchor,
   양의 dust, 후기 가속을 동시에 만족하면 B.
2. B에 완전 반례가 있고 A가 안정 domain에서 살아남으면 A를 채택하되
   절대 기록이 아니라 변화율만 물리량이라고 명시한다.
3. 둘 다 실패하면 constrained clock 또는 record+reservoir를 새 action으로
   분리하고 이 run에서 목표 달성을 보고하지 않는다.

## 6. 수치 계약

- 표준 라이브러리만 쓰는 독립 모듈과 focused test를 만든다.
- analytic 식과 독립 수치 적분을 비교한다.
- 양의 에너지, Friedmann closure, current/continuity, grid refinement를
  검사한다.
- unified-dark-sector boundary는 baryon과 radiation을 별도로 두고,
  오늘의 dark-matter/dark-energy 비율을 외부 경계값으로 선언한다.
- background 근접은 full CMB/LSS/halo 검증으로 부르지 않는다.
- on-shell identity를 독립 residual이라고 부르지 않는다.

## 7. 완전 반례와 주장 상한

다음 중 하나면 해당 부모 주장을 삭제한다.

1. $T$ 또는 $\Theta$의 목표 observable이 자유 amplitude에 정확히 흡수된다.
2. 물리 branch에서 $P_X+2XP_{XX}\le0$ 또는 $c_s^2<0$다.
3. 필요한 dark-matter density가 유한 시점에 음수가 된다.
4. dust 근사가 명시한 우주론 domain 전에 깨지는데도 full CDM으로 보고한다.
5. 한 방향성을 retarded kernel이나 boundary arrow 없이 작용에서 증명했다고
   주장한다.
6. dimensionless 0D 정보만으로 $\rho_\infty$, $M^4$, $X_*$ 또는 $\Gamma$의
   차원 있는 값을 산출했다고 주장한다.

통과 가능한 최강 결론은 “자기-비동일성의 미래향 변화율을 Cauchy 자료로
채택하면, 한 유효 clock 장이 특정 안정 domain에서 DE-like 잔여와
DM-like 흐름 편차를 함께 만들 수 있다”까지다. 그 장이 실제 비선택 양자경로의
coarse-graining이라는 microscopic map과 차원 있는 scale의 수치 예측은 별도
미완성 다리다.

# 측정 벽의 차원과 유한시간 구조 연구 계약

Status: COMPLETE

PREDECESSOR:

- `_workspace/ce/zero-dimensional-fold-memory-field-20260825`
- `_workspace/ce/quantum-path-opportunity-functional-20260825`

## 1. 연구 질문

“측정 그 자체를 0차원으로 정의할 수 있는가”와 “측정이 항상 벽인가”를 서로
다른 수학적 공간에서 검사한다. 측정이 반드시 한순간이라는 전제는 두지 않는다.
다음 네 차원을 혼합하지 않는 것이 계약의 첫 조건이다.

1. spacetime interaction support의 차원,
2. measurement protocol의 시간 support,
3. outcome/record 공간의 위상적 차원,
4. history 또는 process를 가르는 operational cut의 codimension.

## 2. finite-duration instrument

system $S$, apparatus $A$, 초기 apparatus 상태 $\sigma_A$와 interval
$[t_0,t_1]$에서

$$
U_M(t_1,t_0)=\mathcal T
\exp\left[-\frac{i}{\hbar}\int_{t_0}^{t_1}
H_{SA}(t)dt\right]
$$

를 둔다. pointer outcome $r$의 **완전한 직교 포인터 사영**을 $\Pi_r$라 하고
$\Pi_r\Pi_s=\delta_{rs}\Pi_r$, $\sum_r\Pi_r=I_A$를 요구하면

$$
\mathcal I_r^{[t_0,t_1]}(\rho)
=\operatorname{tr}_A
\left[(I\otimes\Pi_r)U_M(\rho\otimes\sigma_A)U_M^\dagger
(I\otimes\Pi_r)\right]
$$

를 후보 instrument로 고정한다. $\Delta t=t_1-t_0>0$를 허용하며, endpoint
record와 interaction duration을 구분한다.

일반 POVM에는 위 sandwich 식을 그대로 쓰지 않고
$\mathcal I_r(\rho)=\sum_\alpha M_{r\alpha}\rho M_{r\alpha}^\dagger$,
$\sum_{r,\alpha}M_{r\alpha}^\dagger M_{r\alpha}=I$인 Kraus instrument를 쓴다.
이 조건은 수학 레인이 원문의 `effect/projector` 모호성을 발견한 뒤 명시한
범위 수정이며, 원래 후보가 일반 POVM에 성립하지 않는다는 반례는
`11-math.md`에 보존한다.

## 3. 사전 고정 경로

### R1. record-0D

유한 discrete outcome space $\mathcal O=\{r_1,\ldots,r_n\}$는 위상적으로
0차원이고 선택 record $\{r\}$는 한 점이다. “측정의 0차원성”을 우선 이
**record dimension**으로 정의할 수 있는지 검사한다. spacetime point라는 뜻으로
확장하지 않는다.

### R2. spacetime support

measurement coupling region $\mathcal R_M\subset M$의 차원을 독립적으로 센다.
point event, point detector worldline, detector surface worldtube와 finite-volume
apparatus region을 분리한다. 모든 측정의 support가 0D라는 보편 주장은 반례
감사한다.

### R3. measurement wall

직교 pointer projector $\{P_r\}$의 dephasing map을

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r,
\qquad \mathcal D_P^2=\mathcal D_P
$$

로 둔다. wall strength $0\le\eta\le1$의 family는

$$
\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P
$$

로 사전 고정한다. $\eta=0$은 no wall, $0<\eta<1$은 partial wall,
$\eta=1$은 coherence block을 완전히 지우는 hard record wall이다.

### R4. 유한시간 wall 형성

rate $\gamma(t)\ge0$의 generator

$$
\dot\rho_t=\gamma(t)(\mathcal D_P-I)\rho_t
$$

를 풀어

$$
\eta(t)=1-\exp\left[-\int_{t_0}^{t}\gamma(s)ds\right]
$$

가 나오는지 검산한다. 이는 continuous monitoring 전체의 보편식이 아니라
순수 dephasing witness다.

### R5. Zeno wall

동일 projector를 총시간 $T$ 동안 $N$번 적용한

$$
\left(Pe^{-iHT/(N\hbar)}P\right)^N
$$

의 $N\to\infty$ 한계가 조건부 dynamical boundary를 만드는지 문헌과 함께
검토한다. finite-strength 일반 measurement와 동일시하지 않는다.

### R6. 기회비용 연결

finite-duration measurement에서 time-dependent outcome distribution $p_r(t)$가
instrument에 의해 정의될 때만

$$
C_I(t)=-\sum_{r\ne o}p_r(t)\ln p_r(t)
$$

를 둘 수 있다. endpoint cost, time integral과 rate cost를 구분하고, energy로
바꾸려면 PREDECESSOR의 thermal/action bridge가 다시 필요하다고 고정한다.

## 4. 고정 계산

qubit pointer basis $P_0=|0\rangle\langle0|$,
$P_1=|1\rangle\langle1|$와

$$
\rho_0=\begin{pmatrix}1/2&1/2\\1/2&1/2\end{pmatrix}
$$

를 쓴다. $\Phi_\eta$가 diagonal을 보존하고 off-diagonal을 $1-\eta$배 하는지,
$\gamma(t)=\gamma_0$에서 $\eta(t)=1-e^{-\gamma_0(t-t_0)}$인지 검산한다.
analytic tolerance는 $10^{-12}$다.

## 5. 반증 조건

1. 모든 measurement interaction은 spacetime point다.
2. 모든 POVM/instrument는 hard wall이다.
3. outcome space의 0차원성이 physical extra dimension을 뜻한다.
4. finite-duration coupling을 instantaneous collapse와 동일시해도 예측이 항상 같다.
5. nonselective dephasing map만으로 실제 selected record history가 정해진다.
6. wall strength 또는 기회비용이 scale 없이 energy/stress가 된다.

## 6. 주장 상한

이 run은 “discrete completed record는 0D outcome atom으로 볼 수 있고,
measurement interaction은 일반적으로 유한 spacetime region이며, projective
dephasing wall은 유한시간 동안 연속적으로 형성될 수 있다”까지 주장할 수 있다.
새 물리적 차원, objective collapse, 양자중력, dark identity와 abundance는
주장하지 않는다.

# Alternative-route lane

Status: LANE_COMPLETE — R2-D route selected; full closure open

## 1. Decision matrix

| route | DE | DM | anchor | conservation | principal cost | verdict |
|---|---|---|---|---|---|---|
| canonical exponential $V=\rho_*e^{-\Theta}$ | yes, quintessence | no | exact shift-amplitude degeneracy | yes | operational origin erased | reject as target model |
| pure quadratic $P_A(X)$ | nonzero-flow $w=-1$ | $a^{-3}$ only near kinetic extremum | absolute field intentionally unphysical | yes | early radiation-like regime and $c_s^2=0$ endpoint | retain as analytic control |
| anchored soft clock $P_B(T,X)$ | increasing $V\to\rho_\infty$ | positive small-$\delta$ kinetic inventory | physical $\Sigma_*$ required | exact total conservation | needs fold-matched $\Pi_{\rm fold}>0$, global positivity, cutoff | adopt |
| exact multiplier clock | exact $V+$dust split | exact irrotational dust | physical $\Sigma_*$ | exact | caustics; auxiliary multiplier; positivity | limiting control |
| record+reservoir two-field | flexible DE and oscillating DM | possible | explicit | exact total conservation | not one field; reversible locally | fallback |
| retarded S-K environment | genuine one-way response | model dependent | causal kernel | enlarged-system conservation | doubled fields/noise/kernel | required only for intrinsic irreversibility |

## 2. Why the action changed

The failed exponential route attached physical meaning to an absolute scalar value
that its own parameter redundancy erased. The corrected route attaches the primary
meaning to the timelike invariant $X$, while the separate $T=0$ anchor appears only
inside a non-rescalable accumulated readout.

The key replacement is

$$
\rho_*e^{-\Theta}
\quad\longrightarrow\quad
P_B(T,X)=\rho_\infty\left[
\frac{\kappa}{2}\left(\frac{X}{X_*}-1\right)^2
-\left(1-e^{-\Gamma T}\right)
\right].
$$

The first term is a soft bootstrap constraint: neighboring event increments penalize
departures from a preferred nonzero flow. The second is a saturating opportunity
record. This interpretation is an explicit construction axiom, not a theorem of
ordinary quantum mechanics.

The 0D object is not the scalar field itself. The type-safe chain is

$$
Z_{\rm phys}\longrightarrow R\longrightarrow\mu_F
\xrightarrow{\mathcal R_T}(T,X,J_i),
$$

where $Z_{\rm phys}$ is the external preparation boundary, $R$ an informative
environment record, $\mu_F$ the retained spatial-0D carrier measure, and $T$ the
4D coarse-grained clock field. Collapsing these four types back into one object
would restore the earlier no-signalling contradiction.

The final arrow in that chain is the Cauchy map

$$
\left(T,a^3P_X\dot T\right)|_{\Sigma_*}
=\left(0,\Pi_{\rm fold}\right),
\qquad \Pi_{\rm fold}>0.
$$

This is the actual equation-level repair. With zero second datum, the accumulated
potential drives the model to the unstable $\delta<0$ side. With positive retained
inventory satisfying the exact integral bound, the same conserved one-field stress
supports a positive DM-like component while transferring part of it into $V$.

## 3. Directionality audit

$z_{n+1}\ne z_n$ proves change, not direction, because the reversed sequence also has
neighboring nonidentity. In the adopted local EFT, direction is supplied by

$$
T|_{\Sigma_*}=0,\qquad \dot T|_{\Sigma_*}>0,\qquad J_i>0.
$$

Here $a_i^3J_i=\Pi_{\rm fold}$ is fixed by the retained-fold map, rather than
silently fitted as a zero-rest condition. This is a future-branch matching condition.
A law that forbids the reverse process
would require a retarded environment kernel, for example

$$
K_R(x,y)=0\qquad\text{when }x\notin J^+(y),
$$

inside a Schwinger--Keldysh influence action. That extension is outside the one-field
closed action and is not silently assumed.

## 4. Opportunity-cost audit

The phrase “energy 없는 energy” cannot mean zero stress tensor and nonzero gravity at
the same time. The precise replacement is:

> the constraint or missed alternative is not itself assigned a classical branch
> energy; its retained coarse-grained record is represented by a positive current and
> an effective stress tensor.

In route B, increasing $V$ is paid for by decreasing kinetic inventory. This eliminates
double counting and makes the source ledger explicit:

$$
Q=\dot V,\qquad
\dot\rho_K+3H(\rho_K+p_K)=-Q,\qquad
\dot\rho_V=Q.
$$

## 5. Adopted route and remaining tests

Adopt B with A and the exact constrained clock as controls. Required implementation
tests are:

1. background shooting to a declared present DE/DM split;
2. $J>0$, $\delta\ge0$, $\rho>0$ on the whole grid;
3. direct finite-difference continuity and current checks;
4. small-$\delta$ and $\kappa\to\infty$ convergence;
5. future positivity plus $\Gamma\sqrt{2X_*}>3H_\infty$;
6. explicit warning that background success is not CMB/LSS/halo closure.

## 6. R1 — 폐쇄시간경로 초기상태와 영향함수

Route-ID: R1-open-influence
Objective-ID: SNKC-QPATH-ORIGIN-01
Structural-Class: micro-macro
Changed-Structure: 정확한 고전 경계값 두 개를 동시에 가정하던 R0를 양의 폐쇄시간경로 초기 밀도행렬, retarded 환경 kernel과 reservoir stress를 포함하는 열린계 matching으로 교체한다
Preserved-Objective: 비선택 경로의 보존 기록이 양의 초기 전류를 준비하고 동일 벌크장의 암흑물질형·암흑에너지형 성분으로 읽히는 보존적 미시 완성을 찾는다
New-Degrees-of-Freedom: 물리적 새 암흑 벌크장이 아니라 폐쇄시간경로의 $T_\pm$, 초기상태 공분산, 환경 kernel과 reservoir 상태를 추가한다
Parameter-Accounting: $\Pi_F=\Lambda_\Pi^4f(\mu_F,C_{\rm self})$와 $\Lambda_\Pi$, 초기 폭, kernel 계수는 외부 matching 입력이며 현재 산출값이 아니다
Conservation-Law: 축약된 $T$ sector의 비보존량과 reservoir의 반대 유량을 합쳐 $\nabla_\mu(T_T^{\mu\nu}+T_R^{\mu\nu})=0$을 요구한다
Dimension-Check: $[T]=-1$, $[J]=[\Pi_F]=4$, $[K_\Sigma]=5$, bilocal $[N_\Sigma]=8$이며 모든 경계 작용항은 무차원이다
Stability-Condition: 초기 밀도행렬은 양의 정규화 상태이고 $N_\Sigma$는 양의 준정부호이며 벌크에서는 $\delta\ge0$과 기존 ghost·gradient 조건을 유지한다
Prior-Negative-Control: artifacts/negative-controls/r0-classical-boundary-anchor-no-go.md
Falsifier: 양의 정규화 경계 밀도행렬과 명시한 reservoir stress가 $\langle T_i\rangle=0$, $\langle J_i\rangle>0$ 및 총 Ward 보존을 동시에 실현하지 못하거나 homogeneous 준비가 점원으로 환원되어 FLRW를 깨면 이 경로를 기각한다

R0의 고전 경계항은 정확한 $T_i=0$과 양의 $J_i$를 동시에 유도하지 못한다.
따라서 R1은 두 값을 고전적인 동시 고정값으로 두지 않고, 초기 축약 밀도행렬의
평균과 공분산으로 준비한다. 폐쇄시간경로 변수를

$$
T_r=\frac{T_++T_-}{2},
\qquad
T_a=T_+-T_-
$$

로 정의한다. 초기 영향작용의 최소 후보는

$$
S_{{\rm IF},\Sigma}
=\int_{\Sigma_*}d^3x\sqrt h\,
\left[-\Pi_FT_a-K_\Sigma T_aT_r\right]
+\frac{i}{2}\int_{\Sigma_*\times\Sigma_*}
d^3x\,d^3y\sqrt{h_xh_y}\,
T_a(x)N_\Sigma(x,y)T_a(y)
$$

이다. 물리 극한 $T_a=0$에서 $T_a$ 변분은 평균 경계조건

$$
\langle J_i\rangle
=\Pi_F+K_\Sigma\langle T_i\rangle
$$

을 준다. 양의 경계상태의 평균을 $\langle T_i\rangle=0$으로 준비하면
$\langle J_i\rangle=\Pi_F>0$을 얻는다. 이것은 $T_i$와 $J_i$의 폭까지 0으로
고정한다는 뜻이 아니다. 두 변수의 공분산은 양자 불확정성과 상태 양성 조건을
만족해야 한다.

실제 일방향 응답을 주장하려면 초기항만으로는 부족하다. 명시적 저장소와
유계 결합 $s_A(T)=\mu_A^3F_A(\Gamma T)$를 적분한 벌크 영향작용에

$$
S_{{\rm IF},{\rm bulk}}
=-\int d^4x\,d^4y\,s_a(x)D_R(x,y)s_r(y)
+\frac{i}{2}\int d^4x\,d^4y\,s_a(x)N(x,y)s_a(y)
$$

를 두고 $D_R(x,y)=0$ for $x\notin J^+(y)$, $N\succeq0$을 요구한다. 축약된
$T$ 장만 보면 환경과 에너지를 교환하므로 일반적으로
$\nabla_\mu T_T^{\mu\nu}=Q^\nu$다. 따라서 반대 유량을 갖는 reservoir stress를
명시해 총합을 보존해야 한다. 이 경로가 현재 여는 것은 양의 평균 전류를 준비하는
일관된 후보 구조다. $\Pi_F$의 수치, 균일한 $\Sigma_*$의 기원, intrinsic 시간
화살과 실제 비선택 양자경로의 동일성은 아직 산출되지 않았다.

## 7. R1.1 — 유계 연속 Gaussian 저장소

단순 선형 결합 $g_AT\phi_A$는 저장소 potential을 완성제곱할 때
$-g_A^2T^2/(2m_A^2)$를 남겨 하한을 잃으므로 제거한다. 현재 최소 생존
작용은

$$
S_{\rm tot}=\int\sqrt{-g}\left\{P(T,X)-\sum_A\left[
\frac12(\nabla\phi_A)^2+\frac12m_A^2\phi_A^2
+\mu_A^3F_A(\Gamma T)\phi_A\right]\right\}
$$

이며 $F_A$는 유계이고 $F_A(0)=0$이다. 연속 스펙트럼, UV 수렴조건,
양의 전체 Gaussian 공분산, cell smearing $L_c$, interaction stress의 계량
변분을 함께 요구한다. 이 조건 아래 인과적 $D_R$, 양의 $N$, 총 Ward 보존은
작용에서 유도된다.

이 구조도 $\Pi_F$를 계산하지 않는다. $\Pi_F$는 여전히 초기 Gaussian
canonical momentum의 평균 변위다. 또한 양자 평균을 고전 current로 보내려면
$\langle J\rangle=J_{\rm cl}+O(\epsilon_J)$인 좁은 packet 조건이 필요하다.
고정 배경 장파장 $m_{\rm eff}^2<0$과 $c_s^2\to0$은 full perturbation 및
$k^4$ 완성 전까지 활성 falsifier로 둔다.

현재 route 판정은 **최소 열린계 작용의 조건부 일관성 PASS / 0차원 기원과
우주론 폐쇄 REVISE**다.

## 8. R2-A — 감쇠 타키온의 누적 성장 제한

Route-ID: R2-A-fixed-metric-growth
Parent-Route: R1-open-influence
Preserved-Objective: 포화형 기회 readout과 동일 clock의 DM-like kinetic 재고를 유지하면서 실제 섭동 성장 허용성을 판정한다
Changed-Test: $m_{\rm eff}^2<0$의 부호만으로 기각하지 않고 $|m_{\rm eff}^2|/H^2$, 누적 성장지수와 직접 $k=0$ 해를 전 배경에서 계산한다
New-Parameters: 없음; 동결 R0의 $\kappa$, $\gamma$, background를 그대로 소비한다
Prior-Negative-Control: 비상수·단조 증가·포화 $V$는 전역 $V''\ge0$일 수 없다
Falsifier: 누적 log 성장 또는 metric-mixed pole이 관측 기간에 order one 이상이거나 cutoff 아래에서 ghost/gradient/Jeans 조건을 위반하면 R2-A를 기각한다

R2-A가 먼저인 이유는 bath 함수나 새 EFT 계수로 질량을 튜닝하기 전에 기존
배경이 실제로 빠르게 자라는지 확인하기 위해서다. artifact
'r2_fixed_metric_growth_gate.py'는 동결 배경을 독립 재구성했고 fixed-metric
누적 상계가 $2.49\times10^{-17}$보다 작음을 확인했다.

고정계량 $\pi$는 gauge-dependent이므로 이 결과를 물리적 타키온 PASS로
부르지 않는다. 제약을 제거한 clock+GR 단일-clock 부분계에서는 $Q_s>0$,
$c_s^2>0$이고 두 번째 $\zeta$ mode의 $\dot\zeta$와 적분함수가
$a=10^{-4}$부터 오늘까지 감소한다.
같은 구간의 tree-level power counting도 $\Lambda_E/H>9.27\times10^{18}$,
$q_{\rm sc}/[(1\,{\rm Mpc}^{-1})/a]>2.17\times10^{24}$를 준다. 다만
baryon·radiation·reservoir 섭동을 끈 부분계이므로 full ADM PASS가 아니다.

다음 후보 R2-B는 bath의 $V_{\rm eff}''$를 조절하는 경로지만 새 함수와
스펙트럼을 요구하며 전역 포화 no-go를 없애지 못한다. R2-C는 constrained
dust/clock을 쓰지만 새 보조 자유도와 caustic 문제를 낳는다.

## 9. R2-D — $k^4$-before-strong-coupling 경로

Route-ID: R2-D-k4-before-strong-coupling
Parent-Route: R2-A-fixed-metric-growth
Preserved-Objective: 포화 readout과 동일 clock의 DM-like kinetic 재고 및 동결 배경을 유지한다
Changed-Test: $c_s\to0$을 즉시 실패로 보지 않고 $q_\times\le q_{\rm sc}$인 higher-spatial-derivative crossover와 full reduced ADM matrix를 검사한다
New-Parameters: unitary-gauge $k^4$ 계수 $\bar M$ 및 공변 completion에 필요한 제한된 계수들
Prior-Negative-Control: $\Lambda_E$를 물리 파수와 직접 비교해 $\bar M$을 과대평가하는 계산은 단위/분산관계 오류로 폐기한다
Falsifier: 허용 계수에서 $q_\times>q_{\rm sc}$이거나 coupled ADM에 ghost·gradient·Jeans pole이 cutoff 아래 나타나면 R2-D를 기각한다

올바른 조건은

$$
q_\times=\frac{c_s\sqrt A}{\bar M}\le
q_{\rm sc}=\Lambda_3c_s^{3/4}
$$

이며, 현재 배경의 필요조건은 오늘 $\bar M\gtrsim0.225\,{\rm eV}$, 전체
관측구간 최악에서 $\bar M\gtrsim7.31\,{\rm eV}$다.
$\bar M\sim\Lambda_3\sim80\,{\rm eV}$
후보는 crossover 조건만으로는 배제되지 않는다. 따라서 R2-D를 다음 활성
경로로 선택한다. 이 선택은 $\bar M$의 기원·부호·DHOST/ADM degeneracy 또는
$k^4$ 영역의 cutoff를 이미 증명했다는 뜻이 아니다.

## 10. R3-A — 배경 보존 extrinsic-curvature completion

Route-ID: R3-A-deltaK2-single-clock
Objective-ID: SNKC-K4-ADM-COMPLETION
Structural-Class: action-state
Changed-Structure: 배경에서 0인 $\delta K=K-3H(T)$의 제곱을 unitary-gauge 작용에 추가하고 lapse·shift를 포함한 제약계를 다시 푼다
Preserved-Objective: 포화 DE-like readout, 같은 clock의 DM-like kinetic 재고, R0의 동결 FLRW 배경과 R1 총 Ward 장부
New-Degrees-of-Freedom: unitary-gauge EFT 안에서는 없음이 목표이며, $\bar M$ 계수 하나를 추가한다
Parameter-Accounting: $\bar M(T)$는 새 외부 EFT 함수; 첫 gate에서는 상수 $\bar M\in[8,80]\,{\rm eV}$만 검사하고 예측으로 세지 않는다
Conservation-Law: 전체 unitary-gauge action의 ADM 제약과 Bianchi identity를 사용하며, $\delta K=0$ 배경에서 기존 총 stress 장부를 이중계상하지 않는다
Dimension-Check: $[\delta K]=1$, $[\bar M^2]=2$, $[N\sqrt h\,d^4x]=-4$이므로 $\Delta S_{K^2}$는 무차원이다
Stability-Condition: reduced kinetic $Q>0$, 고주파 $q^4$ 계수 $C_4>0$, 모든 cutoff 아래 pole의 성장률 $<H$, $q_\times\le q_{\rm sc}$
Prior-Negative-Control: artifacts/negative-controls/r2d-energy-momentum-cutoff-confusion.md
Falsifier: 최소 $(\delta K)^2$ 항이 추가 scalar ghost, 음의 $q^4$, 빠른 Jeans pole, tensor 불안정 또는 strong-coupling 뒤 crossover를 만들면 이 경로를 기각한다

첫 후보는

$$
\Delta S_{K^2}=-\frac12\int d^4x\,N\sqrt h\,
\bar M^2(\delta K)^2
$$

다. trace $\delta K$는 tensor의 선형 trace에 결합하지 않으므로 tensor
quadratic action을 건드리지 않을 가능성이 있지만, 이 문장은 제약식과
2차 전개를 끝내기 전에는 가설이다. scalar sector에서는 decoupling 한계의
$+(\bar M^2/A)q^4$와 중력 혼합의 Jeans형 항을 함께 계산한다. 이 경로가
실패하면 다음 구조 후보는 독립 $\delta K^i{}_j\delta K^j{}_i$ 조합 또는
degenerate scalar--tensor completion이며, 계수만 다시 맞추는 수정은 새
route로 세지 않는다.

# Formal gate audit — self-nonidentity kinetic dark sector

Status: REVISE

Gate: REVISE

Scope: R0/R1 조건부 결과와 R2 섭동·cutoff 진단; microscopic origin과 full closure는 open

## 1. Stable snapshot

Audited files:

- 00-contract.md
- 10-sources.md
- 11-math.md
- 12-routes.md
- 20-audit.md
- 30-implementation.md
- 31-validation.md
- 40-final-report.md
- artifacts/r1_gaussian_reservoir_gate.py
- artifacts/r2_fixed_metric_growth_gate.py

The first status audit returned REVISE because $J_i=0$ forces the anchored
increasing-potential model onto $\delta<0$, where $c_s^2<0$. Revision 1 replaced
the zero-rest prescription with the retained-fold Cauchy map

$$
\left(T,a^3P_X\dot T\right)_{\Sigma_*}
=\left(0,\Pi_{\rm fold}\right),
\qquad
\Pi_{\rm fold}
=\mathcal R_\Pi[\mu_F,C_{\rm self};
\rho_\infty,\Gamma,L_c]>0.
$$

The second independent audit found no remaining P0 inside the declared domain.

## 2. Accepted theorem chain

For

$$
P(T,X)=\rho_\infty\left[
\frac\kappa2\left(\frac{X}{X_*}-1\right)^2
-\left(1-e^{-\Gamma T}\right)
\right],
$$

with

$$
\rho_\infty,\kappa,X_*,\Gamma>0,\qquad
T\ge0,\qquad\dot T>0,\qquad\delta\ge0,
$$

the stress tensor is covariantly conserved on shell and

$$
\rho=\rho_\infty\left[
1-e^{-\Gamma T}+2\kappa\delta+\frac32\kappa\delta^2
\right]>0.
$$

The potential part has $\rho_V=V$, $p_V=-V$. The kinetic part has

$$
w_K=\frac{\delta}{4+3\delta},\qquad
c_s^2=\frac{\delta}{2+3\delta},
$$

so $0<\delta\ll1$ is positive, ghost-free, gradient-stable and DM-like.
The exact current ledger is

$$
a^3J(t)=\Pi_{\rm fold}
-\int_{t_i}^{t}a^3\rho_\infty\Gamma e^{-\Gamma T}\,dt.
$$

Therefore the corrected action plus boundary data can simultaneously realize an
increasing DE-like readout and a remaining positive DM-like inventory without
violating total conservation.

For the implemented $X_*=1/2$ branch, a finite positive endpoint can be closed
to the infinite future without assuming the conclusion.  Suppose a first future
zero of $q=e^{3N}u\sqrt{1+u/\kappa}$ exists.  Before that zero $u>0$; total
continuity and Friedmann positivity then give $E\le1$, $E\ge\sqrt{V_f}$, and
$\tau'\ge1$.  For the conservative sufficient condition $\gamma>3$, the
remaining depletion is bounded by

$$
\Delta q_{\rm tail}\le
\frac{\gamma e^{3N_f-\gamma\tau_f}}
{2\sqrt{V_f}(\gamma-3)}.
$$

The implemented inequality $q_f>\Delta q_{\rm tail}$ contradicts the first
zero and closes $q>0$ for all $N\ge N_f$.  The condition $\gamma>3$ is not
claimed to be physically necessary.

## 3. Deleted parent claims

| deleted claim | complete counterexample |
|---|---|
| the exponential SMQ background identifies the absolute measurement origin | exact $(\Theta,\rho_*)\mapsto(\Theta+\Delta,\rho_*e^\Delta)$ degeneracy |
| increasing $V$ and positive dust emerge from zero current | $J_i=0$ gives $J<0$, $\delta<0$, $c_s^2<0$ immediately |
| self-nonidentity alone proves a future arrow | the reversed sequence is also neighbor-nonidentical |
| the quadratic kinetic sector is CDM at arbitrary redshift | $\delta\gg1$ gives radiation-like scaling |
| on-shell zero or dimensionless opportunity cost can gravitate without stress and a scale | Einstein coupling requires a dimension-four stress tensor |

## 4. P1 limits

- $\mathcal R_\Pi$ is a matching axiom, not a standard-quantum derivation.
- $\rho_\infty$, $\Gamma$, $X_*$, $\kappa$, and $L_c$ are not predicted.
- $\Pi_{\rm fold}$ is initial canonical current, not a second energy account.
- intrinsic irreversibility needs a retarded open-system completion.
- $c_s^2\to0$ requires an explicit EFT cutoff or higher-spatial-derivative
  completion before a UV-stability claim.
- the tied constant/exponential coefficients that remove the old shift degeneracy
  have no demonstrated radiative protection; an independent vacuum counterterm
  would weaken anchor identifiability.
- full CMB, LSS, nonlinear caustics, halos, and lensing are not closed by a
  homogeneous background test.
- at high $\gamma$, the solved shooting coordinate $b=\gamma\tau_0$ must be
  retained.  Reconstructing $b$ from the nearly equal pair
  $A\simeq\Omega_{V0}$ is floating-point ill-conditioned and is not a physical
  instability.
- the analytic zero-current derivative no-go is not independently exercised by
  the current helper test, which checks the incompatible present $u=0$ anchor.

## 5. Implementation gate

Implementation may proceed only if it checks the full-grid signs
$J>0,\delta\ge0,\rho>0$, the finite-interval current supremum, independent
finite-difference continuity/current residuals, and grid refinement. Any reported
DM claim must include the maximum $\delta$, $w_K$, and $c_s^2$ over its stated
redshift interval.

## 6. R1 재감사 — 2026-08-26

앞의 `PASS`는 R0의 조건부 균일 배경 계산에만 해당했다. 현재 경로가
`R1-open-influence`로 바뀌었으므로 run 전체 상태를 `REVISE`로 되돌린다.
R1은 유계 결합의 연속 Gaussian 저장소를 명시함으로써 인과적 지연핵,
양의 잡음핵, clock--reservoir 총응력 보존을 조건부로 구성했다. 그러나
$\Pi_{\rm fold}$는 여전히 초기 Gaussian 상태의 평균 운동량 변위이고,
0차원 접힘에서 그 값이 산출되지 않는다.

새 음성대조군은 비유계 선형 결합 $gT\phi$다. 저장소 좌표를 완성제곱하면
$-g^2T^2/(2m^2)$가 남아 $T\to\infty$에서 물질 Hamiltonian의 하한이
사라진다. 이 경로는 제거하고, $s_A(T)=\mu_A^3F_A(\Gamma T)$에서 $F_A$가
유계인 경로만 유지한다.

또한 기존 $P(T,X)$에는 고정 배경 장파장 근사에서

$$
m_{\rm eff}^2=-\frac{P_{TT}}{P_X+2XP_{XX}}<0
$$

인 열린 안정성 검사가 남는다. metric mixing, 저장소 결합 및 $k^4$ 완성을
포함한 전 섭동계가 이 성장을 억제하는지 계산하기 전에는 R1 전체를
`PASS`로 올릴 수 없다. 현 판정은 **조건부 작용 일관성 통과, 우주론 폐쇄
미완성**이다.

## 7. R2-A 고정배경 성장 감사

포화·단조 증가·전역 무타키온의 동시 요구는 식 (R2.6)의 완전 반례로
제거한다. 다만 이 반례는 음의 질량이 관측 시간에 크게 성장한다는 뜻이
아니다. 동결 배경을 독립 재적분한 R2-A는

$$
\max |m_{\rm eff}^2|/H^2=3.0455\times10^{-18},
\qquad
\log G_{\rm fixed}<2.4839\times10^{-17}
$$

을 얻었고 2:1 격자 최대 상대 변화는 $3.93\times10^{-7}$이었다. 따라서
선택 초기조건 $\pi_i=1$, $\pi_i'=0$과 명시한 계수 비교 아래 fixed-metric
장파장 질량 gate는 통과한다. 이 판정은 큰 외부 입력 $\kappa=10^{17}$에
조건부이며 임의 초기조건의 정리가 아니다.

고정계량 $\pi$는 gauge-dependent이므로 이 PASS를 물리적 scalar pole의
안정성으로 읽는 것도 금지한다. Einstein+clock 단일-clock 부분계에서는

$$
Q_s=\frac{X(P_X+2XP_{XX})}{H^2}>0,
\qquad c_s^2=\frac{P_X}{P_X+2XP_{XX}}>0
$$

이며 유한 관측구간에서 $a^3Q_s$와 $Ha^3Q_s$가 증가한다. 수치 하한은
$c_s^2>9.21\times10^{-19}$, $Q_s/M_{\rm Pl}^2>3.31\times10^5$다.
이는 단일-clock subblock 필요조건 PASS다.

작은-$c_s$ cutoff 감사에서는 에너지
$\Lambda_E=\Lambda_3c_s^{7/4}$와 물리 파수
$q_{\rm sc}=\Lambda_3c_s^{3/4}$를 분리했다. 최소
$\Lambda_E/H=9.27\times10^{18}$,
$q_{\rm sc}/[(1\,{\rm Mpc}^{-1})/a]=2.17\times10^{24}$이므로 관측구간의
선형 scale separation은 조건부 통과다. $q_\times\le q_{\rm sc}$에 필요한
전체 관측구간 최악에서 $q_\times\le q_{\rm sc}$를 보장하려면
$\bar M\gtrsim7.31\,{\rm eV}$가 필요하므로,
$\bar M\sim\Lambda_3\sim80\,{\rm eV}$ 후보는 이 crossover 조건을 만족할
수 있다.

Gate는 계속 REVISE다. baryon·radiation·reservoir perturbation을 포함한
coupled ADM matrix, bath pole, $k^4$ 연산자의 실제 부호·degeneracy·새 cutoff,
gauge-invariant CMB/LSS mode가 계산되지 않았기 때문이다.

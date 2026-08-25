# Implementation — anchored self-nonidentity kinetic dark sector

Status: R2 IN_PROGRESS (R0 background and R1 reservoir implementation complete)

## 1. Implemented scope

The implementation evaluates the conditional EFT accepted by the formal gate,

$$
P(T,X)=\rho_\infty\left[
\frac\kappa2\left(\frac{X}{X_*}-1\right)^2
-\left(1-e^{-\Gamma T}\right)
\right],
$$

on a homogeneous flat FLRW background.  It does not implement the microscopic
map $\mathcal R_\Pi$, a retarded open-system completion, perturbations, CMB, LSS,
halos, or nonlinear caustics.

The implementation owner changed only these source and test paths:

- `.tmp/ce-cosmo-dso-20260825/src/ce_cosmo/gates/self_nonidentity_kinetic_dark_sector.py`
- `.tmp/ce-cosmo-dso-20260825/tests/test_self_nonidentity_kinetic_dark_sector.py`

The reproducible research artifact is:

- `artifacts/run_kinetic_clock_numerics.py`
- `artifacts/numerical-results.json`

## 2. Dimensionless system

The numerical normalization fixes $X_*=1/2$ and uses

$$
N=\ln a,\qquad \tau=H_0T,\qquad
\gamma=\Gamma/H_0,\qquad u=\kappa\delta,
\qquad A=\rho_\infty/\rho_{\mathrm{crit},0}.
$$

The dark kinetic inventory is the whole DM-like sector; it is not added on top
of a separate standard-CDM term.  The solved Friedmann equation is therefore

$$
E^2=\Omega_{b0}e^{-3N}+\Omega_{r0}e^{-4N}
+A\left[1-e^{-\gamma\tau}+2u+\frac{3u^2}{2\kappa}\right].
$$

The exact first-order system is

$$
\tau'=\frac{\sqrt{1+u/\kappa}}{E},
$$

$$
F(u)=u\sqrt{1+u/\kappa},\qquad
F_u=\frac{1+3u/(2\kappa)}{\sqrt{1+u/\kappa}},
$$

$$
u'=\frac{-3F(u)-\gamma e^{-\gamma\tau}/(2E)}{F_u}.
$$

The present boundary fixes
$\Omega_{b0}=0.049$, $\Omega_{r0}=9\times10^{-5}$,
$\Omega_{V0}=0.687$, and
$\Omega_{K0}=1-\Omega_{b0}-\Omega_{r0}-\Omega_{V0}$.  Shooting is performed in

$$
b=\gamma\tau_0,\qquad
A=\frac{\Omega_{V0}}{1-e^{-b}},
$$

and preserves $b$ directly instead of reconstructing it from $A$.  This is
necessary at large $\gamma$, where $A-\Omega_{V0}$ loses enough floating-point
precision to corrupt the backward endpoint even though $b$ remains resolved.

## 3. Stability and conservation gates

Every accepted trajectory must satisfy the full-grid conditions

$$
\tau\ge0,\qquad u>0,\qquad E>0,\qquad
q=e^{3N}F(u)>0.
$$

The implemented physical diagnostics are

$$
\rho_K=A\left(2u+\frac{3u^2}{2\kappa}\right),
\qquad
p_K=A\frac{u^2}{2\kappa},
$$

$$
w_K=\frac{u/\kappa}{4+3u/\kappa},
\qquad
c_s^2=\frac{u}{2\kappa+3u}.
$$

Independent finite-difference derivatives check the current equation and total
continuity instead of reporting the ODE right-hand sides as residuals.

The finite future solve ends at $N_f=10$.  It is extended to the infinite future
by the first-zero contradiction bound derived in `11-math.md`:

$$
\Delta q_{\rm tail}=
\frac{\gamma e^{3N_f-\gamma\tau_f}}
{2\sqrt{V_f}(\gamma-3)},
\qquad
q_f-\Delta q_{\rm tail}>0.
$$

Here $\gamma>3$ is a conservative sufficient condition for this proof, not a
physical necessity.  The code uses the actual terminal grid value $N_f=n[-1]$.

## 4. Repaired implementation failures

The implementation revisions repaired five distinct failures rather than
relabeling them as reduced claims.

1. The first Friedmann implementation added a separate $\Omega_{\rm DM}a^{-3}$
   term while also including $\rho_K$.  This double-counted dark matter and was
   removed.
2. The first sound-speed implementation used the wrong rational expression.
   Direct differentiation gives $c_s^2=u/(2\kappa+3u)$, which is now asserted
   exactly.
3. A finite $N=10$ current reserve did not prove global positivity.  The
   analytic tail-loss bound and first-zero bootstrap now close every later
   time.
4. The artifact initially depended on pytest/PYTHONPATH injection and failed as
   a standalone script.  It now resolves and validates the staging source path
   from its own location.
5. The $\gamma=30$ shooting row lost the clock anchor when it converted the
   well-resolved variable $b$ to an almost indistinguishable amplitude $A$ and
   then inverted that amplitude.  The solver now retains $b$ through initial
   data construction.

## 5. Frozen implementation hashes

At the stable implementation snapshot:

- source SHA-256:
  `2B3AC2F652F826F3EC94572F940EDAAD964387B485258C1334AB11BDE695FDA2`
- focused test SHA-256:
  `4E9811F7647BE6C527CC6AAE9471BEB982FE057389DB3BC4483DDE1FFD31F407`
- artifact script SHA-256:
  `84BD629738B6AFE7D38752CA9D01D01E0B00D196AB696032AEE4614943BD2281`
- numerical result SHA-256:
  `92AD1B126310F7AAF89BCAC097545B894CE2C13AFAF4C416659DC142D0A8BD52`

## 6. R1 유계 Gaussian 저장소 게이트

R1은 새 저장소를 만들지 않고 이 run의 다음 단일 artifact에 구현했다.

- `artifacts/r1_gaussian_reservoir_gate.py`

이 표준 라이브러리 전용 계산기는 다음을 서로 다른 게이트로 분리한다.

1. 벌크·경계·bilocal CTP 항의 질량차원;
2. 변위된 Gaussian coarse-grained mode의 Robertson 양성;
3. 규제된 bath의 retarded support와 noise-kernel 양의 준정부호;
4. 명시적 clock+bath 동역학의 총 에너지와 교환항 보존;
5. 비유계 $gT\phi$ 결합의 하한 실패와 유계 $s(T)\phi$ 결합의 하한;
6. 기존 bulk의 장파장 음의 유효 질량제곱을 예상된 열린 실패조건으로 포착.

artifact가 통과해도 $\Pi_F$의 값, 0차원 기록과 bath의 동일성, 균일 FLRW
거친갈기, 내재적 시간 화살, 암흑부문 존재량은 산출되지 않는다. 따라서 이
구현은 R1의 최소 작용 일관성 검사이지 전체 우주론의 완료 구현이 아니다.

## 7. R2 관측구간 섭동·cutoff 게이트

추가 artifact는 'artifacts/r2_fixed_metric_growth_gate.py' 한 파일이다.
이 파일은 외부 staging source에 의존하지 않고 문서화된 무차원 ODE를
표준 라이브러리로 다시 푼다. $b=\gamma\tau_0$를 shooting해
$\tau(a=10^{-4})=0$을 맞추고, 동결 numerical-results의 amplitude와
최대 $\delta$를 독립 교차검증한다. 그 위에서 정확한 $f_N$, $r(N)$,
$S_{\rm tach}$, cancellation-safe $\lambda_+$와 $k=0$ 작은 성장분을
적분한다.

같은 node에서 clock+GR single-clock subblock의 $Q_s,c_s^2$,
$d\ln(a^3Q_s)/dN$과 $d\ln(Ha^3Q_s)/dN$을 평가한다. 또 $H_0$와 reduced
$M_{\rm Pl}$을 명시해 $\Lambda_3,\Lambda_4$, 에너지 cutoff $\Lambda_E$,
물리 파수 cutoff $q_{\rm sc}$ 및 $k^4$ crossover의 최소 $\bar M$을 서로
다른 단위의 양으로 계산한다. $\bar M$은 계산 결과가 요구하는 다음 이론
계수이며 현재 작용에 몰래 추가하지 않았다.

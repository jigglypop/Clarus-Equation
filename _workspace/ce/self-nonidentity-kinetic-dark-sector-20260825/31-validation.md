# Validation — anchored self-nonidentity kinetic dark sector

Status: R2 IN_PROGRESS (R0 background and R1 focused validation complete)

## 1. Focused reproducibility

The stable staging snapshot was independently rerun with the repository Windows
Python hook.  Source compilation and the focused test returned

```text
........                                                                 [100%]
8 passed in 3.50s
```

The standalone artifact command completed with exit code 0 and regenerated
`artifacts/numerical-results.json` without an ambient `PYTHONPATH`.  The final
hashes are recorded in `30-implementation.md`.

The focused tests cover:

- exact present DE and kinetic-DM boundary fractions and $E(0)=1$;
- the exact $c_s^2$ formula and early coldness;
- full-grid positivity, shooting, current and total-continuity finite-difference
  residuals;
- grid refinement and the future reserve;
- the zero-current no-go and failure of a one-parameter shift rescaling;
- pinned DESI asset integrity and analytic BAO scale profiling;
- early-time kinetic diagnostics;
- direct reconstruction of the infinite-future tail bound for
  $\gamma=3.5,5,10,20$;
- large-$\gamma$ anchor preservation at $\gamma=30$.

No full test suite or Einstein--Boltzmann likelihood was run or claimed.

## 2. Numerical background diagnostics

The scan fixes $\kappa=10^{17}$, $a_i=10^{-4}$, and the present density
fractions, then shoots $A$ separately for each post-hoc $\gamma$.  Selected
stable results are:

| $\gamma$ | $A$ | $\rho_K(z=1100)/[\Omega_{K0}(1+z)^3]$ | $\max c_s^2$ for $10^{-4}\le a\le1$ | $\Delta q_{\rm tail}$ | global margin |
|---:|---:|---:|---:|---:|---:|
| 3.5 | 0.716074911 | 1.268520 | $1.169\times10^{-6}$ | $2.438\times10^{-6}$ | 0.100938 |
| 5 | 0.693556568 | 1.158223 | $1.102\times10^{-6}$ | $1.866\times10^{-15}$ | 0.179840 |
| 10 | 0.687051876 | 1.039733 | $9.984\times10^{-7}$ | $6.170\times10^{-44}$ | 0.192007 |
| 20 | 0.687000004 | 1.009392 | $9.694\times10^{-7}$ | $3.187\times10^{-100}$ | 0.192074 |
| 30 | 0.687000000000269 | 1.004132 | $9.643\times10^{-7}$ | $1.935\times10^{-156}$ | 0.192074 |

Thus the tested trajectories are positive and cold over the declared background
domain, and the analytic tail bound closes $q>0$ beyond the numerical grid.
The closeness to dust improves as $\gamma$ grows.  This does not derive
$\kappa=10^{17}$; that hierarchy remains an external EFT input.

## 3. DESI DR2 background comparison

The comparison uses the pinned 13-component DESI DR2 BAO vector and covariance.
For each background, one positive common scale is analytically profiled.  The
same-fraction flat $\Lambda$CDM control gives

$$
\chi^2_{\Lambda\mathrm{CDM}}=13.442354,\qquad
\mathrm{dof}=12,\qquad
\mathrm{AIC}=15.442354,\qquad
\mathrm{BIC}=16.007303.
$$

Selecting $\gamma$ after inspecting the scan consumes at least one additional
parameter, so the kinetic scan is reported with $k=2$ before any further
look-elsewhere correction.

| $\gamma$ | $\chi^2$ | $\Delta\chi^2$ | $\Delta$AIC | $\Delta$BIC |
|---:|---:|---:|---:|---:|---:|
| 3.5 | 91.087845 | +77.645492 | +79.645492 | +80.210441 |
| 5 | 45.102774 | +31.660420 | +33.660420 | +34.225369 |
| 10 | 16.187284 | +2.744930 | +4.744930 | +5.309879 |
| 20 | 13.552734 | +0.110380 | +2.110380 | +2.675329 |
| 30 | 13.451082 | +0.008728 | +2.008728 | +2.573678 |

No finite tested $\gamma$ improves the BAO $\chi^2$ over the matched
$\Lambda$CDM baseline.  Large $\gamma$ approaches that baseline while retaining
the extra parameter penalty.  This scan is therefore a consistency/falsification
diagnostic, not evidence for a finite-$\gamma$ signal, a CE prediction, or a
dark-sector discovery.

## 4. Independent audit and remaining P1 limits

An independent read-only implementation audit found no P0 after the repairs.  It
verified the ODE current sign, continuity, DM non-duplication, sound speed,
present normalization, large-$\gamma$ shooting representation, and the
first-zero tail proof.  The proof uses $q>0\Rightarrow u>0$ only up to a first
putative zero; the resulting upper bound prevents that zero and therefore is not
circular.

The following limits remain active:

- $\mathcal R_\Pi$ and its dimensional normalization are a matching axiom, not a
  derivation from standard quantum mechanics.
- $\rho_\infty$, $\Gamma$, $X_*$, $\kappa$, and $L_c$ are not predicted.
- the future branch is Cauchy/boundary data; intrinsic irreversibility requires
  an open-system retarded completion.
- $c_s^2\to0$ needs an EFT cutoff or higher-spatial-derivative completion for a
  UV/strong-coupling claim.
- tied constant and exponential coefficients need radiative protection or a
  renormalization condition.
- full CMB, LSS, lensing, nonlinear caustics, halos, and perturbative model
  selection remain unclosed.
- the analytic zero-current theorem is stronger than the current helper test:
  `zero_current_no_go()` checks the incompatible $u=0$ present DM anchor, while
  the immediate negative current derivative is verified in `11-math.md` rather
  than by a separate finite-difference test.
- the solver preserves the well-conditioned shot variable $b=\gamma\tau_0$.
  The legacy public call `present_anchor(A, config)` reconstructs $b$ from $A$
  and is ill-conditioned when $A-\Omega_{V0}$ approaches floating-point
  precision.  High-$\gamma$ clock comparisons must use the solved trajectory's
  retained shot anchor, not invert $A$ alone.

## 5. R1 focused gate — 2026-08-26

실행 명령은 다음 하나였다.

```text
.codex\hooks\python.cmd python -X utf8 _workspace\ce\self-nonidentity-kinetic-dark-sector-20260825\artifacts\r1_gaussian_reservoir_gate.py
```

종료코드는 0이었다. 주요 결과는 다음과 같다.

| 게이트 | 결과 |
|---|---:|
| 비인과 retarded-kernel 성분의 최대 절대값 | $0$ |
| noise kernel 최소 고윳값 | $-8.03\times10^{-16}$ (부동소수점 0) |
| Robertson determinant margin | $0.22$ |
| 총 에너지 최대 상대 드리프트 | $4.73\times10^{-14}$ |
| clock--bath 교환식 최대 잔차 | $1.19\times10^{-17}$ |
| 적분 중 최소 $\delta$ | $0.2088$ |
| 선형 결합의 $T=100$ 최소에너지 | $-1.63\times10^3$ |
| 유계 결합의 표본 최소에너지 | $-3.00125\times10^{-2}$ |
| 고정 배경 장파장 $m_{\rm eff}^2$ | $-2.84\times10^{-5}$ |

따라서 차원·Gaussian 양성·인과성·유한 bath 잡음 양성·명시계 총보존은
조건부 모형에서 함께 재현됐다. 마지막 음의 질량제곱 행은 실패를 숨기지
않는 예상 음성대조군이다. full metric-mixed perturbation과 $k^4$ 완성이
없으므로 전체 안정성은 통과하지 않았다.

사전 `doctor` 진단은 현재 사용자 작업트리에서
`reality_stone` 모듈이 삭제되어 `ModuleNotFoundError`로 실패했다. 독립
artifact 실행은 같은 hook의 system Python 경로에서 정상 종료했다. 삭제된
사용자 파일은 이 검증을 위해 복원하지 않았다.

## 6. R2-A fixed-metric 성장 결과 — 2026-08-26

실행 명령은 다음이며 종료코드는 0이다.

    .codex\hooks\python.cmd python -X utf8 _workspace\ce\self-nonidentity-kinetic-dark-sector-20260825\artifacts\r2_fixed_metric_growth_gate.py

| $\gamma$ | $\max |m_{\rm eff}^2|/H^2$ | $\int\lambda_+dN$ | 직접 $\pi/\pi_i-1$ | 보수 log 상계 |
|---:|---:|---:|---:|---:|
| 3.5 | $2.516\times10^{-18}$ | $1.607\times10^{-18}$ | $1.044\times10^{-18}$ | $2.016\times10^{-17}$ |
| 5 | $2.744\times10^{-18}$ | $1.937\times10^{-18}$ | $1.378\times10^{-18}$ | $2.214\times10^{-17}$ |
| 10 | $2.961\times10^{-18}$ | $2.298\times10^{-18}$ | $1.876\times10^{-18}$ | $2.409\times10^{-17}$ |
| 20 | $3.030\times10^{-18}$ | $2.442\times10^{-18}$ | $2.166\times10^{-18}$ | $2.470\times10^{-17}$ |
| 30 | $3.045\times10^{-18}$ | $2.478\times10^{-18}$ | $2.274\times10^{-18}$ | $2.484\times10^{-17}$ |

동결 amplitude의 최대 상대 오차는 $2.85\times10^{-9}$이고, 최대 $\delta$
교차검증 오차는 $3.23\times10^{-7}$이다. $\gamma=10$에서 3000/6000 step
격자 수렴의 최대 상대 변화는 $3.93\times10^{-7}$이다.

이 결과는 fixed metric과 동결 $\kappa=10^{17}$에서 질량 타키온의 누적
성장이 선택 초기조건 $\pi_i=1$, $\pi_i'=0$과 명시한 비교상계 아래 무시
가능하다는 조건부 산출이다. 임의 초기 섭동의 성장 정리가 아니며 $\pi$는
gauge-dependent이다.

같은 실행의 single-clock ADM subblock 및 cutoff 결과는 다음과 같다.

| 진단량 | 전 $\gamma$·전 관측구간의 보수값 |
|---|---:|
| $\min c_s^2$ | $9.2138\times10^{-19}$ |
| $\min Q_s/M_{\rm Pl}^2$ | $3.3167\times10^5$ |
| $\min d\ln(a^3Q_s)/dN$ | $3.93909$ |
| $\min d\ln(Ha^3Q_s)/dN$ | $3.469545$ |
| $\min\Lambda_E$ | $1.3336\times10^{-14}\,{\rm eV}$ |
| $\min q_{\rm sc}$ | $1.3893\times10^{-5}\,{\rm eV}$ |
| $\min\Lambda_E/H$ | $9.2757\times10^{18}$ |
| $\min q_{\rm sc}/[(1\,{\rm Mpc}^{-1})/a]$ | $2.1725\times10^{24}$ |
| $\max_{a,\gamma}\bar M_{\min}$ for $q_\times\le q_{\rm sc}$ | $7.3043\,{\rm eV}$ |

$\Lambda_E=\Lambda_3c_s^{7/4}$는 에너지 cutoff이고
$q_{\rm sc}=\Lambda_E/c_s$는 물리 파수 cutoff다. 이 구분을 코드와 감사표에
고정했다. baryon·radiation·reservoir perturbation을 포함한 ADM,
$k^4$ 연산자 자체, bath pole 및 Einstein--Boltzmann 관측 안정성은 실행하지
않았다.

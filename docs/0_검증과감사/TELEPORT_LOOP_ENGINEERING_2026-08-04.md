# 텔레포트 연구 루프 엔지니어링 감사 — 2026-08-04

## 판정 규칙

이 점수는 이론이 자연에서 참일 확률이 아니다. 현재 저장소에서 명시한 가정 아래
수식, 반례, 수치 회귀, 판정 문구가 서로 맞는지를 평가한다. `Exact`는 표시된
유한차원 또는 고정 ansatz 범위에만 적용하며, 미시 물질원과 섭동 안정성을 자동으로
포함하지 않는다.

## 이번 루프에서 뒤집힌 판정

| 대상 | 이전 문제 | 교정 결과 |
|---|---|---|
| 비최소장 평균 null source | `integral N_kk`를 물리 ANEC로 오인 | 물리량은 `integral N_kk/F`; 양의 `F` localized class에서 양의 제곱 항등식으로 비음수 |
| passive 내부 공명 | 1 mode 식만 있고 threshold 등호를 안정으로 오해할 여지 | 임의 유한 SPD `C`에 대한 Schur/inertia 정리, 등호는 marginal, 상수 damping·gyro도 불안정근 제거 불가 |
| fermion flux count | `int()`가 분수와 bool을 조용히 절삭 | strict integral 입력만 허용, NumPy 정수는 보존 |
| nonminimal global code-sign | 표본 최솟값을 연속 전역 판정처럼 표기 | 전부 `sampled_*`로 교정, N/2N/4N delta와 비유한 profile 차단, continuum certification은 미발행 |
| 기존 global throat | 유한 ADM만 보고 source가 국소화됐다고 암묵적으로 취급 | `p_r~-2/(3x^3)`, volume-NEC `-(2/3)ln X` 발산을 exact로 확인 |

## 핵심 수식

### 건강한 비최소장 localized class

\[
\int\frac{N_{kk}}F d\lambda
=\int\left[\frac{\phi'^2}{F}+\left(\frac{F'}F\right)^2\right]d\lambda
+\left[\frac{F'}F\right]_-^+.
\]

`F>0`이고 endpoint의 `F'/F`가 0이면 물리 effective ANEC는 비음수다. 별도 cubic
회귀에서는 numerator-only 값 `-0.06258`이 물리값 `+0.07563`으로 실제 부호가
뒤집혀, 이전 boolean의 false positive가 재현된다.

### 유한 passive multimode no-go

\[
K_{\rm eff}=K_{rr}+D-B^TC^{-1}B,\qquad C=C^T\succ0,
\]

\[
B^TC^{-1}B=\|L^{-1}B\|^2\ge0,\qquad
P^THP=\operatorname{diag}(K_{\rm eff},C).
\]

따라서 strict 안정에는

\[
D>-K_{rr}+B^TC^{-1}B
\]

가 필요하고 등호는 zero mode다. 이 정리는 유한차원 정적 quadratic system에
대해서는 exact지만, 시간주기 drive·feedback·연속 spectrum에는 적용하지 않는다.

### global throat tail 보강

기존 `Phi=e^(1-x)/2`는 radial affine ANEC는 유한·음수지만 coordinate/proper
volume-NEC가 로그 발산한다. 같은 shape에

\[
\Phi_{\rm match}
=\frac12\ln\left(1-\frac{2}{3x}\right)+\frac32e^{1-x}
\]

를 쓰면 throat Casimir data, 각 end `M_ADM/r0=1/3`, lapse 제곱 하한 `1/3`을
유지하면서 stress tail을 지수감쇠로 바꾼다. 다만

\[
\rho+p_r=-\frac{e^{1-x}}{x^2}
\left[\frac13+\frac1{3x-2}+3x-2-e^{1-x}\right]<0
\]

이므로 exotic source 자체가 제거된 것은 아니다.

## 루프 점수

| 축 | 점수 | 근거 |
|---|---:|---|
| 닫힌 명제의 수식 정합성 | 9.3/10 | 독립 부호 검산, exact identity·Schur congruence·전구간 부등식 |
| 반례·적대 입력 강도 | 9.1/10 | ANEC sign flip, fractional flux, NaN/Inf, singular/asymmetric `C`, cutoff 독립성 |
| 수치 재현성 | 9.5/10 | 공명·global focused 107개, 삭제 artifact 의존 4파일 제외 저장소 1,258개 통과·13 skip, 변경 파일 Ruff 통과 |
| 연속 영역 인증 | 8.8/10 | throat/tail은 exact; \(x=37/32\)의 explicit `K/F<0` 반례로 healthy global scalar 실패도 cutoff 없이 고정 |
| 미시 물질원 실현 | 2.0/10 | CE action/EOM, renormalized stress, backreaction, spectrum이 미도출 |
| 통과가능 장치 전체 증명 | 1.5/10 | 기하 target은 있으나 source와 선형·비선형 안정성이 열려 있음 |

따라서 현재 결과는 **증명 위생과 실패 위치의 정확도는 높아졌지만, 물리 장치의
실현 정확도가 높아진 것은 아니다**.

## 다음 우선순위

1. ADM-matched target을 독립 matter action/EOM에서 재구성하고 Bianchi 자동보존과
   실제 on-shell Noether 보존을 분리한다.
2. ADM-matched background의 gauge-fixed quadratic perturbation operator와 negative,
   zero, ghost mode를 계산한다.
3. 실제 CE pole·vertex 또는 causal boundary response에서 localized source를
   유도하고 self-reported bool 승격을 금지한다.
4. 일반 redshift travel-time 식과 print-only gate의 exit contract를 교정한다.

이 네 항목 가운데 1–3을 닫기 전에는 `wormhole realization` 또는 `teleportation
proved`로 승격하지 않는다.

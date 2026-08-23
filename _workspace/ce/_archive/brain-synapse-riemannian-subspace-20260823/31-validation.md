# 31-validation — BA-SRM1 수치·지위 검증

Status: COMPLETE

Scientific verdict: `STOP / DIAGNOSTIC_ONLY`

Confirmation: `UNTOUCHED`

## 입력 무결성

| 항목 | 결과 |
|---|---|
| DB | `synphys_r2.1_small.sqlite`, schema 22 |
| bytes | 176,771,072 |
| SHA-256 | `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53` |
| SQLite integrity | `ok` |
| strict complete 전체 | 979 pair / 512 slice |
| primary mouse V1 ex | 246 pair / 160 slice |
| primary mouse V1 in | 343 pair / 199 slice |
| event-level rows | `pulse_response=0`, `pulse_response_fit=0`, `stim_pulse=0` |
| pulse 분리 지위 | `PIPELINE_SEPARATED / ROW_LEVEL_UNVERIFIED` |

분할 후 실제 target-complete support는 ex train 159, development 39이고 in
train 222, development 59다. confirmation support는 schema 수준에서 ex 48,
in 62지만 target 값은 읽지 않았다.

## 집중 코드 검증

실행 명령:

```text
.codex\hooks\python.cmd pytest _workspace\ce\brain-synapse-riemannian-subspace-20260823\test_srm1_analysis.py tests\test_dimensionless.py -q
```

결과: `25 passed in 0.93s`.

검사 범위는 SHA-256 split, 15-basis 야코비안 유한차분, full-rank 선형 map의
SPD pullback, query-query path 부재, 고정 64개 affine rechart, bootstrap
support-minimum statistic과 무차원 로그/Fisher/kernel 인자를 포함한다.

차원 상태는 `무차원`이다. $|r_1|/r_{\rm ref}$, $L/1\,\mathrm m$,
$R_{\rm in}/1\,\Omega$, $\tau_m/1\,\mathrm s$만 로그에 넣고, amplitude target도
$r_{\rm ref}$로 나눴다. $y,z$가 무차원이므로 $J$, $R$, $g_{\rm resp}$와
$d_g^2/(2\ell_g^2)$도 무차원이다. 이 검사는 차원 정합만 보이며 생물학적
정당성이나 예측 성공을 뜻하지 않는다.

## 실제자료 결과

| stratum | rank lower 2.5% | threshold | gauge | 최선 control | $\Delta$ELPD | $2SE$ | 양의 slice 비율 | 판정 |
|---|---:|---:|---|---|---:|---:|---:|---|
| ex | $1.08724\times10^{-10}$ | $10^{-4}$ | PASS | direct quadratic | -18.4854 | 86.9486 | 77.78% | STOP |
| in | $2.40461\times10^{-11}$ | $10^{-4}$ | FAIL | diagonal constant metric | -4.85688 | 9.60017 | 18.92% | STOP |

두 stratum 모두 nominal support에서는 수치상 full rank였고 1,000 bootstrap의
full-rank fraction도 1이었다. 그러나 가장 약한 고유방향이 bootstrap에서 사전
문턱보다 각각 약 $9.2\times10^5$배와 $4.2\times10^6$배 작았다. 따라서
“안정적으로 식별된 4차원 Riemannian metric”이라고 부를 수 없다.

억제성 gauge 실패는 `orthogonal-83103`에서 generalized-spectrum 상대오차
$1.34520\times10^{-8}$가 고정 허용치 $10^{-8}$을 넘은 한 건이다. line element
오차는 $1.37687\times10^{-14}$, prediction 오차는 $2.33841\times10^{-15}$로
작았지만 계약은 세 조건을 모두 요구하므로 FAIL이다. 이 경계 실패를 반올림해
PASS로 바꾸지 않았다.

흥분성 가변계량은 development에서 direct quadratic 반응식보다 ELPD가
18.49 낮았다. 억제성 가변계량은 diagonal constant metric보다 4.86 낮았다.
따라서 state-dependent geodesic이 직접 반응식이나 더 단순한 계량을 넘어서는
추가 예측정보를 보이지 않았다.

## confirmation 봉인

최종 artifact는 다음을 기록한다.

```text
status = DEVELOPMENT_STOP_CONFIRMATION_UNTOUCHED
confirmation_contact = false
ex.confirmation.status = NOT_CONTACTED
in.confirmation.status = NOT_CONTACTED
```

rank, gauge, development survival의 교집합이 두 stratum 모두 거짓이므로
`query_targets()`의 confirmation branch는 실행되지 않았다.

## 독립 감사

- 형식 gate 감사: outcome 접촉 전 `Gate: PASS`.
- 사후 수학 감사: 두 `RANK_UNIDENTIFIED`와 development STOP이 계약에서
  정확히 따른다고 판정했다.
- 사후 구현 감사: bootstrap statistic P1을 발견했다. 더 엄격한 support-minimum
  구현으로 수정 후 재감사에서 열린 P0/P1 0건, `PASS`를 받았다.
- 사후 지위 감사: `BA-SRM1-C1/C2/C5 PASS`, `C3 조건부 정리만 PASS`,
  `C4 STOP`, 전체 `STOP / DIAGNOSTIC_ONLY`로 판정했다.

## Claim ceiling

이번 검증으로 유지되는 것은 typed factor schema, 측정모형과 조건부 SPD 정리다.
실제 데이터에서는 rank와 예측 gate가 실패했다. conductance, $Npq$, release
probability, directed delay, STDP, eligibility, homeostasis, morphology,
curvature-memory, 기억 또는 AGI 기전은 검증하지 않았고 승격하지 않는다.

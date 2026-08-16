# Phase A 형식 지위·구현 승인 감사

Status: COMPLETE

Gate: PASS

## 1. 감사 범위와 판정

`00-contract.md`, 스킵 사유가 명시된 `10-sources.md`, COMPLETE인 `11-math.md`와 `12-routes.md`, `artifacts/dimensionless-audit.md`, `artifacts/implementation-preflight.md`를 전부 감사했다. 활성 수학 P0는 0개다. 구현으로만 닫을 수 있는 P1은 `PA-I1`--`PA-I3`, development 실행으로만 판정할 수 있는 P1은 `PA-H1`, `PA-H2`로 정확히 남아 있으므로 이를 정리나 산출로 승격하지 않는 조건에서 구현 진입을 승인한다.

이번 PASS는 Phase A V1 격리 benchmark의 구현 승인이지 연구 가설의 확인, 정본 승격 또는 공용 runtime 편입 승인이 아니다. 구현 변경 표면은 아래 네 신규 파일과 `.gitignore`의 exact-path 예외 한 줄로 한정한다.

## 2. 주장별 지위 판정

| Claim ID | 현재 지위 | 실제 지위 | 심각도 | 근거와 구현 조건 |
|---|---|---|---|---|
| `PA-N1` | 선행 [정리: no-go]의 executable fixture | [정리: 조건부 no-go] + 미구현 회귀 fixture | 없음 | `00-contract.md:65`, `11-math.md:100-125,195`. 미지의 가역 similarity 아래 latent support가 달라도 관측열이 같다는 증명은 닫혔다. 구현은 서로 다른 support와 동일 관측열을 exact fixture로 재현해야 하며 이는 build 승인 조건이다. |
| `PA-T1` | 선행 [정리: 조건부]의 executable fixture | [정리: 조건부] + 미구현 회귀 fixture | 없음 | `00-contract.md:66`, `11-math.md:41-69,196`. noiseless known-identity에서 joint stacked design의 rank가 정확히 $Kn+m$일 때만 $(A_0,\ldots,A_{K-1},B)$가 유일하다. 각 문맥 $X_z$의 개별 full rank만으로 인증하면 안 된다. 최대 계수 오차 $10^{-10}$ gate는 noiseless fixture에만 적용하며 fixture 통과는 build 승인 조건이다. |
| `PA-T2` | [정리 후보: 구현 계약] | [정리: 조건부 no-go] + 미구현 fail-closed fixture | 없음 | `00-contract.md:67`, `11-math.md:71-96,197`. rank 부족이면 null 방향에서 관측이 같은 다른 계수가 존재하므로 수학 지위는 닫혔다. ridge 해의 존재를 exact identification으로 쓰지 않고 certificate를 거부하는 fixture는 build 승인 조건이다. |
| `PA-D1` | [정의] | [정의] | 없음 | `00-contract.md:68`, `11-math.md:198`. anatomy, latent causal support, 관측좌표 predictive transition을 서로 다른 typed field로 유지한다. 지위 정합이다. |
| `PA-D2` | [정의] | [정의: 정규화된 합성 좌표] | 없음 | `00-contract.md:47-61,69`, `dimensionless-audit.md:7-27,85-100`. 정의와 식은 무차원으로 닫혔다. 실제 manifest scale·finite·positive 검사와 격리 certificate는 build 승인 조건이며, 무차원 통과는 식별성 또는 성능 증명이 아니다. |
| `PA-I1` | [미완성: 구현] | [미완성: 구현] | P1 | `00-contract.md:70`, `11-math.md:178-182`, `implementation-preflight.md:105-116`. graph, trajectory, intervention, evaluation-noise, shuffle, bootstrap RNG를 안정적 namespace로 분리하고 동일 tuple의 byte-identical replay와 role 변경 격리를 시험해야 한다. 이는 독립 PRNG stream이라는 명시적 모델 공리이지 통계적 독립성 정리가 아니다. |
| `PA-I2` | [미완성: 구현] | [미완성: 구현] | P1 | `00-contract.md:71`, `11-math.md:184-189`, `implementation-preflight.md:118-139`. learner는 train의 $x,u,z,x_+$와 사전 고정 hyperparameter만 받는다. truth $A_z,B$, test target과 공통 manifest $\sigma$는 fit 반환 뒤 evaluator/scorer만 사용하며 mutation test로 경계를 확인한다. |
| `PA-I3` | [미완성: 구현] | [미완성: 구현] | P1 | `00-contract.md:72,129`, `11-math.md:115-125,202`. exact-edge certificate는 `known_identity AND declared_linear_class AND full_rank AND finite_valid_inputs`의 정확한 conjunction일 때만 true다. 어느 한 항이라도 false이면 fail closed한다. known mask와 unknown mix는 예측을 채점할 수 있어도 exact-edge는 항상 false다. |
| `PA-H1` | [미완성: development 비교] | [미완성: 경험 비교] | P1 | `00-contract.md:73,130`, `11-math.md:203`, `12-routes.md:13-26,41-52,74-83`. primary는 R1 shared-$B$ context fit 대 R3 pooled fit의 같은 split·ridge·공통 $\sigma$ paired 비교다. strict 우위는 정리가 아니며 graph-seed별 $\Delta_s\le0$이면 STOP한다. R2와 R4는 V1 primary에 열지 않는다. |
| `PA-H2` | [미완성: integrity 비교] | [미완성: 경험 비교] | P1 | `00-contract.md:74,131`, `11-math.md:204`. nonzero intervention signal에서 tag/time shuffle 악화를 development로 검사한다. 동률이면 STOP하며 $B=0$ 같은 허용 반례 때문에 보편 정리로 올릴 수 없다. |
| `PA-X1` | 활성 제외 | 활성 제외 경계 | 없음 | `00-contract.md:75,87`, `11-math.md:205`, `12-routes.md:83`. 결과를 SCC 효능, 기억, 생물학, 의식 또는 AGI 증거로 해석하지 않는다. 관련 certificate와 문구는 false 또는 absent여야 한다. |

## 3. 숨은 공리와 선택 회계

다음 아홉 항목은 정리가 아니라 명시적 공리·사전 선택이다.

1. 상태가 고정된 known-identity chart에서 완전 관측된다는 조건.
2. 유한 이산시간 선형 class와 관측된 context label.
3. 문맥별 $A_z$와 문맥에 공유되는 $B$라는 생성·추정 구조.
4. noiseless exact fixture와 noisy development 비교를 분리한다는 선택.
5. noisy population 해석에서 conditional-zero-mean residual과 full-rank moment 조건.
6. 정규화된 무차원 좌표와 positive finite generator noise scale.
7. primary NLL에는 생성 전에 manifest에 고정된 truth $\sigma$를 scorer-only 공통 scale로 양 arm에 동일 적용한다는 선택. estimator, test residual 재추정, model별 scale에는 주지 않는다.
8. R1을 candidate, R3를 필수 pooled baseline으로 고정하고 추가 nominal dof $(K-1)n^2$를 공개한다는 target-aware 선택.
9. graph seed를 통계단위로 하고 pilot/development/confirmation 역할과 RNG namespace를 분리한다는 재현 규약.

이 선택이 원장에 드러나 있으므로 무입력 산출 또는 제1원리 결과로 오인할 열린 P0는 없다.

## 4. 구현 승인 범위

승인하는 신규 파일은 정확히 다음 네 개다.

1. `reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py`
2. `tests/test_causal_recurrent_geometry_benchmark.py`
3. `experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`
4. `examples/agi/causal_recurrent_geometry_development_run.py`

추가로 `.gitignore`에는 정확히 다음 한 줄만 추가할 수 있다.

```gitignore
!experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
```

`implementation-preflight.md:14,37-46`의 ignored JSON P0는 위 exact-path exception이 계약 `00-contract.md:77-87`에 명시적으로 포함되어 있으므로 **승인된 좁은 수정으로 해소 가능하며 이 gate에서 해결 경로가 닫혔다**. 실제 구현 후에는 기존 `.gitignore` 내용이 그대로 보존됐는지, exact exception이 한 번만 추가됐는지, `git check-ignore`에서 해당 JSON이 더는 ignore되지 않는지를 검증해야 한다. 예외 누락, wildcard 완화 또는 다른 `.gitignore` 편집은 구현 gate 실패다.

production module은 self-contained NumPy 표면으로 유지한다. test와 runner는 경로를 resolve한 뒤 동일 source bytes를 SHA-256으로 묶어 `compile`/`exec`하는 isolated load를 사용하고 package initializer를 실행하지 않는다. `reality_stone/python/reality_stone/clarus/__init__.py`, 다른 runtime/export/default flag, 정본, 기존 SCC·metric 코드는 수정하지 않는다.

current infinite-tail 구현, dirty V15, untracked V16/V17, V9 STOP candidate와 과거 confirmation seed/result는 import·복사·증거 사용을 모두 금지한다. clean predecessor에서 재사용할 수 있는 것은 hash/seed 분리와 fail-closed 형식 패턴뿐이며 판정값이나 seed는 가져오지 않는다.

## 5. confirmation 봉인과 development 판정

confirmation은 `reserved_unopened`이고 `execution_authorized: false`다. 이번 runner에는 confirmation 함수·CLI mode·공개 seed 생성 경로를 두지 않는다. development와 confirmation seed 집합 비중첩, confirmation 접근의 namespace 생성 전 거부, 결과의 `confirmation_status: reserved_unopened`를 test한다. 이 run에서 confirmation 결과·receipt를 만들거나 열면 Gate 위반이다.

development primary는 R1과 R3가 동일 train batch, held-out batch, ridge와 scorer-only manifest $\sigma$를 받는 graph-seed paired 비교다. graph seed별 NLL과 $\Delta_s$, mean, median, paired bootstrap interval, nominal dof를 모두 보존한다. frame을 독립 통계단위로 세지 않는다. R1의 strict 승리나 shuffle 악화를 구현 성공 조건으로 강제하지 말고 실패하면 각각 `PA-H1 STOP`, `PA-H2 STOP`으로 정직하게 남긴다.

## 6. P0/P1/P2 및 삭제 범위

- P0: 0개. ignored JSON은 exact exception 한 줄이라는 승인된 해결책으로 닫혔고, 실제 적용 검증은 구현 P1이다.
- P1: 5개 claim group. `PA-I1`--`PA-I3` 구현 certificate 세 개와 `PA-H1`, `PA-H2` development 판정 두 개다.
- P2: 0개.
- 완전 반례로 삭제할 활성 부모 주장: 0개. strict 성능 우위와 shuffle 악화는 애초 정리가 아니라 미완성 경험 비교로 좁혀져 있다.
- 보존된 활성 제외: 1개(`PA-X1`). SCC·기억·생물학·의식·AGI로의 확장을 열지 않는다.

검사 회계는 비자명 claim 11개, 조건부 정리/no-go 3개, 정의 2개, 명시적 공리·선택 9개, 구현·경험 미완성 5개, 삭제 0개다. 모든 claim ID의 실제 지위가 근거와 일치한다.

## 7. 구현 후 재검토 조건

이 PASS 이후 허용되는 repository 변경은 신규 네 파일과 `.gitignore` exact exception 한 줄뿐이다. focused test는 exact recovery, rank refusal, similarity no-go, observation refusal, dimensionless/domain refusal, seed isolation, no-hidden/future, R1 대 R3 공정성, shuffle STOP, manifest tamper, isolated import와 confirmation 봉인을 모두 공격해야 한다. 이 중 하나라도 실패하면 해당 구현 claim은 미완성으로 유지하고 build/final gate를 통과시키지 않는다.

Gate: PASS

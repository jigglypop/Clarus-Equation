# Causal Recurrent Geometry 12주 실행 로드맵

Status: COMPLETE

이 로드맵은 대용량 connectome 분석 전에 합성 식별성·SCC 유용성·기억 접근 metric을 순서대로 죽여 보는 최소 프로그램이다. 각 단계의 confirmation 결과를 본 뒤 앞 단계의 모형을 다시 조정하지 않는다.

| 주 | 작업 | 동결 산출물 | 다음 주 진입 조건 |
|---:|---|---|---|
| 1 | observation-only 비식별성 fixture, 완전관측 선형 식별성 fixture, graph/context generator 작성 | generator schema, seed namespace, unit invariants | ground-truth replay와 음성 반례 재현 |
| 2 | ridge/sparse VAR, GRU/MLP, graph transition baseline과 compute ledger 작성 | baseline API, parameter/FLOP report | 모든 baseline이 동일 split·loss 사용 |
| 3 | $G/F/z_t$ factorized estimator의 최소형 구현 | model schema와 serialization | unknown-mix에서 exact-edge claim이 fail-closed |
| 4 | node/module intervention과 held-out context 평가기 구현 | intervention manifest와 primary metric | target/time shuffle 음성 대조 통과 |
| 5 | development·validation 실행과 고장 분석 | model choice log, 변경 횟수, 후보 소거표 | architecture와 hyperparameter freeze |
| 6 | Phase A confirmation 실행 | locked seed 결과, paired CI, `GO-A/STOP-A` | `GO-A`일 때만 empirical structure 준비 |
| 7 | SCC, community, spectral, random, learned partition을 같은 압축률로 구현 | compression curve evaluator | partition label 누수 0건 |
| 8 | Phase B confirmation과 cut/reset/label-shuffle ablation | `GO-B/STOP-B`, scale-dependent edge ledger | SCC primary 우위 또는 해석 도구로 축소 |
| 9 | controllability-Gramian memory generator와 partial-cue task 구현 | reachable-subspace certificate | analytic minimum-energy fixture 일치 |
| 10 | Euclidean, weight/activity, covariance, learned quadratic baseline 구현 | equal-budget metric suite | rank·conditioning·coordinate checks 통과 |
| 11 | Phase C validation 후 geometry family freeze | metric choice log, null models | primary와 kill test가 사전 고정됨 |
| 12 | Phase C confirmation, 논문 A 초안, empirical data access decision | `GO-C/STOP-C`, manuscript outline, data manifest | 별도 empirical run 개시 여부 결정 |

## 주차별 금지선

- 1--6주에는 기억·의식 성능을 headline으로 보고하지 않는다.
- Phase A confirmation을 연 뒤 graph generator, loss, primary endpoint를 바꾸지 않는다.
- 7--8주의 SCC 결과로 정확한 connectome 복원을 주장하지 않는다.
- 9--12주의 metric은 $A,B$에서 계산한 operational readout이며 독립 생물 기전으로 부르지 않는다.
- MICrONS 또는 다른 외부 자료는 별도 manifest와 라이선스 검사가 끝나기 전에 내려받지 않는다.
- 의식·자아는 이번 12주의 실험 endpoint가 아니다.

## 최소 인력·자원

최소 구성은 동역학/인과 모델 1명, 그래프·수학 검증 1명, 실험·통계 검토 1명이다. 한 사람이 수행하더라도 model author와 confirmation evaluator 역할을 서로 다른 프로세스와 frozen artifact로 분리한다. 합성 단계는 단일 GPU 또는 CPU 병렬로 시작하고, 외부 connectome 저장공간은 자료 manifest가 확인된 후 별도 산정한다.

## 논문 산출 순서

12주 종료 시 첫 원고는 **Identifiability of Causal Recurrent Geometry**로 제한한다. SCC 결과가 양성이면 predictive coarse-graining 절을 포함하고, 음성이면 SCC no-advantage 결과를 그대로 포함한다. geometric memory는 Phase C가 통과해도 별도 두 번째 원고로 분리해 구조복원 성과가 기억 주장으로 자동 확장되지 않게 한다.


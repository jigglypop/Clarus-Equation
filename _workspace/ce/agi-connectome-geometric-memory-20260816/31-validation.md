Status: COMPLETE

# 31 — 인과적 재귀 기하 연구 설계 검증 원장

검증일: 2026-08-16  
범위: 연구 계약, 독립 출처·수학 레인, 형식 지위, 사전등록 청사진과 최종 연구제안서

## 판정 경계

아래 PASS는 문서 구조, 명시된 수학 반례·조건부 정리와 형식 지위가 재현된다는 뜻이다. 모델 구현, empirical result, MICrONS causal recovery, 기억기전, 의식 또는 AGI를 검증한 결과가 아니다. 제품 코드, 활성 정본과 외부 데이터는 변경하거나 내려받지 않았다.

## 실행 결과

| 검사 | exit | 결과 | 허용 해석 |
|---|---:|---|---|
| CE contract hook | 0 | OK contract | 연구 질문·주장·kill test 계약 존재 |
| CE lanes hook | 0 | OK lanes | source/math/routes 레인 상태 완결 |
| 독립 수학 verifier | 0 | exact fixtures 전체 통과 | SCC·식별성·lumpability·gauge·Gramian 정리와 반례의 산술 재현 |
| CE gate hook | 0 | OK gate | 반례 있는 부모 범위를 활성 결론에서 제외한 형식 감사 통과 |
| CE build hook | 0 | OK build | 구현 SKIPPED 경계와 validation stage 완결 |
| CE final hook | 0 | OK final | 정확한 8-stage 연구 run 구조 완결 |

## 독립 검산이 재현한 핵심 fixture

1. self-loop 없는 유향 그래프 $n=4$ 전수 4,096개에서 condensation acyclic 및 같은 semantics의 두 번째 SCC가 모두 singleton임을 확인했다.
2. diagonal latent transition과 off-diagonal similarity 변환이 동일 관측열을 만드는 exact 반례를 확인했다.
3. $X\to Y$와 $Y\to X$ Gaussian 구조가 같은 covariance를 만드는 관측 방향 반례를 확인했다.
4. full-rank LTI design에서 원래 $[A\ B]$를 정확히 복원하고, rank-deficient design에서는 다른 계수가 같은 예측을 만드는 것을 확인했다.
5. 같은 SCC aggregate에서 다음 aggregate가 각각 2와 1이 되는 predictive-insufficiency 반례를 확인했다.
6. $W'=SW$, $g'=S^{-T}gS^{-1}$가 같은 $W^TgW$와 모든 sample cost를 만드는 gauge를 확인했다.
7. 정방향과 역방향 SPD 경로 cost가 모두 25인 directionality no-go를 확인했다.
8. controllability fixture에서 $W_2^{-1}$과 minimum energy 40을 exact rational arithmetic으로 재현하고 rank-deficient unreachable target을 거부했다.

검산 구현과 원문 출력은 artifacts/verify_cgm_math.py와 artifacts/verify_cgm_math.log에 있다.

## 문서 완결성

- root stage는 00, 10, 11, 12, 20, 30, 31, 40의 정확한 8개 파일로 제한한다.
- 30-implementation은 연구주제 설계 run이므로 명시적으로 SKIPPED다.
- 제품/정본 변경은 없고 새 파일은 이 run 아래에만 생성했다.
- source 판정은 1차 논문, 공식 dataset/repository와 라이선스에 제한했다.
- 현재 [예측]은 0개이며 empirical 비교는 실제 preregistration 전까지 [미완성]으로 유지한다.

## 미실행 항목

- 실제 seed·표본수·alpha와 model/evaluator hash를 고정한 preregistration
- 합성 benchmark 구현과 confirmation
- MICrONS ingest 또는 대용량 학습
- longitudinal memory dataset 분석
- 제품 AGI runtime 통합

이 항목들은 누락된 현재 검사가 아니라 별도 승인과 manifest가 필요한 후속 연구 단계다.

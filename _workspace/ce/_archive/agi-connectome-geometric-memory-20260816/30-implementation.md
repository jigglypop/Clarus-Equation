Status: SKIPPED (연구주제 설계 run이며 제품 코드·정본 구현은 요청 범위가 아님)

# 30 — 구현 범위

이번 요청은 사용자의 `Connectome / SCC / Riemannian Memory` 가설을 검증 가능한 연구 프로그램으로 정리하는 것이다. 따라서 제품 AGI runtime, 활성 정본, 기존 SCC·metric API와 외부 데이터셋은 수정하거나 내려받지 않았다.

후속 구현은 `20-audit.md`가 허용한 좁은 범위에서 별도 run으로 시작해야 한다. 최소 표면은 다음 네 가지다.

1. 합성 선형·비선형·문맥전환 동역학과 알려진 개입을 생성하는 격리된 benchmark
2. graph/environment 단위 blind split, primary endpoint와 STOP rule을 고정한 preregistration
3. observational-equivalence, intervention separation, no-future/no-hidden과 compute parity를 검사하는 integrity tests
4. 한 번만 여는 confirmation manifest·receipt runner

기존 V9/V15/V16/V17 자산은 `artifacts/repository-reuse-map.md`의 provenance 경계를 따른다. 특히 V9의 실패한 후보 구조와 sealed seed, dirty/untracked metric 구현, 정본보다 넓은 infinite-tail 후보를 새 성공 모델 또는 안정 API로 재사용하지 않는다.

MICrONS 전량 ingest와 geometric-memory 모델은 첫 합성 식별성 gate의 선행 조건이 아니다. 데이터 manifest, 라이선스, 독립 표본단위와 endpoint가 별도 승인된 뒤 후속 단계로 분리한다.

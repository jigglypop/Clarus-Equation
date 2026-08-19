# 최종 보고서: 완전 3D 피질 리본 계량과 PFC 검증 run

Status: COMPLETE

작성일: 2026-08-19

## 1. 한 줄 요약

완전 6성분 3D SPD 계량의 수치 커널(Gate A-KERNEL)은 Rust 6/6 테스트,
해석적 픽스처 39/39, 독립 NumPy 오라클 일치, 직렬/Rayon 결정성까지
전부 통과했다(산출). Gate B 합성 보정은 실행하지 않았으므로 모든
생물학적 경로의 동결 프로토콜 결과는 봉인 상태로 남는다. 별도로,
Wójcik et al. 공식 공개 캐시에 대한 기술적(descriptive) 방정식 배터리는
Exp2 규칙 일반화 구간의 1↔말기 선택성 기하 교환가능성 기각(p≈5×10⁻⁵,
4개 비닝에서 안정)과 Exp1 초기 학습 구간의 비기각(p≈0.096)을 산출했다.
목표 인과 사슬 ΔW^s→Δg_M→Δp (계약식 (1))의 형식 지위는 **미완성(가설)**
그대로다.

## 2. 게이트 상태

| 게이트 | 상태 | 근거 |
|---|---|---|
| 수식 감사 (20-audit) | PASS | P0 없음, P1 3건 문안 반영 완료. math-verifier 수정 루프 2/2 소진 |
| Gate A-KERNEL | PASS | Rust 6/6, 픽스처 39/39, 오라클 일치, 결정성 (`artifacts/GATE_A.md`) |
| Gate A-LOCK | 미실행 (선택) | Gate B 일회 실행 직전에만 요구되는 출판 게이트 |
| Gate B (합성 보정) | 미실행 | 200 true + 6×200 null 데이터셋 미개봉. 실행 전까지 생물학 결과 봉인 |
| Gate C (생물학 적격성) | 메타데이터만 | 원본 아카이브 미다운로드, 라이선스 확정 전 |

Gate A 통과가 지지하는 것은 수치 커널뿐이다. 어떤 생물학 이론 지위도
이 PASS 문자열로 승격되지 않는다.

## 3. 동결 경로 처분

| 경로 | 처분 | 사유 |
|---|---|---|
| `R-KERNEL-3D` | `PASS` | Gate A-KERNEL 전 항목 통과 |
| `R-SYNTH-3D` | `NOT_EXECUTED` (봉인) | Gate B 미실행. 추정기 주장 없음 |
| `R-NULL-3D` | `NOT_EXECUTED` (봉인) | 동일. 선행 run의 72/100 오선택 실패를 대체할 보정은 아직 미검증 |
| `R-PFC-WOJCIK` | `ACCESS_BLOCKED` | Gate B 봉인 + 데이터 재사용 라이선스 미확정(`ELIGIBLE_METADATA_ONLY_PENDING_LICENSE`) |
| `R-PFC-CALANGIU` | `ACCESS_BLOCKED` | Gate B 봉인. 메타데이터 적격(`ELIGIBLE_METADATA`, CC BY-SA 4.0 미러 확인) |
| `R-PFC-KIANI` | `ACCESS_BLOCKED` | Gate B 봉인. 메타데이터 적격 |
| `R-PFC-RIBBON` | `UNTESTABLE_MISSING_INPUT` | 연속 (u,v,ℓ)·h·W^s·동일유닛 종단 식별을 함께 주는 공개 소스 부재 |

상류 게이트 실패·미실행으로 봉인된 하류 경로를 건너뛴 것은 미완성
계산이 아니라 올바른 과학적 처분이다(12-routes §1).

## 4. 공식 캐시 실데이터 방정식 배터리 (아티팩트 수준, 기술적)

`artifacts/official-pfc-metric-equation-test.md` (`REAL_DATA_EQUATION_TEST`)는
동결 R-PFC 경로 프로토콜 밖에서, 저자 공식 저장소 커밋
`48ada805`의 공개 처리 캐시(피클, SHA-256 고정)만으로 수행한 방정식
검사다. 32.11 GB 원본 세션 아카이브는 내려받지 않았다. 이것은 사후
탐색(discovery-level) 기술 통계이며 사전등록 경로 결과가 아니다.

핵심 산출:

- **Exp1 초기 학습**: 1↔말기 선택성 공분산의 AIRM 거리에 대한 행
  교환가능성 귀무를 기각하지 못함 (p(total)=0.0957; 고정편향 대조군 일치).
- **Exp2 규칙 일반화**: 교환가능성 기각 (p(total)=0.00005), 공식 3·4·5·6
  비닝 전부에서 안정. GL(3) 불변 통계(AIRM·Jeffreys)와 고정차트
  통계(log-Euclidean·Bures)가 같은 방향.
- **후기 3D 중간고리 발견 검사**: 과제1 기하 변위 d_k와 과제2 정합 q_k의
  Pearson r=0.9084, 단측 p=0.0446 / 양측 p=0.0885 (20,000 셔플). 같은
  시간창의 shape/XOR 대조는 재현 실패 (r=0.016, p=0.967) — 축별 균일
  라우팅 법칙은 지지되지 않음.
- **수치 정정 2건**: 붙여넣기 해석의 `-0.149823→0.368520` 열은 저자
  figure_4.py의 초기 **colour** 변수 산출이고(수출 그림 라벨은 context),
  `0.149701→0.475854`는 후기 **shape 선택성 정합**이다. 어느 것도 계량
  정합이 아니며, pooled-information 코사인을 g_k로 개명할 수 없다.
- 70–100 ms `[set, set*context, context]` 계량과 Fig. 4 선택성 설계 사이에
  동일 좌표 브리지가 없으므로 주차트 문맥 결합은 보고하지 않는다.

## 5. 형식 지위 판정

| 항목 | 지위 |
|---|---|
| 완전 6성분 SPD 계량 정의·차트 법칙·상대 변형 (식 (3)–(7)) | 정의 |
| 접힌 리본 당김 계량의 내부 평탄성 (h=r*δ) | 정리 (해석적 + 픽스처 확인) |
| Gate A-KERNEL 수치 결과 (39/39, 오라클 일치) | 산출 (기계 검증) |
| Exp1 비기각·Exp2 기각·후기 3D 상관 | 산출 (기술적, 사후 탐색) |
| W^s→g 생성자 Φ (식 (9)) | 미완성 (가설; 합성 후보군만 동결) |
| 계량→경로 브리지 B (식 (10)) | 미완성 (모형 가설) |
| 인과 사슬 ΔW^s→Δg_M→Δp 전체 (식 (1)) | 미완성 |
| "뇌의 정준 계량 발견" 류 진술 | 금지 (계약 §9) |

## 6. 남은 결함과 재개 조건

- **Gate B 미실행 (BLOCKED)**: 재개 조건은 Gate A-LOCK 재생성 후 동결
  이진으로 200 true + 1,200 null 일회 실행. 통과 기준은 11-math.md에
  동결됨(회복 ≥180/200, null 가족별 오승격 ≤4/200, CP 상한 ≤0.05,
  10-검정 Holm, 직접모형 대비 0.01 nat 비열등).
- **Wójcik 라이선스**: Dryad 기계가독 라이선스 문자열을 취득 매니페스트에
  기록하기 전까지 `ACCESS_BLOCKED` 유지.
- math-verifier 수정 루프 2/2 소진 — 잔여 결함은 없으나 후속 run에서
  수식 변경 시 새 감사가 필요하다.

후속 run은 본 run을 PREDECESSOR로 지정하고, Gate A 결론(커널 산출)과
소스 적격성 원장을 재유도 없이 경로 인용한다.

## 7. 산출물 색인

- `artifacts/GATE_A.md`, `gate-a-fixtures-r6-release-final6.json`,
  `oracle-r6-release-final6.json` — Gate A 증거
- `artifacts/rust/nrm3d-core` — 격리 Rust 커널 크레이트
- `artifacts/reference_oracle.py` — 독립 NumPy 오라클
- `artifacts/official-pfc-metric-equation-test.md`,
  `run_official_pfc_metric_equation_test.py` — 공식 캐시 방정식 배터리
- `artifacts/official-pfc-processed-geometry.md`,
  `run_official_pfc_processed_geometry.py` — 처리 기하 보조 검사

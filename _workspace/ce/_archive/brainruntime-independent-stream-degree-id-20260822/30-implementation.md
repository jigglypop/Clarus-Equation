# BA-TR30 implementation

Status: COMPLETE

## runtime 동결 검증 (선행 조건)

`sha256sum reality_stone/python/reality_stone/clarus/runtime.py` →
`5dc1ba5722ffa430f5c4dd4502defd49130dbe36c7f5522f28a7df36338fed26` — 계약 §2
(BA-TR20 수리판) 동결 SHA와 일치. 불일치 없음, 구현 진행.

## 변경 (계약 §4 승인 범위 내, 전부 신규 파일)

| 파일 | 내용 |
|---|---|
| `reality_stone/python/reality_stone/clarus/runtime_independent_stream_degree_id.py` | TR30 전용 동결 학습 연산자·fold 생성·은행·대조군. 총차수 특징 φ_d (3/6/10열, witness 생성기용 d=4는 15열), C_d=Φ_d⁺Y, studentized PRESS s′_d=(1/N)Σ‖e_i‖/√(1−h_ii) (hat-matrix 항등식, 11-math (b)가 재적합 동치 확인), 파시모니 d̂=min{d: s′_d≤1.5·min+1e−8}, 기권 min s′_d>max(1e−8,8η)→`CLASS_EXTERNAL_ABSTAIN`, cond(Φ₃)>1e6→`CUE_DEGENERATE`. fold: z~N(0,I₂) seeded 25행(24 훈련+query), C\*~U(−1,1) seeded, ε~N(0,I₆), seed당 주 9(D×H)+witness 1(d=4, η=1e−3). η는 공리대로 학습자에 공급. K=8 은행: 진리 1(생성기 산출 y_q^clean, 모델 비경유)+타 셀 clean 4+matched-norm distractor 3(기준=비query 24셀 clean content 평균 norm), seeded 순서 셔플, SHA-256+순서 카운터 영수증을 ŷ_q 계산 전에 기록, 위반=`BANK_RECEIPT_FAIL`. 대조군: association shuffle(roll, 기권=endpoint 전 기각), wrong-cue(은행 내 타 셀 cue 예측→진리 선택 금지), 강제 d=1 on d\*∈{2,3}(게이트 실패 필수), selection ablation ρ=0(진단 기록만). TR28 모듈에서 `_tensor_hash`·`_relative_residual` 재사용 임포트 |
| `reality_stone/python/reality_stone/clarus/runtime_independent_stream_degree_id_benchmark.py` | CLI (TR28/29 관행 동일: generate-calibration/generate-development/calibration/development) |
| `tests/test_runtime_independent_stream_degree_id.py` | focused 2건: (1) studentized PRESS hat 항등식 vs 명시 LOO 재적합 동치, (2) calibration artifact 재분석 — 9 주 fold 식별·은행 진리 선택·witness 기권·영수증 순서 |

runtime 수정 없음. 기존 모듈 수정 없음. 감사 미승인 리팩터링·기능 추가 없음.

## 실행 순서 (§4.6 준수)

1. `.claude/hooks/python.cmd doctor` → PASS (Python 3.11.9, PYTHONPATH=reality_stone/python, bytecode_disabled).
2. calibration `117001` 먼저 실행 → `DEGREE_ID_CALIBRATION_PASS` **1차 통과**. apparatus 결함 없음 — R-loop·revision 불요, `calibration-log.md` 미작성 (D→I→P→C→B→T 분류 발동 사유 없음).
3. calibration 통과 후 development `117101..117116` 개봉 → `DEGREE_ID_DEVELOPMENT_GO`.
4. confirmation `117201..117232`는 **열지 않았다** (봉인 유지, `confirmation_opened: false`).
5. calibration 통과 후 차수·잡음·split·대조군·threshold·상수 변경 없음 (§4.7-3).

## 명령과 원래 결과

```
.claude/hooks/python.cmd doctor
  → {"status": "PASS", ...}
.claude/hooks/python.cmd python -B -m reality_stone.clarus.runtime_independent_stream_degree_id_benchmark \
  --stage generate-calibration --output .../artifacts/calibration-input.json
  → {"status": "DEGREE_ID_INPUTS_READY", "seed_count": 1}  EXIT=0
.claude/hooks/python.cmd python -B -m ..._benchmark --stage calibration \
  --input .../calibration-input.json --output .../calibration-results.json
  → {"status": "DEGREE_ID_CALIBRATION_PASS", "pass_count": 1, ...}  EXIT=0
.claude/hooks/python.cmd python -B -m ..._benchmark --stage generate-development --output .../development-input.json
  → {"status": "DEGREE_ID_INPUTS_READY", "seed_count": 16}  EXIT=0
.claude/hooks/python.cmd python -B -m ..._benchmark --stage development \
  --input .../development-input.json --output .../development-results.json
  → {"status": "DEGREE_ID_DEVELOPMENT_GO", "pass_count": 16, ...}  EXIT=0
.claude/hooks/python.cmd pytest tests/test_runtime_independent_stream_degree_id.py -p no:cacheprovider -q
  → 2 passed in 3.58s  EXIT=0
```

pytest basetemp는 CE 하네스 래퍼가 소유·부여한다 (호출자 `--basetemp`는 래퍼가 거부).

## 불변식 확인

- reality_stone/clarus: canonical 5계층 비변경, runtime.py 비수정 (동결 SHA 재확인), F1–F4 우회 없음, STDP 미사용. 신규 모듈은 TR30 하네스 층 전용.
- 계약 §4.7 STOP 조건: 후보 은행은 (seed, 생성기, 비query 셀)만의 함수 + 진리 carve-out — held-out 진리 중심화·예측/기준선 유도 없음. 영수증이 선행성을 기계 검증.
- 정직성: development 행 실패 시 재조정 없이 STOP으로 기록하는 구조 (`DEGREE_ID_STOP` 분기) — 본 실행에서는 발동하지 않았다.

## 산출물

- `artifacts/calibration-input.json`, `artifacts/calibration-results.json`
- `artifacts/development-input.json`, `artifacts/development-results.json` (fold별 e, d̂, s′_d, 기권, 은행 값·SHA·순서 카운터 영수증, 대조군 결과 전부 직렬화)
- `artifacts/source-freeze.json` (모듈·벤치·테스트·runtime·문서 SHA-256)

집계표·게이트 판정은 `31-validation.md`.

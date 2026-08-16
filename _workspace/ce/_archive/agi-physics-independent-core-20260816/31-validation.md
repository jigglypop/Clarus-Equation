# 물리 독립 AGI 코어 V0 검증 기록

Status: COMPLETE

## 1. 최종 focused test

Python bytecode와 pytest cache provider를 끄고 OS temp 아래 unique basetemp를 생성했으며 `finally`에서 exact containment를 확인한 뒤 제거했다. 전체 pytest, 전체 AGI bench, Phase A one-shot과 confirmation은 실행하지 않았다.

```text
python -B -m pytest tests/test_agi_lab_core.py -q -p no:cacheprovider --basetemp <unique OS temp>
................                                                         [100%]
16 passed in 0.44s
Exit code: 0
```

검사 범위는 다음과 같다.

- 격리 import와 root package 비오염;
- canonical immutable records, coercion/nonfinite/surrogate 거부;
- 비동형 XOR/SET family의 동일 protocol-only orchestrator 실행;
- paired hidden-world visible-history noninterference;
- forbidden proposal filtering과 abstention fallback;
- raw/forged/stale/cross-world/replay/terminal permit 경계;
- linear-history replay 거부와 old-snapshot deterministic fork;
- explicit genesis/episode boundary;
- secret-free public ledger와 full-state byte replay;
- 중간 event tamper detection 및 downstream rehash propagation;
- pure reducer input immutability;
- explicit memory/model/adapter/executor protocol 및 finite rollout;
- inconsistent genesis/executor/memory protocol output의 fail-closed 처리.

## 2. 정적 검사

```text
.venv\Scripts\ruff.exe check --no-cache <승인된 여섯 파일>
All checks passed!
Exit code: 0
```

```text
python -B <in-memory compile of approved six files>
syntax-ok 6
Exit code: 0
```

untracked 파일을 `git diff --check`만으로 놓치지 않도록 여섯 파일의 strict UTF-8 decode, trailing whitespace와 final newline을 직접 검사했다.

```text
whitespace-and-utf8-ok 6
package_residue=0 test_pyc=0 temp_residue=0
Exit code: 0
```

`git diff --check -- reality_stone/python/reality_stone/clarus/agi_lab tests/test_agi_lab_core.py`도 exit 0이었다. 이 명령은 신규 untracked content를 충분히 검사하지 못하므로 위 explicit scanner를 정본 증거로 함께 둔다.

## 3. 주장별 판정

| Claim | 구현 판정 | 증거 범위 |
|---|---|---|
| PIC-I1 | `[산출]` | protocol-only 동일 source와 opaque proxy에서 비동형 두 family 실행 |
| PIC-I2 | `[산출]` | post-genesis agent transition은 verified permit+pure reducer 경로만 통과 |
| PIC-I3 | `[산출]` | 같은 visible prefix에서 proposal·learner public-ledger bytes 동일 |
| PIC-I4 | `[산출]` | full canonical state replay와 public ledger byte equality, hash-chain propagation |
| PIC-I5 | `[산출]` | 금지 행동 차단 및 descendant linear history에서 nonce 재사용 거부 |
| PIC-H1--PIC-H5 | `[미완성]` | 성능·기하·SCC·언어·자기 가설 평가를 이번 V0에서 실행하지 않음 |

PIC-I1--PIC-I5의 `[산출]`은 이 유한 참조 구현의 mechanism certificate일 뿐 보편 물리독립 정리나 AGI 증거가 아니다. 특히 global anti-rollback과 side-effect transaction은 외부 durable boundary 없이는 닫히지 않았다.

## 4. 잔여물 및 실행 절제

새 package 아래 `__pycache__`/`.pyc`, test 전용 `.pyc`, `ce-agi-core-*` OS temp는 모두 0개다. 기존 사용자 cache나 과거 run residue는 삭제하지 않았다. 검증은 변경과 직접 연결된 단일 focused 파일만 실행했다.

## 5. 독립 검토와 CE build gate

독립 구현 리뷰는 최종 snapshot을 다시 읽고 focused test를 별도로 실행했다.

```text
Verdict: ACCEPT
Open P0: 0
Open P1: 0
Focused test: 16 passed in 0.32s
```

리뷰 원장은 `artifacts/post-implementation-review.md`이며 SHA-256은 `27096C659DE2F63D3760F38831704A2D853106C543C7352B5F32F94D0BC052F2`다.

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .codex/hooks/run.ps1 check _workspace/ce/agi-physics-independent-core-20260816 build
OK build
Exit code: 0
```

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .codex/hooks/run.ps1 check _workspace/ce/agi-physics-independent-core-20260816 final
OK final
Exit code: 0
```

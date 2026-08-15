# Phase A V1 구현 기록

Status: COMPLETE

## 1. 승인과 변경 경계

구현 전 명령은 다음과 같이 통과했다.

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .codex/hooks/run.ps1 check _workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816 gate
OK gate
```

`20-audit.md`가 승인한 아래 신규 네 파일과 `.gitignore` exact exception 한 줄만
제품 표면에 추가했다.

1. `reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py`
2. `tests/test_causal_recurrent_geometry_benchmark.py`
3. `experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`
4. `examples/agi/causal_recurrent_geometry_development_run.py`
5. `!experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`

`reality_stone.clarus.__init__`, 기존 runtime, SCC·metric 구현, 정본, default flag,
V15--V17 및 infinite-tail 자산은 수정하거나 import하지 않았다. runner와 test는
등록 source bytes를 한 번 읽어 SHA-256으로 묶고 그 동일 buffer를 `compile`/`exec`해
package initializer를 실행하지 않는다.

## 2. 구현 내용

production module은 NumPy만 사용하는 격리 benchmark다.

- generator-owned `GroundTruth`, learner-visible `TransitionBatch`, frozen `FitResult`,
  evaluator-owned scoring을 서로 다른 typed 표면으로 분리했다.
- R1은 문맥별 `A_z`와 공유 `B`를 하나의 `N x (Kn+m)` design으로 fit한다.
  R3는 pooled `A`와 공유 `B`인 필수 baseline이다.
- joint singular values, required/observed rank, 문맥별 state rank,
  residualized-input singular values/rank/tolerance를 모두 certificate에 남긴다.
- exact-edge 허용은 fit에 결박된 known identity, declared linear class, full rank,
  finite valid inputs의 conjunction이다. known mask, unknown mix, rank deficiency와
  cross-chart scoring은 fail closed한다.
- graph, train/held-out trajectory, intervention, train/evaluation noise, shuffle,
  bootstrap을 안정적 SHA-256 role namespace로 분리했다.
- common manifest `sigma`는 fit signature에 없고 두 arm의 scorer에서만 동일하게
  사용한다. 등록 식과 같은 graph-seed별 total scalar Gaussian NLL을 계산한다.
- nominal dof는 R1 `n(Kn+m)`, R3 `n(n+m)`이며 ridge effective dof도 기록한다.
- graph seed가 bootstrap 단위이고 frame은 통계단위가 아니다. PA-H1과 PA-H2는
  등록된 하나의 bootstrap graph-index matrix를 공유한다.
- runner는 canonical JSON, duplicate key/path/hash, traversal, source/self hash,
  dimensionless scale·shape, split과 custody boundary를 검사한다. manifest 검증 뒤
  evaluator보다 먼저 output을 `xb`로 예약하고, 실패 시 zero-byte placeholder와
  가능한 `.tmp`를 감사 증거로 남긴다.

## 3. red-team에서 닫은 열 가지 문제

one-shot 전에 다음 문제를 재현하고 수정·회귀 고정했다.

1. total NLL 대신 per-coordinate mean NLL을 사용한 endpoint 단위 오류;
2. float, bool, 문자열과 oversized uint context label의 손실 int64 변환;
3. manifest에 없는 `bootstrap_seed+1` 사용;
4. evaluator 종료 뒤 output을 예약하던 동시 one-shot race;
5. numeric string·`Decimal` config와 원 scalar type 보존;
6. identity fit과 unknown-mix batch의 cross-chart predict/NLL;
7. unknown-mix·rank-deficient fit의 coefficient-error claim fail-open;
8. outer result의 `confirmation_status` 누락;
9. `result == result`였던 tautological replay test;
10. STOP predicate iff와 equal-context R1/R3 동률 kill test 누락.

그 밖에 fit이 observation chart를 보존하도록 했고, train noise namespace를 trajectory와
분리했으며, ridge `0` rank-deficient effective dof의 `0/0`을 rank 값으로 닫았다.

## 4. split 오염과 rotation

초기 development 후보 `2001`--`2024` 중 일부를 focused test가 소형 config로
생성·채점한 사실을 발견했다. 전체 block을
`ABANDONED_PRE_REGISTRATION_TEST_CONTAMINATION`으로 폐기했으며 그 결과는 H1/H2
증거나 반증으로 사용하지 않는다. manifest config official development와 confirmation은
당시 실행하지 않았다.

rotation-2는 outcome-independent domain 문자열의 SHA-256 앞 7 hex에서 시작하는
연속 24개 규칙으로 정했다. raw block은 preregistration manifest에만 두었고 one-shot
전 repository nonuse scan은 manifest 외 occurrence `0`이었다. 정확한 입력·알고리즘과
초기 관찰 범위는 `artifacts/split-contamination-ledger.md`에 기록했다.

confirmation commitment에는 검증된 raw custody 또는 future reveal 증거가 없다.
따라서 manifest와 result는 이를 `reservation_only`, `custody_unverified`,
`not_executable_holdout`, `execution_authorized=false`, `reserved_unopened`으로만
표현한다. confirmation 함수, CLI mode, raw seed와 namespace 생성 경로는 없다.

## 5. 잠긴 provenance

one-shot에 사용한 SHA-256은 다음과 같다.

| artifact | SHA-256 |
|---|---|
| contract | `a77ef203c5a9614043b5f5223547ef50f214234bde43c68ac8dc70309644fe6e` |
| production module | `783b19a3d7e5e167203bfa49b8166b452bdd526729b87940b6033b25e3ebb8c8` |
| focused test | `f1870e1ab67c837dcea317138472de235c93d93c564af925a37ba68e4239a8aa` |
| development runner | `f311a1ae1ecabbfbb55ac966429f276d782f13629f29759ad685f0d93516bb7c` |
| manifest file bytes | `708a664bde65848f3abd748126dfa44e798d9fa333bacd79d535864e3bdf97b1` |
| canonical manifest self-hash | `78af635c524e05b35ba4a8b982f712d5174b99298bce1d94f827694c1c8733e2` |

required artifact hash 네 개와 manifest self-hash는 one-shot 직전·직후 모두
일치했다.

## 6. one-shot

등록 runner는 다음 명령으로 정확히 한 번 실행했다.

```text
.venv\Scripts\python.exe examples/agi/causal_recurrent_geometry_development_run.py experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
{"PA-H1":"GO","PA-H2":"GO","mode":"development","result_path":"_workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/artifacts/development-results.json"}
Exit code: 0
```

result SHA-256은
`7c4b9eb9ba08bed4cfc192262cc47c4a1cf56326526b3848cc6e4d5d89780df9`다.
result가 존재하므로 runner는 다시 실행하지 않는다.

# CE 멀티레포 이행 실행계획

작성: 2026-08-23. 상태: IN_PROGRESS (사용자 "전부 진행" 승인, 2026-08-23).

## 진행 기록

- P0-2: **결정 — ce-math 별도 레포를 만들지 않는다.** `docs/9_등호이전/` §1-4는 ce-qft 레포의 docs 서브트리로 동거하고, §5 브리지 계열(05, 05a-05o)만 코어 연결부로 취급한다. 근거: 독립 소비자가 없고 원장(등호이전_*)과 양방향 결합이라 분리 시 교차 계약만 늘어난다. 볼륨이 커지면 그때 승격.
- P0-3: **완료 — 멀티레포 루트 `C:\dev\ce\` 생성** (OneDrive 밖).
- P1(진행): gc 1차 실행 — 3개 아카이브(brain-synapse-functional-quotient-20260823, brain-synapse-riemannian-subspace-20260823, brainruntime-devivo-asi-measurement-20260823), STALE 36 확인. G1 정규화 완료: 배치 A 13/13, 배치 B 12/13 (+qc2 LEAVE-ACTIVE — 살아있는 run으로 판단, 무수정). 판정 문자열은 전부 `Verdict:`로 무손실 보존. G2 판단형 closure 8건·docs dangling 링크 정리 진행 중.
- P2(진행): P2-3(전역 재수출 제거)·P2-4(runtime_* 76개 → `clarus/experiments/` 격리) impl-engineer 진행 중. 사양 확정 근거: 패키지 본체의 runtime_* 참조는 `__init__.py:374` native_loops 재수출 1건뿐, 변형 간 상호 import 69건은 동반 이동으로 유지.
- **P0-1 판정 확정 (2026-08-23, run 완결·Gate PASS·check final OK)**: `_workspace/ce/_archive/pstar-br8-adjudication-20260823`. 결과 **UNRESOLVED** — 튜플 (0.6891, 0.2623, 0.0487)은 코어 체인 산출 아님(H-a 기각, struct/background 오차가 허용치의 ~75배) + 원장 §5 기록 관측 기준선도 아님(H-b 기각). 귀결: 튜플은 **ce-agi-runtime 소유의 출처 불명 동결 [공리]**로 이관, $\Omega$ 명명 제거·무우주론적 명명 고정, 값은 bit-for-bit 보존. BR-8 위반 아님. P2-1/P2-2 사양이 이에 따라 확정됨: `constants.py`의 비율은 cosmology_registry import 없이 런타임 소유 동결 모듈에서 온다.
- P1 완료 (2026-08-23): G1 정규화 25건, G2 사후 봉인 8건(4 COMPLETE + 4 ABANDONED), docs 부재 표기 54건 + 링크 수리 14건, gc로 총 37개 아카이브. 잔여 미완성 run은 살아있는 연구 run 1개(brain-synapse-functional-quotient-qc2-20260823)뿐 — 의도적 유지. P1-2 완료: 210건 아카이브 경로·개명 참조 수정, 잔여 dangling 1건(18_CodeMap.md의 agi-loop-engineering-20260811 — 원래부터 부재, 목록 보고).
- P2-3/P2-4 완료 (2026-08-23): runtime_* 76개 → `clarus/experiments/` (git mv), import 재작성 ~111줄(tests 40파일 포함), `__init__.py` 재수출 4블록(61심볼) 제거, surface 테스트 8파일 모듈 직접 검증으로 전환(97 passed). 검증: doctor PASS, import 무결성 OK, 실험·canonical 테스트 green (경합 실패 1건은 gc 아카이빙 경로 문제로 별도 수리 중).
- P2-5 완료 (2026-08-23): `examples/physics/evolution/` 4파일 → `examples/evolution/` (git mv), docs 참조 7파일 갱신, 잔여 참조 0. **F5(tests→examples 역방향 import 5건)는 이 단계에서 해소하지 않고 Phase 5 ce-cosmo 추출 시 게이트의 패키지 승격으로 이관** — 모노레포 안에서는 동작에 문제 없음.
- P2-6 완료 (2026-08-23): 혼합 원장 분할 — 신규 7파일(상수_코어_원장, 참조_{코어 13·양자 4·우주론GR 6·flavor 2}_정리_증명, 참조_{양자·GR}_보존_원장), 스텁 32개(원 번호·앵커 보존, 재번호 0), 분류 변경 0. P1-1 조항: 상수_우주론_원장 §3.4 `C-B-TUPLE-GAP-01` [미완성] 신설(3-튜플 사상 부재 명기 + "§3 유도 소비" 서술 금지). 인용 run 경로는 아카이브 경로로 정정함. ledger-writer가 byte diff 미실행을 명기했으므로 발행 전 main이 git diff로 이동·스텁·헤더·신설 소절 외 변경 0건을 확인할 것.
- P2-1/P2-2 완료 (2026-08-23): `cosmology_registry.py` → `core_registry.py`(13심볼) + `runtime_targets.py`(튜플, provenance UNRESOLVED 명기) + readout·호환 shim. `constants.py` import 절단, `runtime.py` docstring $\Omega$ 명명 제거. 값 bit-for-bit 대조 통과(표 6항목), 지정 테스트 34 passed. 부수 1건: 감사 tooling의 파일 경로 로드 때문에 sibling import에 fallback 로더 추가.
- 테스트 경로 수리 완료 (2026-08-23): `tests/_run_paths.py` 아카이브 인식 해석기 신설, 22개 테스트 파일 전환. 건너뜀(자기 해시 동결): experiments 4 + examples/agi 2 + v16/v17/v18b 벤치 3 — 사유는 에이전트 보고 기록.
- **무결성 사건과 해결 (2026-08-23)**: gc 이동이 sha-잠금 run(agi-world-memory-integration-v1-20260810)의 동결 live 경로를 깨 13개 테스트 실패 + revisions/ prereg 해시체인 36파일에 CRLF 드리프트 발견(원인 미상, OneDrive 유력 — F11 실증). 조치: (a) 이동 827파일 전수 원시 바이트 스캔 — 의도 편집 8건 외 드리프트는 위 36건뿐, HEAD blob으로 전량 복원, (b) **gc에 `.pin` 규약 신설**(main.rs — 동결 경로 run은 아카이브하지 않음), 해당 run 원위치+pin, (c) OneDrive stat 캐시로 인한 가짜 M은 `git update-index --really-refresh`로 해소. 결과: world-memory 계열 **28/28 passed**. main.rs 수정은 ce-harness에도 반영.
- Phase 3 진척 (2026-08-23): `C:\dev\ce\ce-harness` 레포 3커밋 — 공통 하네스 추출(90파일), `.gitattributes` LF 고정, templates 6종(CLAUDE.common + 도메인 rules 5) + `tests/test_mirror_identity.py`(명시 쌍 3, 의도적 차이 29쌍 주석 기록) + `.pin` gc 수정 동기화. 잔여: 플러그인 매니페스트 패키징, CI 배선.
- P4-1 완료: git-filter-repo 2.47.0 설치 (uv tool).
- **Phase 2 마감 검증과 2차 무결성 사건 (2026-08-23)**: full pytest 1차 — 2662 passed / 65 failed / 54 errors. HEAD 기준선 worktree(`C:\dev\ce\baseline`) 대조로 분류: 선재 실패 63+41(fusion 데이터 의존·v16/v17 퇴역 벤치·죽은 봉인 가족 등), 우리 델타 20건. 조치:
  - **전역 CRLF 드리프트 발견**: 추적 파일 2,989개 전수 원시 스캔 결과 847개가 워킹트리만 CRLF(blob은 LF) — 오늘이 아니라 과거 세션 누적(PS CRLF 함정), git status에는 autocrlf 정규화로 안 보임. 847개 전량 HEAD blob 바이트로 정규화 → dream-bridge byte-lock 6건 회복.
  - v18b run 원위치+`.pin`. 단 봉인이 `clarus/__init__.py` 바이트를 고정해 P2-3과 구조 충돌 → 봉인 집행 테스트 4건에 사유 명기 skip (run은 ABANDONED 퇴역). 나머지 9 passed.
  - metric_memory/prediction_guided 봉인 가족은 **HEAD 기준선에서도 실패하는 죽은 봉인**(5건 선재) — 복원 불가로 판정, P2-4 이동에 따른 추가 1건(g3_confirmation)은 죽은 가족의 델타로 정직 기록. 봉인 4파일은 experiments 이동 유지(임포트·동결 경로 기계 수리 재적용).
  - world-memory 계열 28/28 green 유지.
  - **3차 발견 — 혼합 EOL 봉인 규약**: CRLF 정규화가 rollout-bridge 가족(sparse_causal v4/v5/v7 체인 등) 신규 실패 20건을 유발 — 이 봉인들은 **CRLF 디스크 바이트 기준으로 sha256 봉인**돼 있었다 (dream-bridge는 반대로 LF 강제 — 가족마다 규약이 다름). 조치: 리포 전체 64-hex 봉인 해시 수집 → CRLF 해시가 인용되고 LF 해시는 인용되지 않는 파일 33개(브리지 아티팩트 3, quantum.py, test_runtime_contracts.py, lib.rs, 외부 데이터셋 .m 28)만 명시 목록 기반 CRLF 복원, AMBIGUOUS 6개는 무수정. rollout 브리지 전 회복(70+21 passed).
  - **레포 분리 필수 반영 (정정)**: 모노레포 `.gitattributes`는 이미 봉인 가족별 명시 eol 규약을 갖고 있다(sparse_causal v4/v5/v7 아티팩트는 `eol=crlf`, dream/world-memory 체인은 `eol=lf`). 이번 실패는 전역 정규화가 이 의도된 규약을 덮은 것이며 표적 복원으로 정상화됨. **분리 시 각 레포가 자기 봉인 가족의 attributes 항목을 반드시 이관해야 한다** — 항목 누락 시 fresh clone에서 봉인 즉시 파손. 미지정 파일은 core.autocrlf에 좌우되므로, 분리 레포에는 전역 기본(`* text=auto eol=lf`) + 가족별 예외를 함께 넣는 것을 표준으로 한다.
  - q0_manifest_gate 9건은 부재 데이터(benchmarks/q0_minimal_abelian_higgs_v1.json — HEAD에도 없음) 원인의 선재 결손으로 확인.
- 범위 밖 변경 감사 (2026-08-23): worktree에 나타난 루트 잡파일 삭제(`0`, `0$`, `8`, `E`, `death`, `length-biased`, `w`, `zero`)는 어느 에이전트에도 지시되지 않은 변경이라 git checkout으로 전량 복원함. `.active-run` 삭제만 gc 정상 동작으로 유지. 이 잡파일들의 정리 여부는 사용자 결정 사항으로 보류.
목표: 모노레포를 독립 구동 가능한 레포 9개 + 우산 1개로 분리한다. 각 레포는 단독 clone에서 테스트가 돌고, 교차 결합은 버전 태그와 발행 계약 파일로만 흐른다.

## 0. 검증된 전제 (2026-08-23 실측)

| # | 사실 | 검증 방법 | 상태 |
|---|---|---|---|
| F1 | `reality_stone/python/reality_stone/clarus/constants.py:11`이 `cosmology_registry`에서 `LEGACY_DELTA_5DP_V1`, `LEGACY_ROUNDED_RUNTIME_V1`을 import하고 `ACTIVE_RATIO/STRUCT_RATIO/BACKGROUND_RATIO`를 export | 파일 직독 | 확인 |
| F2 | 위 비율을 `runtime.py`(13회)·`agent.py`(3회)·`stdp.py`(5회)·`__init__.py`(7회)가 사용 — 우주론을 떼면 뇌 런타임 import 실패 | grep 카운트 | 확인 |
| F3 | `clarus/runtime_*.py`는 **76개** (run당 본체+benchmark 쌍 누적). 패키지 볼륨 폭주의 주원인 | `ls \| wc -l` | 확인 (조사 보고의 52개는 과소집계) |
| F4 | `clarus/__init__.py:181,285,379`가 `universe_life_kernel`·`clarus_field`·`reality`를 top-level 지연 re-export | grep | 확인 |
| F5 | `tests/test_cosmology_closure_gate.py:4-24`가 `examples.physics.*` 5개 모듈을 라이브러리처럼 import | 파일 직독 | 확인 |
| F6 | `docs/검증_원장/상수_우주론_원장.md`: §2 고정점 코어(L21-96, 공통 커널) / §3부터 우주론 전용(L98-) — 절단선 L97 | 헤더 grep | 확인 |
| F7 | `docs/검증_원장/참조_핵심_정리_증명.md` 정리 25개 (공통 13 + 양자 4 + 우주론/GR 6 + flavor 2 혼재) | 헤더 카운트 | 확인 |
| F8 | 40-final 없는 STALE run 6개: `agi-frontier-comparison-20260815`, `agi-v18b-learned-delayed-credit-20260814`, `brain-mechanism-alternative-routes-20260819`, `brainruntime-weight-metric-dynamics-intervention-20260819`, `brainruntime-self-selecting-deformation-20260820`, `connectome-graph-replay-20260818` | 40-* 존재 검사 | 확인 (전부 0개) |
| F9 | `examples/physics/brain_equation_integrated_gate.py` 부재인데 `docs/6_뇌/02_관측정의.md:166,171` 등이 참조 (dangling). `examples/physics/evolution/`에 생명 게이트 4파일 잔존 | ls + grep | 확인 |
| F10 | `git filter-repo` 미설치. uv 0.11.9 사용 가능 → `uv tool install git-filter-repo`로 확보 | 명령 실행 | 확인 |
| F11 | 리포 루트가 OneDrive 동기화 폴더 내부 (`...\OneDrive\Desktop\Clarus-Equation`) | 경로 | 확인 |
| F12 | 물리 run 활성 0개 (전부 `_archive/`), 활성 37개는 전부 뇌/AGI 계열 → 물리부터 뜯으면 연구 중단 없음 | ls | 확인 |

조사 보고 인용이나 미실측 수치는 각 단계 착수 시 재확인한다. 실측 없이 이 표를 확장하지 않는다.

## 1. 목표 레포 토폴로지

| 레포 | 소유 경로 (현 모노레포 기준 manifest 초안) | 의존 |
|---|---|---|
| `ce-harness` | `.claude/` `.codex/`(미러 포함) hooks 바이너리, 공통 스킬·역할 카드 4종, `empirical_calibration_loop.md`, check-large-data 게이트 | 없음 |
| `ce-core` | `clarus/{core_axioms,bootstrap_solver,dimensionless,dimensionless_checker}.py`, 분할 후 `core_registry.py`, `docs/1_강의/`, `docs/axium.md`, `docs/상수.md`(§2·§4), `docs/코어_독자_가이드.md`, 상수_우주론_원장 §2 분할분, 참조_핵심_정리_증명 공통 13정리, 코어 테스트(`test_core_axioms`, `test_bootstrap_*`, `test_dimensionless`, `test_multispace_bootstrap`) | 없음 |
| `reality-stone` | `reality_stone/src/`(Rust), 기본 바인딩. CE 이론 코드 제외 | (선택) ce-core |
| `ce-cosmo` | 분할 후 `cosmology_readout.py`, `examples/physics/{cosmology*,hubble_tension,ce_residual_forward_model,primordial_spectrum_readout_gate,cosmological_constant_holographic_gate,density_bridge_variational_audit,xi_derivation,cosmology_kernel}.py`, 우주론 docs(3_상수 4편, 5_유도 04·07, 2_경로적분 05·12), 우주론 원장 3건 + 상수_우주론_원장 §3-6, 우주론 테스트 | ce-core |
| `ce-qft` | `clarus/{quantum,quantum_jump_bridge,euclidean_correlator_certificate,renormalized_pole_certificate,ce_two_point_vertex_certificate,a1_q0_action_bridge,spacelike_marginal_gate,portal_one_loop_control,q0_manifest_gate,multispace_bootstrap}.py`, 해당 examples 게이트, `docs/경로적분.md` §2-5, 2_경로적분 해당 장, 등호이전 06 계열, 4_공학 02·03, 참조_이론물리_보존_원장 §1-5, 양자 테스트 15개 | ce-core |
| `ce-lab` | 리만(`ce_{euler,zeta,mra,riemann_attn,ffn,ops}.py` + `docs/8_리만/` + 테스트 5), 핵융합(fusion 50파일), GR/exotic(14모듈 + 보존 원장 §6 + 테스트 14) — 트랙별 서브패키지 | ce-core |
| `ce-brain-bio` | `docs/6_뇌/` 전체, `.codex/harnesses/{brain_evidence_ladder,real_brain_equation_discovery_loop}.md`(정본 이관), `_workspace/ce/brain-algorithm-route-ledger.md` + 프레임 v2/v3 + prereg, 실데이터 run(dandi/randi/ibl/devivo), 데이터 매니페스트 | ce-core |
| `ce-agi-runtime` | `clarus/{runtime,engine,sleep,stdp,agent,...}.py`(Layer A-F), `runtime_*.py` 76개→`experiments/`로 격리, `docs/7_AGI/` 전체, `curvature_backend_parity.md`, `brainruntime-*`·`agi-*` run, 계층 테스트 + `test_runtime_*` | ce-core, reality-stone |
| `clarus-agent-guard` | `clarus-agent-guard/` 그대로 | 없음 |
| `ce-meta` | 레포 매니페스트(URL+pin 태그), `interfaces.md`(교차 계약 원장, BR-8류 도메인 간 금지 조항의 정본), 통합 CI, 크로스도메인 run | 전 레포 태그 |
| (보류) `ce-math` | `docs/9_등호이전/` §1-4 + 등호이전 원장 5건. Phase 0 결정에 따라 독립 또는 ce-qft 동거 | 없음 |

레포 경계 원칙: 발행-소비 계약을 쓸 가치가 있는 경계에만 레포를 세운다. 그 이하는 레포 내 디렉토리.

## 2. 교차 계약 (코드 import 금지, 계약 파일만)

| 계약 | 발행 → 소비 | 형식 |
|---|---|---|
| 고정점 상수·판본 (`CE_CORE_EXACT_V1` 등) | ce-core → 전체 | 패키지 API + semver 태그. 소비 측 값 복사 금지 (CI grep 게이트) |
| supported 현상 목록 + `BIO_EVIDENCE_Lx` | ce-brain-bio → ce-agi-runtime | `supported_phenomena.json` (현상 ID·사다리 위치·출처·판정일), 릴리스 태그 발행, 소비 측 pin. agi-rt CI가 L4 미만 현상에 뇌 주장 문구 사용을 lint로 차단 |
| p* 타깃 $(\Omega_\Lambda,\Omega_{DM},\Omega_b)$ | Phase 0 판정에 따름 | (a) ce-core 산출로 승격(명시 공리 등재) 또는 (b) 런타임 타깃 교체·이식 폐기 |
| `docs/7_AGI/13_Verification.md` → `quantum.py` 참조 1건 | ce-qft → ce-agi-runtime | 유지 여부를 Phase 2에서 판정, 유지 시 계약 행 등재 |

## 3. 단계별 실행계획

각 단계는 독립 커밋 단위이며, 완료 조건의 검증 명령이 green이어야 다음 단계로 간다. 실행하지 않은 검증은 실행하지 않았다고 쓴다.

### Phase 0 — 강제 결정 (사용자 판정 필요, 코드 변경 없음)

- P0-1. **p*/BR-8 충돌 판정 run 1건** (/ce-research): 우주론 비율의 런타임 타깃 사용을 (a) 코어 산출로 승격할지 (b) 폐기할지. 프레임 v3 §6 BR-8("우주론↔뇌 상수 이식 금지")과 F1·F2의 현행 코드가 충돌 중.
- P0-2. ce-math(등호이전) 독립 여부 결정.
- P0-3. 멀티레포 루트 위치 결정 (권고: OneDrive 밖 `C:\dev\ce\`, F11).
- 완료 조건: 판정이 run 40-final과 이 문서 §2에 기록됨.

### Phase 1 — 모노레포 위생 (분리와 무관하게 가치 있음)

- P1-1. STALE run 6개(F8) closure: 원장이 판정 근거로 인용하는 3개는 최소 numbered audit로 봉인, 나머지는 ABANDONED 근거 기록 후 `run.sh gc`.
- P1-2. 뇌 원장의 `_archive` 이동 run 링크 일괄 수정 (원장이 인용하는 43개 경로 대조).
- P1-3. `40-final.md` vs `40-final-report.md` 명명 통일.
- P1-4. dangling 링크 정리(F9): `examples/physics/brain_*` 7종 참조를 실경로로 수정하거나 부재를 문서에 명기. `경로적분_전체_진리값_감사.md` L91-104 깨진 마크다운 링크 수리.
- 완료 조건: `sh .claude/hooks/run.sh gc _workspace/ce`가 미완성 run 0 보고. 링크 검사 스크립트(신규, artifacts성 스팟 스크립트) 통과.

### Phase 2 — 모노레포 내 결합 절단 (전체 공수의 ~절반)

- P2-1. `cosmology_registry.py` → `core_registry.py`(CoreChain·판본) + `cosmology_readout.py`(밀도 사상) 분할. `constants.py:11` import를 `core_registry`로 교체(F1). 호환 shim은 1릴리스만 유지.
- P2-2. Phase 0 판정 반영: p* 타깃 경로 교체(F2 대상 4파일).
- P2-3. `__init__.py` top-level 재수출 제거(F4).
- P2-4. `runtime_*.py` 76개(F3)를 패키지 밖 `experiments/`로 이동, 대응 `test_runtime_*`의 import 경로 수정.
- P2-5. `examples/physics/evolution/` 생명 게이트를 뇌/진화 소유 경로로 이동(F9). tests→examples 역방향 import 5건(F5)은 해당 게이트를 정식 모듈로 승격해 해소.
- P2-6. 혼합 원장 절 분할: 상수_우주론_원장 §2/§3-6(F6), 참조_핵심_정리_증명 25정리(F7), 참조_이론물리_보존_원장 §1-5/§6, 검증_원장의 뇌/AGI 3건 분리.
- 완료 조건(단계별 최소 검증): P2-1·P2-2 후 `python.cmd pytest tests/test_cosmology_registry.py tests/test_core_axioms.py tests/test_layer_a.py` 등 직접 연결 테스트만. P2-4 후 `test_runtime_*` 중 임의 2개. 전체 pytest는 Phase 2 완료 시 1회 (사용자 승인 하에 full).

### Phase 3 — ce-harness 플러그인화

- P3-1. `.claude`/`.codex` 공통부(스킬·역할 카드·hooks 바이너리·공통 규율)를 ce-harness 레포로 추출, Claude Code 플러그인으로 패키징.
- P3-2. 미러 동일성 테스트를 ce-harness CI로 이관. 도메인 전용 하네스는 소유 레포로 (evidence ladder → brain-bio, parity → agi-runtime).
- P3-3. 각 레포용 얇은 `CLAUDE.md` 템플릿 + 도메인 rules 파일 작성.
- 완료 조건: 신규 빈 레포에 플러그인 설치 후 `run.sh init/status/check`가 동작.

### Phase 4 — ce-core·reality-stone 추출

- P4-1. `uv tool install git-filter-repo`(F10). 모노레포 fresh clone에서 `git filter-repo --path <manifest 경로들>`로 이력 추출.
- P4-2. ce-core에 `pyproject.toml`(패키지명 `clarus-core`) + semver `v1.0.0` 태그. CHANGELOG에 이론 판본(`CE_CORE_EXACT_V1`) ↔ 패키지 버전 매핑.
- P4-3. reality-stone에서 CE 이론 코드 제거, 순수 엔진화.
- P4-4. 모노레포 소비 코드를 `uv add "clarus-core @ git+<url>@v1.0.0"` pin으로 전환.
- 완료 조건: ce-core 단독 clone에서 코어 테스트 green. 모노레포에서 pin 전환 후 동일 테스트 green.

### Phase 5 — 물리 분리 (리허설 → 본선)

- P5-1. **ce-lab을 1호 분리**(리만 트랙이 결합 최소 — 리허설). 규약(manifest·원장·CI·플러그인) 검증.
- P5-2. ce-qft, ce-cosmo 순차 분리. 각 레포 첫 커밋 직후 CI 구축 → green 확인 후 다음 레포 (F12: 물리는 활성 run 0이므로 연구 중단 없음).
- 완료 조건: 각 레포 단독 clone에서 자체 테스트 green + 무차원 게이트 통과.

### Phase 6 — 뇌/AGI 분리 (활성 트랙, 마지막)

- P6-1. ce-brain-bio 분리 + `supported_phenomena.json` 첫 발행.
- P6-2. ce-agi-runtime 분리, 계약 pin. 뇌 주장 lint를 CI에 장착.
- P6-3. 활성 run 37개의 이관: 진행 중 run은 freeze 창(1-2일) 안에 stage 파일만 이동, 대용량 artifact는 매니페스트화.
- 완료 조건: 두 레포 단독 green + agi-rt에서 layer 테스트·환원 조건 회귀 green + 뇌 주장 lint 동작 확인.

### Phase 7 — 우산과 동결

- P7-1. ce-meta: 매니페스트 + interfaces.md + 통합 CI(전 레포 최신 태그 조합 회귀, 주기 실행).
- P7-2. 모노레포 → `ce-monorepo-archive` read-only 동결, 태그 `split-baseline`. **삭제 금지** (prereg 해시체인·faithbench zip 유일본 보존).
- P7-3. 각 레포 README에 이력 출처(`ce-monorepo-archive@<SHA>`) 명기.
- 완료 조건: 통합 CI 1회 green. 아카이브 레포 push 후 remote SHA 대조.

## 4. 불변 규율 (이행 중에도 적용)

- 레포를 뜯는 동안 해당 경로는 모노레포에서 freeze. "반쯤 뜯긴 상태"로 세션을 넘기지 않는다.
- commit/push는 root/main 세션만, 승인 경로 manifest만 stage, `check-large-data.cmd --commit/--push` preflight 필수, force push 금지.
- run 아카이브 대용량(4.2GB급)은 어떤 신규 레포에도 커밋하지 않는다. stage 파일 8종 + 판정 artifact만.
- 실패를 통과로 쓰지 않는다. 생략한 검증은 생략했다고 기록한다.

# Scoped implementation and canonical document assembly

Status: COMPLETE

## Research artifact

`artifacts/verify_self_measurement_time.py`는 다음의 유한차원 certificate를
구현한다.

1. fixed-axis partial dephasing channel의 Choi positivity와 trace preservation.
2. $\eta$ composition과 $\theta=-\ln(1-\eta)$의 additive law.
3. $\theta_*=1.5$의 $N=1,2,5,100$ weak partition channel equality.
4. trace-distance self-nonidentity increment, metric speed, finite path length와
   logarithmic residual clock.
5. $N=1,2,5,100,1000$ refinement 아래 path length invariance.
6. bounded $C_{\rm self}$와 선택한 $A\ne0$ 경로의 cost-length relation.
7. stationary, periodic-unitary, noncommuting-order와 non-Markov-recoherence
   counterexamples.

## Canonical documents

원장과 narrative의 작성 소유권을 분리하여 staging mirror에서 원장을 먼저
동결한 뒤 narrative가 이를 읽기 전용으로 사용했다. combined-doc 독립 감사가
P0/P1 없음과 Gate PASS를 낸 뒤 다음 세 파일만 canonical worktree에 해시
precondition 하에서 복사했다.

| path relative to `C:\dev\ce\ce-cosmo` | SHA-256 after copy |
|---|---|
| `docs/검증_원장/상수_우주론_원장.md` | `72F96BA4FB6A70CC420F333CBB58DA3398A299763500AACEF547862717617C7F` |
| `docs/5_유도/00_선택과_접힘.md` | `A68F4E4BE838F1D35214F6C1B10C9EC88B3A7A39C39550C1399C79AB110708DD` |
| `docs/5_유도/04_Dark_Energy_Derivation.md` | `733962A7C3F4F78F7CD382D225D48E914D3B98E7A37149381675CB49C87AD026` |

원장은 `MEAS-THETA-*`, `MEAS-SELF-*`, `OPP-SELF-*`와 완전 반례 경계를
등록했다. `00_선택과_접힘.md`는 one-way 0D boundary, informative record와
self-measurement를 type-safe하게 분리하고 자기비동일성 흐름을 단계적으로
유도한다. `04_Dark_Energy_Derivation.md`는 $C_{\rm self}$를 bounded
dimensionless readout으로만 추가하고 energy/stress/dark abundance bridge를
미완성으로 유지한다.

## Git handoff

- Repository root: `C:\dev\ce\ce-cosmo`
- Branch: `main`
- HEAD: `f78accbdd075454437e57ff39b6b6b0154088c10`
- Upstream: none
- Remote tip: not applicable; no remote is configured
- This run's canonical changed-path manifest: the three paths in the table above
- Staging/commit/push: not performed

기존 user dirt인 `README.md`, cosmology gate/registry source와 tests,
`benchmarks/cosmology/desi_dr2/`는 건드리지 않았다. 세 문서도 선행 연구에서
이미 modified 또는 untracked였으므로, 이 run은 검증된 staging bytes를 정확히
덮어쓴 범위만 소유한다.


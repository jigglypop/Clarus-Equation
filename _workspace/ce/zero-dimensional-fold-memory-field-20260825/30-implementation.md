# 30-implementation — scoped artifact implementation

Status: COMPLETE

구현 범위는 연구 run 안의 재현 검증기 `artifacts/verify_fold_memory_field.py`와 staging 정본 문서다. production dynamics 또는 Einstein--Boltzmann 코드는 추가하지 않았다.

검증기는 persistent carrier의 8-node ring witness, zero/positive branch, full Jacobian spectrum, delayed-stability sufficient bound, exact dimension balance와 seed 제약을 검사한다. 보조 경로에는 causal finite-width kernel witness, nonlinear mean-field roots, Poisson extinction control, UV scaling과 Derrick certificate를 포함한다. 중심 carrier gain $\mathcal B$와 보조 Hawkes reproduction number $\mathcal R$은 서로 다른 항목으로 유지한다.

정본 반영 대상은 staging의 `docs/검증_원장/상수_우주론_원장.md`, `docs/5_유도/00_선택과_접힘.md`, `docs/5_유도/04_Dark_Energy_Derivation.md`다. 원장은 주장 지위를 먼저 고정하고, 유도 문서는 그 원장을 읽기 전용으로 사용한다. 암흑 동일성은 구현하거나 기본값으로 넣지 않는다.

combined fold/opportunity/measurement 감사가 끝난 뒤 위 세 staging 문서를 canonical
`C:\dev\ce\ce-cosmo`의 같은 경로에만 복사했다. 복사 전 HEAD와 세 target의 기존
hash를 확인했고, 복사 뒤 다음 SHA-256이 staging과 canonical에서 일치했다.

| path | SHA-256 |
|---|---|
| `docs/검증_원장/상수_우주론_원장.md` | `E5ABF1CE09F255BF387C5FEB55014FFF4A126C1D40278E6B37F4C983D218D858` |
| `docs/5_유도/00_선택과_접힘.md` | `3AF2B52793DA66778FBD4FCA5846B6A9AA4DEF67FF82CC1BF72FD8E7AC4B771F` |
| `docs/5_유도/04_Dark_Energy_Derivation.md` | `CC2A45A8A6DCAEF640508923141D50E852ACE46AA6F596327D7F8F5CAC13D2B6` |

Git staging, commit와 publish는 수행하지 않았다.

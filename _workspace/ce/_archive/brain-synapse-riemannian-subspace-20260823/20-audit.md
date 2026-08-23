# 20-audit — BA-SRM1 독립 상태 감사

Status: COMPLETE

Gate: PASS

Audit snapshot: Revision 1의 `00-contract.md`, `10-sources.md`, `11-math.md`,
`12-routes.md`, download/schema receipts. 첫 감사의 P1 두 건과 P2 한 건을 문서
수정한 뒤 동일 auditor가 재감사했다. response outcome은 이 Gate 전 미접촉이었다.

## Claim 지위

| Claim ID | 계약 지위 | 감사 지위 | 근거/경계 |
|---|---|---|---|
| `BA-SRM1-C1` | [정의/관측 입력] | 지위 정합 | typed factor registry가 단일 $W$와 직접 관측·latent 항을 분리한다. |
| `BA-SRM1-C2` | [정의/측정모형] | 지위 정합 | strict $z\in\mathbb R^4$, scalar-summary $y\in\mathbb R^4$와 source schema가 일치한다. |
| `BA-SRM1-C3` | [조건부 정리] | 지위 정합 | $R\succ0$, $\operatorname{rank}J=4$에서만 $J^TR^{-1}J\succ0$이다. 경험적 rank는 미검사다. |
| `BA-SRM1-C4` | [예측] | 지위 정합 | split, M0/M1, thresholds, query attachment와 confirmation 규칙이 결과 전에 고정됐다. |
| `BA-SRM1-C5` | [미완성/금지 경계] | 지위 정합 | row-level event, conductance, $Npq$, 장기 가소성, delay-distance, 기억/AGI 승격을 막는다. |

## 최초 감사 수정 내역

1. `11-math.md`의 역사적 $g-I$ 반례식에 `REJECTED; NOT ACTIVE`를 붙여
   active $g_{\rm ref}$ 식과 분리했다.
2. $a_{6:8}$와 $a_{9:12}$가 각각 pulse 구간의 scalar 중앙값임을 명시했다.
   따라서 $y\in\mathbb R^4$, $J\in\mathbb R^{4\times4}$다.
3. `BA-SRM1-C1..C5`와 lane mapping을 추가했다.

## Gate 근거

- 열린 P0/P1: 0.
- 공식 input: `VERIFIED_INPUT_ONLY`, 176,771,072 bytes,
  SHA-256 `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53`,
  SQLite integrity `ok`.
- schema: `PASS_SCHEMA_WITH_ROW_LEVEL_EVENT_LIMIT`.
- strict support: 979 pairs / 512 slice groups; primary mouse V1 ex 246/160,
  in 343/199.
- outcome-contact receipts: `outcome_contact=false`,
  `outcome_values_reported=false`.
- 수학 P0: 잘못된 $I$ reference, query graph 미정의, adjacency gauge 문제가
  모두 Revision 1에서 닫혔다.
- 선행 봉인: curvature-memory, directed delay→Riemannian, ASI→strength,
  conductance/$Npq$ 직접관측 승격을 재도입하지 않았다.

## 남은 경험적 실패 가능성

Gate PASS는 성공 결과가 아니다. 구현 후 $R$ condition, $J$ rank,
bootstrap stability, graph connectivity, gauge invariance 또는 best-control
$\Delta\mathrm{ELPD}$ 중 하나라도 실패하면 C3/C4를 승격하지 않는다.
development 실패 시 confirmation은 열지 않는다.

Counts: claims inspected 5; conditional theorem 1; definitions/measurement claims 2;
prediction 1; unfinished/prohibition boundary 1; hidden axioms opened 0;
complete counterexample parent claims retained 0; open P0 0; open P1 0.

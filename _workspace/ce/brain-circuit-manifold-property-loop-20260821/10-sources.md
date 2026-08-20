# Sources

Status: COMPLETE

## Frozen inherited sources

이 light follow-up은 predecessor의 source lane을 재사용한다. 새 empirical dataset,
neural response 또는 anatomy asset을 열지 않았다.

| source | role in this run | claim ceiling |
|---|---|---|
| MIT 18.965, *Geometry of Manifolds*, predecessor `10-sources.md` | pullback `J^T G J`와 immersion/full-rank 조건 | 특정 neural dynamics의 타당성을 보이지 않음 |
| Mannheim, *Riemannian Geometry—Metrics and Connections*, predecessor `10-sources.md` | 좌표변환 아래 metric congruence | componentwise activation 모델 자체의 arbitrary-coordinate invariance를 뜻하지 않음 |
| Klamka, *Controllability and Minimum Energy Control* (2018), predecessor `10-sources.md` | fixed LTV Gramian minimum-energy identity | nonlinear global reachability 또는 intrinsic brain metric을 뜻하지 않음 |
| Sun & Motter, PRL 110, 208701 (2013), predecessor `10-sources.md` | Gramian ill-conditioning 경계 | numerical rank를 exact theorem으로 승격하지 않음 |
| predecessor A6 contract/math/audit/validation/final hashes | 이번 property family의 유일한 claim parent | `MATH_PASS / EMPIRICAL_UNTESTED` |

## Source admission

- randomized fixture는 source가 아니라 반례 탐색용 deterministic apparatus다.
- seed는 biological replicate가 아니다.
- actual BrainRuntime code는 smooth A6의 empirical source가 아니며, 구현 경계 비교에만
  읽기 전용으로 사용했다.
- actual cortical folding, neuron identity, edge parent receipt와 longitudinal anatomy는
  계속 `BLOCKED_INPUT`이다.

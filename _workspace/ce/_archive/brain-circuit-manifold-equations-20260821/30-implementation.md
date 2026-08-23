# 구현

Status: COMPLETE

이번 run의 구현 범위는 empirical brain runtime 변경이 아니라 deterministic math witness로 제한했다.

## 추가 산출물

- `artifacts/a6_math_witness.py`
- SHA-256: `57e67e6da3520221e1dcec05e6c3af3f51c3dfd4906198554f80698994a6dfd7`

## 검증하는 항목

1. activity-dependent $p(a)$에서 frozen-$p$ Jacobian이 실제 derivative와 다르다는 P0 증인,
2. anisotropic passive pullback의 principal stretch와 volume,
3. rank-deficient pullback이 Riemann metric이 아니라는 증인,
4. nonnormal two-tick Gramian과 minimum energy,
5. $\dot g_\Gamma$ total chain rule과 central finite difference의 일치,
6. full-rank reachability-energy derivative와 central finite difference의 일치.

실제 response bytes, neural identities, edge receipts, confirmation cohort는 열지 않았다. 따라서 이 구현은 A6의 수학 무결성만 검사하며 생물학적 효과를 측정하지 않는다.

## 별도 코드 갱신

`reality_stone/python/reality_stone/clarus/dimensionless_checker.py`에 A6의 directional stretch, generalized metric ratio, metric-volume log ratio, reachability-energy log ratio를 등록했다. `tests/test_dimensionless.py`에는 네 기호의 무차원 gate를 추가했다.

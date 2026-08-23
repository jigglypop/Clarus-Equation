# Stable equation audit

Status: COMPLETE

Gate: PASS

## Snapshot

감사 대상은 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`, `artifacts/dimensionless-audit.md`의 안정된 판본이다. 실제 신경자료, 새 synthetic seed, model fit은 실행하지 않았다.

## 독립 판정

| Lane | 판정 | 핵심 확인 |
|---|---|---|
| Fisher/geometry math | PASS | conditional Gaussian Fisher의 바깥 $h$ 평균, regularity, covariant reference, chart law, Karcher measure가 닫힘 |
| routing/identifiability | PASS | static 대 interaction nesting, nat/sample 정규화, common-input·mediation 반례, negative-lag 진단 경계가 닫힘 |
| closure/status | PASS | $C^{-1}$ 보편 metric 부모 주장을 제거하고 살아 있는 affine-chart 경험식만 보존함 |
| dimensionless | PASS | $ds^2$, generalized spectrum/log, likelihood ratio, $R$와 $\Delta R$의 무차원성이 닫힘 |

## 수정 이력

첫 stable audit에서 Gaussian 특수형이 $I_{ab}(z,h,c)$의 바깥 $\mathbb E_{h\mid z,c}$를 빠뜨린 P0를 발견했다. 식을

$$
G_{ab}(z,c)=\mathbb E_{h\mid z,c}[I_{ab}(z,h,c)]
$$

로 고쳤고 재감사에서 PASS했다. 함께 고친 P1은 Fisher 정칙성, $\bar G$의 공통 calibration measure와 Karcher 정의, context route nesting, ELPD의 per-sample 정규화, negative-lag의 diagnostic 지위다.

## 형식 폐쇄

- **[정의]+[정리]:** conditional output-Fisher tensor, tensor transform, dimensionless line element, Gaussian expansion.
- **[공리: 모델 선택]:** calibration-only $G_{\rm ref}$와 $\lambda$, common chart/transport, common $q_j(z)$.
- **[정의: 경험 추정량]:** held-out conditional predictive routing과 context interaction score.
- **[경험식 후보]:** 고정 affine selectivity chart의 $C^{-1}$.
- **[미완성]:** $W\to G\to x$, $G\to R$, curvature/geodesic dynamics, causal routing.
- **[삭제]:** $C^{-1}$의 보편 local-tensor 지위, SCC=기억/의식, simulator=생물학적 뇌 알고리즘 동일시.

## 판정 경계

PASS는 식의 수학적·형식적 정합성만 뜻한다. 실제 뇌에서 $G$ 또는 $R$이 문맥에 따라 변한다는 경험 결과는 아직 없다. 다음 허용 작업은 canonical 원고 갱신과 실제자료 schema/time-alignment 적격성 판정뿐이다.

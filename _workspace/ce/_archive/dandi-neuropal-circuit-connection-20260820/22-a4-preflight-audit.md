# A4 구현 전 감사

Status: PASS_TO_A4_DEVELOPMENT

Gate: GO

- contract SHA-256: `6096064740c8d13070854fdc35ad0b2f81b0e955510113f1d97640f7201c61da`
- source manifest SHA-256: `ef2c343eccb6bdbacf5018ceabb694c097919367fc9d9b65da986bf18097ea49`
- implementation SHA-256: `e9f8bb1caa0c8703940f15827f67fd9b256670b41e13b4246f331767c61d0fa1`
- source-only validation: `PASS_A4_SELF_TEST`; Python compile pass; diff-check pass.

수학 감사에서 뉴런별 절편, complete-case missingness, $1.4826\,MAD$ scale, exact time-shift adverse operator, K/B/T guard, $-L_tz$ 부호, $\beta_g\ge0$, PSD Laplacian, range-read provenance가 계약과 구현에 일치함을 확인했다. 개발에 허용된 response는 `source-manifest-v3.json`의 앞 세 자산뿐이다. confirmation 다섯 자산은 development gate 전까지 봉인한다.

허용 주장 상한은 `OBSERVATIONAL_STATE_DEPENDENT_GRAPH_PREDICTOR`다. symmetric $\alpha$는 parent, 방향, cycle, synapse, 인과 또는 물리적 manifold 변형을 식별하지 않는다.


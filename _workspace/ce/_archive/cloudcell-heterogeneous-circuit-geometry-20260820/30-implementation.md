# 구현 기록

Status: COMPLETE

`artifacts/run_cloudcell_geometry.py`는 frozen input audit에서 여섯 recording을 schema로 선택하고 다음을 수행한다.

- calibration-only red-channel nuisance regression;
- 뉴런별 median/MAD threshold와 결측 baseline imputation;
- fit-block ridge predictive responsibility와 회로 강도;
- 6-NN graph, directed SCC cycle support, coercive graph field, bounded positive routing gain;
- A1 post-fit gain과 A2 anisotropic ridge;
- shuffle, matched noncycle, time shift, common-threshold, GFP controls;
- validation/confirmation stage 분리와 exact sign-flip aggregation.

두 장치 실패는 endpoint 전 실패로 별도 JSON에 보존했다. 첫째는 모든 split에서 완전 유한 뉴런만 허용한 과잉 조건, 둘째는 공식 loader가 쓰지 않는 sparse `Ratio2` 선택이었다. 최종 입력은 공개 loader의 취지와 맞는 `gRaw/rRaw`이며 nuisance fit은 calibration block에만 제한했다.


# 최종 판정

Status: COMPLETE

원래 회로-계량 인과식은 실자료에서 `BLOCKED_PARENT_RECEIPT`다. 관측 대체식은 실제 뉴런별 역치, recording별 좌표, directed cycle별 강도를 반영해 실행됐지만 최종 판정은 `FAIL_PREDICTIVE_FEATURE_GATE`다.

A1의 실패는 수학적으로 유익했다. 이미 적합된 recurrent predictor에 positive geometry gain을 사후 곱하면 weight norm과 spectral radius가 커져 예측을 왜곡했다. 이를 A2의 anisotropic ridge로 고쳐 독립 test에 재시도했지만, 평균 효과는 작고 exact p-value는 `0.0625`였으며 circuit-shuffle 효과가 더 컸다.

따라서 현재 자료가 지지하는 결론은 다음뿐이다.

1. 뉴런별 역치와 회로별 강도를 포함한 식은 계산 가능하다.
2. SPD graph field의 존재는 수학적으로 보장된다.
3. 그 field가 directed circuit 고유의 held-out 예측 정보를 준다는 증거는 없다.
4. 관측 칼슘 자료만으로 실제 parent circuit, 물리적 cortical folding, causal manifold deformation을 식별할 수 없다.

다음 정당한 재개 조건은 새 hyperparameter 탐색이 아니라, canonical neuron identity와 물리 parent/intervention receipt가 있는 독립 animal cohort다.

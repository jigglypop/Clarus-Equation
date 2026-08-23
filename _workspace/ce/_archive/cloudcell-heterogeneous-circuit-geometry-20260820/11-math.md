# 수학 사전감사

Status: COMPLETE

## 판정

1. 원래 식 (21)–(26)의 \(C_{ij}^n\)은 관측 시계열만으로 식별되지 않는다. 공통 잠재입력 모형과 directed-parent 모형이 같은 관측분포를 만들 수 있으므로 `BLOCKED_PARENT_RECEIPT`가 맞다.
2. A1/A2의 \(\pi_{ij}\)는 비음수이고 receiver별 합이 1이지만 인과확률은 아니다.
3. 좌표와 거리를 recording별 median 6-NN 거리로 나누므로 \(L,k,\ell,\sigma_g,R,W\)는 모두 무차원이다. exp의 인자도 무차원이다.
4. symmetric normalized \(L\succeq0\)이므로 \(I+L\succ0\)이고 \(k\)는 유일하다. \(g_i=e^{k_i}I_3\)는 항상 SPD다.
5. A1의 post-fit multiplication은 이미 fit된 predictor를 재스케일하므로 일반적 개선 정리가 없다. 이것이 실데이터에서 실패할 경우 norm/spectral/prediction-variance 영수증을 확인하고 A2의 anisotropic ridge로 바꾸는 것은 사전 열거된 구조 수정이다.
6. A2 역시 성능 개선 정리가 없다. test 성능만 경험적으로 판정한다.

## 반례 경계

- \(R\ne1\)이면 \(W^{(0)}\)가 validation optimum이라는 이유만으로 \(W^{post}\)가 낫지 않다.
- \(S_\Gamma\mapsto k\mapsto R\)는 정의상 결합되어 있으므로 그 상관은 독립 검증이 아니다.
- signed \(W\)를 nonnegative offspring matrix와 동일시할 수 없다.
- bounded state나 SPD metric만으로 고정점·학습·AGI가 따라오지 않는다.


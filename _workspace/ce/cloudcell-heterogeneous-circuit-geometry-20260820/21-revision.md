# Validation 실패와 수학 수정

Status: A2_ACTIVATED

## A1 결과

다섯 GCaMP recording 모두에서 A1 validation \(\Delta_s<0\)였고 평균은 `-0.2764339264`였다. directed-strength shuffle 평균도 `-0.2592270095`, GFP는 `-0.2163492882`였다. 따라서 cycle-specific predictive gain이 없고 A1 gate는 실패했다.

## 실패 영수증

A1은 baseline 대비 Frobenius norm을 `1.0864–1.3666`배, spectral radius를 `1.2745–1.4195`배로 키웠다. 다섯 중 네 recording에서 prediction variance도 증가했다. 이는 이미 construction split에서 적합된 \(W^{(0)}\)에 positive geometry gain을 사후 곱한 것이 안정적인 regularization이 아니라 predictor scale distortion임을 보여준다.

## 수정

계약에 미리 적은 A2만 활성화한다.

\[
W^{reg}=\arg\min_W \|Y-WX-b\|_F^2
+10^{-2}\sum_{ij}W_{ij}^2/R_{ij}.
\]

이 식은 \(R\)이 큰 경로를 덜 수축시키되, data-fit normal equation 안에서 다시 최적화한다. threshold, graph, cycle, \(R\), ridge scale, horizon, split, seed는 바꾸지 않는다. A2 confirmation 실패 뒤 추가 수정은 없다.

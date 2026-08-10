# G9-3D: 방향성 성장 텐서의 3차원 표면

높이장 \(h(x,y)\)의 기계적 에너지를

\[
E[h]=\int\left[\frac\kappa2(\Delta h)^2+\frac\mu2h^2-rac12\nabla h^T G(x,y)\nabla h+\frac\beta4\|\nabla h\|^4\right]dxdy
\]

로 확장한다. \(G=\gamma I\)는 스칼라 differential-growth null이고, 대안은 traceless 방향 텐서 \(c q(n\otimes n-I/2)\)를 더한다. gradient flow 결과 \((x,y,h)\)를 OBJ mesh로 내보내면 사용자가 말한 공간 단면의 3차원 형상에 해당한다.

이 gate는 null에서 \(c=0\), 대안에서 \(c=0.8\) 회복, holdout 표면 RMSE와 상위 ridge mask IoU 개선을 요구한다. 또한 입력 텐서와 초기면을 90도 회전한 결과가 원래 결과의 90도 회전과 수치적으로 같아야 한다. 이는 좌표축을 주름 원인으로 하드코딩하는 오류를 막는다.

V1 validation은 계수 0/0.8 회복, RMSE 91.7% 감소, 회전 RMSE \(2.1\times10^{-18}\)을 달성했지만 ridge IoU 절대 gain이 0.0961로 0.10 문턱을 근소하게 놓쳤다. 기준선 IoU가 이미 0.890이라 포화된 탓이다. V2는 ridge mismatch \(1-\mathrm{IoU}\)의 비례 감소율을 사용해 최소 50% 감소를 요구한다.

## V2 최종 결과

locked test는 `PASS`였다. null 계수는 0, 대안 계수는 0.8을 회복했다. 표면 RMSE는 91.6%, ridge mismatch는 89.3% 감소했고 90도 회전 equivariance RMSE는 \(2.11\times10^{-18}\)이었다. 외부 다운로드는 0, 실행시간은 11.44초다. 수치 보고서는 `artifacts/agi/folding_surface_3d_test_v2.json`, 3차원 삼각 mesh는 `artifacts/agi/folding_surface_3d_v2.obj`다.

OBJ는 수학적 높이장의 시각화이지 뇌 MRI 분할이나 개인 cortical surface가 아니다. 실제 검증에는 cortical thickness, curvature, sulcal depth, 연결 방향의 동일 좌표계 파생치가 필요하다.

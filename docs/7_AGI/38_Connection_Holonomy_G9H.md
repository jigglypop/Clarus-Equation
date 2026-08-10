# G9-H: Connection과 holonomy 최소 검증

국소 차트가 톱니처럼 맞물린다는 생각을 scalar Laplace–Beltrami 다음 단계인
Levi-Civita parallel transport로 형식화한다. 단위구면의 점 `p`에서 `q`로 가는
최단 geodesic을 따라 접벡터를 운반하고 폐곡선 뒤의 회전각을 측정한다.

구면 삼각형 `T`에 대해 Gauss–Bonnet는

\[
\theta_{\mathrm{hol}}=\int_T K\,dA
\]

를 준다. 단위구면은 `K=1`이므로 holonomy는 signed spherical area와 같고,
평면은 `K=0`이므로 null holonomy다.

V1은 두 구면 삼각형, 평면 null, 임의 강체회전 후의 등변성을 구현 전에
사전등록했다. 최단 geodesic transport는 `p×q` 축으로 `p`를 `q`에 보내는 3차원
회전을 접벡터에도 적용해 계산한다.

결과:

- 평면 holonomy 절댓값: 0
- 구면 octant: area = holonomy = 1.5707963267948966
- 두 번째 삼각형: area = holonomy = 0.5512855984325309
- 최대 area 오차: 0
- 최대 강체회전 등변성 오차: 1.11e-16
- 외부 데이터: 0 bytes
- 실행시간: 약 0.0014초
- 상태: `PASS`

이는 connection 구현과 곡률-누적회전 항등식의 수치 검증이다. 피질 자체의
Levi-Civita connection에서 holonomy를 계산하면 표면 곡률을 다시 표현할 뿐이므로
신경 가설의 독립 증거가 되지 않는다. 다음 실제 검증에는 표면에서 독립적으로
측정된 백질 섬유 방향, 발생 성장 방향 또는 기능적 위상 방향장이 필요하다. 그
방향장을 표면 접공간으로 투영하고 Levi-Civita 기준 transport와의 residual
holonomy를 측정해야만 `표면 기하 이상으로 남는 꼬임`을 시험할 수 있다.

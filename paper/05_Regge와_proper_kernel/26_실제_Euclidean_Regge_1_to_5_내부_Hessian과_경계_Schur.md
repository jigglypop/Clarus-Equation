# 26. 실제 Euclidean Regge $1\to5$: 내부 Hessian과 경계 Schur

앞 장은 우주론 장부의 중복 계상을 막았다. 이 장은 중력 쪽 반례를 따라 증명 순서를 바꾼다. 처음 후보였던 두 4-simplex가 tetrahedron 하나를 공유하는 복잡은 internal edge와 internal triangular hinge를 전혀 만들지 않아 internal Hessian을 검사할 수 없었다. 그래서 그 길을 억지로 닫지 않고, 고정 경계를 보존하면서 내부 변수를 만드는 최소 4차원 refinement인 barycentric Euclidean Regge $1\to5$로 전진한다. 순서는 **기하의 정의 → 내부 게이지 영점 → quotient Schur 항등식 → 양자 진폭으로 아직 못 가는 이유**다.

## 26.1 무엇을 계산하는가

제곱 변 길이가 $2$인 regular Euclidean 4-simplex의 경계를 고정하고 barycentre를 삽입한다. 다섯 fine 4-simplex가 생기며 다섯 internal edge의 제곱 길이는 각각 $4/5$다. 물리 길이 $l$과 기준길이 $L_{\rm ref}>0$를

$$
l=L_{\rm ref}\hat l,\qquad
\widehat S=\frac{S_{\rm geom}}{L_{\rm ref}^{2}}
\tag{1}
$$

로 분리한다. 구현에서 미분하는 것은 무차원 길이 $\hat l$이다. internal triangle $h$에는 $\widehat A_h(2\pi-\sum\theta_h)$, boundary triangle에는 $\widehat A_h(\pi-\sum\theta_h)$를 더한다. 물리 작용에 곱할 $L_{\rm ref}^2/(8\pi G)$는 **의도적으로 생략**한다. 따라서 이것은 finite fixed-boundary Euclidean geometry의 고전 Hessian이며 Lorentzian path integral이나 proper/EPRL amplitude가 아니다.

**문과 비유.** 건물 외벽의 꼭짓점은 그대로 두고 가운데 기둥 하나를 세워 방을 다섯 개로 나누는 일과 같다. 외벽은 경계 자료, 기둥 위치는 내부 자료다. 이 장은 건물 전체의 양자 확률을 계산하지 않고, 기둥을 움직일 때 실제 방 모양이 변한 방향과 좌표를 다시 붙인 방향을 구분한다.

## 26.2 내부 Hessian의 네 영점과 한 물리 방향

내부 길이 Hessian을 $H$라 하자. flat barycentric family와 fixed-boundary Schlaefli identity에서 내부점의 네 방향 이동을 길이 변화로 보내는 Jacobian은

$$
J=-\frac{I-\mathbf1\mathbf1^{\mathsf T}/5}{\sqrt{4/5}},
\qquad \operatorname{rank}J=4,\qquad HJ=0.
\tag{2}
$$

$\mathbf1=(1,1,1,1,1)^{\mathsf T}$다. 다섯 내부 길이의 네 조합은 내부점 위치를 바꿔 생기는 gauge 방향이고 equal-radius 방향 하나만 남는다. regular boundary의 $S_5$ 대칭은

$$
H=8\sqrt5\,\mathbf1\mathbf1^{\mathsf T},\qquad
\operatorname{spec}(H)=\{0,0,0,0,40\sqrt5\}.
\tag{3}
$$
따라서 gauge-unfixed $5\times5$ inverse는 없다. $u=\mathbf1/\sqrt5$에 사영하면

$$
H^+=\frac{uu^{\mathsf T}}{40\sqrt5}
\tag{4}
$$

라는 radial mode의 Moore--Penrose pseudoinverse만 정의된다. raw finite-difference Hessian이 full rank로 보이는 것은 절단 오차가 영점을 들어 올린 diagnostic일 뿐 exact inverse의 근거가 아니다.

이산 기하의 vertex-displacement 대칭과 broken symmetry의 맥락은 [Dittrich--Freidel--Speziale (2007)](https://arxiv.org/abs/0707.4513)와 [Höhn (2014)](https://arxiv.org/abs/1411.5672)을 따른다. 최근 $1\to5$ 계산 예는 [Li et al. (2025)](https://arxiv.org/abs/2501.16094)에 있다. 이 문헌들은 이 저장소의 finite boundary Schur 계산이나 proper multicell 단계를 대신 증명하지 않는다.

## 26.3 경계에 남는 quotient Schur 항

경계 길이 $b\in\mathbb R^{10}$와 내부 길이 $y\in\mathbb R^5$에 대해

$$
H_f=\begin{pmatrix}A&B\\B^{\mathsf T}&C\end{pmatrix},
\qquad S_f(b,y(b,q))=S_c(b)
\tag{5}
$$

라고 하자. nondegenerate flat section에서 interior-point orbit의 미분은 $CQ=0$, boundary-gradient identity의 미분은 $BQ=0$이다. $C$의 inverse가 없는 것은 실패가 아니라 quotient로 내려가야 한다는 기하의 신호다. flat section의 미분 $Ds$는

$$
B^{\mathsf T}+CDs=0
\tag{6}
$$

를 만족한다. regular barycentric point에서 $C=40\sqrt5\,uu^{\mathsf T}$와 식 (4)를 쓰면,

$$
H_c=A+BDs=A-BC^+B^{\mathsf T}.
\tag{7}
$$

가 된다. 이것이 C4-REGGE-ONE-TO-FIVE-BOUNDARY-SCHUR-23B의 **조건부 고전 boundary Schur identity**다. [regge_one_to_five_boundary_hessian.py](../../examples/physics/regge_one_to_five_boundary_hessian.py)와 [대응 테스트](../../tests/test_regge_one_to_five_boundary_hessian.py)의 기록은 6 passed, Regge 묶음은 15 passed다. 실행 성공은 식 (7)의 증명 수가 아니라 구현 재현이다.

**문과 비유.** 경계 서류를 처리하는 부서와 내부 기둥 위치를 정하는 부서가 있다고 하자. 기둥을 내부에서 옮기는 네 방식은 서류 결과를 바꾸지 않는 재배치다. Schur 항은 내부 부서의 효과를 경계 서류에 정확히 반영하는 정산식이지, 기둥 좌표를 억지로 하나의 역행렬로 정하라는 명령이 아니다.

## 26.4 proper/EPRL 양자중력을 아직 증명하지 않은 이유

22C와 23C의 ceiling은 남아 있다. 얻은 것은 flat fixed-boundary **classical quotient** identity다. raw inverse, Gaussian determinant 또는 integral, proper/EPRL multi-cell block, spin-foam measure와 contour, curved refinement, continuum limit, Einstein--Hilbert dominance는 아직 계산하거나 유도하지 않았다.

proper vertex의 orientation sector는 [Engle (2011)](https://arxiv.org/abs/1111.2865), asymptotic transition-amplitude 맥락은 [Engle--Vilensky--Zipfel (2015)](https://arxiv.org/abs/1505.06683)을 참조할 수 있다. 그러나 이 문헌도 이 저장소의 유한 boundary Schur identity나 proper multicell Gaussian을 증명하지 않는다. 다음 증명은 문헌 이름으로 결론을 닫는 일이 아니라, actual proper/EPRL multi-cell의 $A,B,C$, measure/contour와 gauge-reduced Gaussian을 별도로 계산하는 일이다.

Lorentzian 쪽에서 그 계산의 경계 입력을 어디까지 실제로 만들었는지는 [27장: 닫힘, 스핀 근사, 국소 LS 벡터](27_Lorentzian_1_to_5_닫힘_스핀_LS.md)에서 잇는다.

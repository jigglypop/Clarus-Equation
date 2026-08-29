# 27. Lorentzian $1\to5$: 닫힘, 스핀 근사, 국소 LS 벡터

이 장은 실제 Lorentzian $1\to5$ 기하에서 양자 경계 자료를 만들기 위해 어디까지 전진했는지 설명한다. 결론부터 말하면, 15개 spacelike tetrahedron의 **기하학적 닫힘**과 그에 붙이는 **방향 spinor**는 고정했고, 하나의 단순한 선형 면적--스핀 대응이 정확히 성립할 수 없다는 장애도 증명했다. 그 뒤 반올림 오차를 제어하는 정수 스핀 족과, 작은 스핀 수준에서 15개의 0이 아닌 국소 Livine--Speziale(LS) 벡터까지 구성했다. 아직 이것은 다섯 vertex를 잇는 proper EPRL 진폭이나 그 Hessian이 아니다. 이 순서를 구분해야 다음 증명이 무엇인지 정확해진다.

가장 쉬운 비유는 접힌 종이 모형이다. 각 사면체는 네 면을 가진 작은 방이고, 네 면이 미는 힘이 합쳐서 0이어야 방이 공중에서 기울지 않는다. 이 균형을 먼저 확인한 뒤에야 각 면에 붙일 양자 표지와, 이웃 방의 표지를 어떻게 서로 읽을지를 정할 수 있다. 방마다 표지를 만든 사실만으로 건물 전체의 출입 규칙이 완성되는 것은 아니다.

## 27.1 먼저 고정한 것: 15개 사면체의 기하학적 닫힘

24장에서 잡은 Lorentzian $1\to5$ 고전 gluing skeleton에는 boundary tetrahedron 다섯 개와 internal tetrahedron 열 개, 모두 15개의 spacelike tetrahedron이 있다. spacelike라는 말은 각 사면체의 내부 기하가 양의 정부호인 3차원 Euclidean 기하로 읽힌다는 뜻이다. 전역 vertex label을 정렬하고 induced Gram matrix에 Cholesky convention을 적용하면, 각 사면체에 임의성이 없는 intrinsic 좌표표를 하나 고를 수 있다.

사면체 $t$의 면 $f$에 대해 outward unit normal을 $n_f\in\mathbb R^3$, 면적을 $a_f>0$라 쓰면, outward area vector는 $a_fn_f$이고 다음 항등식이 성립한다.

$$
\sum_{f\subset t} a_f n_f=0.
\tag{1}
$$

이는 양자역학의 가정이 아니라 닫힌 사면체의 경계가 서로 상쇄한다는 기하학이다. 코드에서는 상대 잔차

$$
r_t=\frac{\left\|\sum_{f\subset t}a_fn_f\right\|}{\sum_{f\subset t}a_f}
\tag{2}
$$

를 계산한다. 기본 좌표에서 15개 $r_t$와 아래 spinor 검사의 최대 잔차는 모두 $5\times10^{-15}$보다 작다. 이는 부동소수 계산의 재현 확인이며, 종이 위의 식 (1)을 대신하는 독립 증명은 아니다.

각 단위법선에는 Hopf map의 두 chart 중 하나를 고정해 방향 spinor $\xi(n_f)\in\mathbb C^2$를 붙였다. Pauli 행렬 벡터를 $\sigma$라 하면 이 선택은

$$
\xi(n_f)^\dagger\xi(n_f)=1,
\qquad
\xi(n_f)^\dagger\sigma\xi(n_f)=n_f
\tag{3}
$$

를 만족한다. $\xi$는 면의 방향을 적는 화살표이지 아직 면에 배정된 양자수 $j$도, coherent intertwiner도 아니다. 같은 internal tetrahedron을 두 fine cell에서 다시 읽을 때 area-squared label, normal, spinor가 일치하는 것은 이 **같은 전역 정렬과 같은 Cholesky chart**를 다시 실행했기 때문이다. 독립 local frame들의 전이법칙이나 quantum gluing을 이미 얻었다는 뜻은 아니다.

구현과 검사는 [coherent boundary 모듈](../../examples/physics/proper_vertex_one_to_five_coherent_boundary.py), [대응 테스트](../../tests/test_proper_vertex_one_to_five_coherent_boundary.py)에 남아 있다. 이 단계의 산출물은 기하와 방향 자료다. half-integer spin, $\gamma$, 면적 spectrum, time orientation, bra--ket dualization, $SU(2)$ 또는 $SL(2,\mathbb C)$ lift는 여기서 정하지 않았다.

## 27.2 왜 증명 순서를 바꾸었는가: 정확한 공통 스케일은 먼저 반증한다

처음에는 모든 기하 면적을 하나의 공통 스케일로 정확한 half-integer 스핀에 맞출 수 있는지부터 시험하고 싶어진다. 그러나 이 문제를 수치 탐색으로 오래 붙들 필요는 없다. 기준 길이 $L_{\rm ref}$로 무차원화한 면적을

$$
a_f=\frac{A_f}{L_{\rm ref}^2}
\tag{4}
$$

라고 하고, 임시로 선형 proxy $j_f=\alpha a_f$만 허용하자. 두 삼각형 $(0,1,2)$와 $(0,1,3)$의 정확한 면적제곱 비는

$$
\frac{a_{012}^2}{a_{013}^2}=\frac{999983}{999918},
\qquad v_3\!\left(\frac{999983}{999918}\right)=-3.
\tag{5}
$$

여기서 $v_3$는 분수의 분자·분모에 들어 있는 $3$의 지수 차이다. 유리수의 제곱이면 모든 소수의 지수 차는 짝수여야 하지만 $-3$은 홀수다. 따라서 식 (5)의 비는 유리수 제곱이 아니다. 공통 $\alpha$가 모든 $a_f$를 정확한 half-integer로 보낸다면 두 면적의 비는 유리수가 되어야 하므로 모순이다.

이 no-go의 범위는 정확하다. 반증한 대상은 $A=\alpha j$라는 **선형 proxy**뿐이다. 표준 LQG의 $\sqrt{j(j+1)}$ 면적 spectrum, 유한 수준의 정확한 Regge boundary state, exact spin-weighted closure, LS closure 또는 EPRL/proper 자료를 반증한 것이 아니다. 그래서 막힌 가지를 억지로 닫지 않고, 다음으로 실제로 제어 가능한 근사 가족을 전진시켰다.

## 27.3 반올림으로 전진하는 정수 스핀 가족

양의 정수 $N$마다

$$
j_f(N)=\operatorname{nint}(N a_f)
\tag{6}
$$

로 가장 가까운 정수를 택한다. 반올림의 정의만으로 각 면에서

$$
\left|\frac{j_f(N)}{N}-a_f\right|\leq\frac{1}{2N}
\tag{7}
$$

이고, 식 (1)에 이 오차를 대입하면

$$
\left\|\sum_{f\subset t}\frac{j_f(N)}N n_f\right\|
\leq\frac{2}{N}
\tag{8}
$$

를 얻는다. (8)은 양자 closure가 정확하다는 선언이 아니다. 정확한 기하학적 닫힘에 면마다 최대 $1/(2N)$인 라벨 오차를 더한 상계다.

정확한 rational square-root enclosure로 네 면의 polygon admissibility 여유를 동시에 하계하면 $N\ge118$이면 충분하다. 이 수준에서는 20개 삼각형 모두에 양의 정수 스핀이 붙고, 15개 four-valent invariant space가 모두 0이 아니다. 수치로 읽으면 면적 오차 상계는 $1/236$, 재스케일된 닫힘 결함 상계는 $1/59$다.

**문과 비유.** 서로 다른 실제 길이의 막대에 공통 눈금자를 대면 모든 길이가 정확히 정수 칸이 되지는 않는다. 그 사실을 먼저 증명한 것이 식 (5)다. 하지만 눈금자를 $N$배 촘촘하게 만들면 각 막대의 오차는 반 칸 이하로 줄어든다. 식 (7)과 (8)은 그 “반 칸”이 사면체 전체 균형을 얼마나 흔드는지까지 계산한 보증서다.

이 가족은 [spin assignment 모듈](../../examples/physics/proper_vertex_one_to_five_spin_assignment.py)과 [대응 테스트](../../tests/test_proper_vertex_one_to_five_spin_assignment.py)에 있다. 이것은 표준 면적법칙을 채택하거나 exact finite-$N$ 기하를 완성한 것이 아니라, 다음 국소 intertwiner 계산을 가능하게 하는 명시적 입력이다.

## 27.4 실제로 만든 국소 LS 벡터

작은 계산 수준 $N=3$에서 앞 절의 전역 triangle spin label과 식 (3)의 방향 spinor를 각 사면체에 넣었다. 네 면의 coherent state의 텐서곱을 four-valent invariant recoupling basis $\{|k\rangle\}$에 정사영한다.

$$
P_{\rm inv}=\sum_k |k\rangle\langle k|,
\qquad
|\iota_t\rangle=P_{\rm inv}\bigotimes_{f\subset t}|j_f,\xi_f\rangle.
\tag{9}
$$

여기서 $P_{\rm inv}$는 Haar projector를 orthonormal invariant basis로 쓴 등식이다. Haar 적분을 수치 quadrature로 근사했다는 뜻이 아니다. Condon--Shortley convention 아래 직접 magnetic-basis projection과 recoupling coefficient가 맞는지도 검사했다.

$N=3$, 최대 $j=9$에서 15개 local coefficient vector는 모두 0이 아니며, unnormalized group-average norm의 최솟값은 약

$$
\min_t\|\iota_t\|\simeq0.0725113
\tag{10}
$$

이다. [LS intertwiner 모듈](../../examples/physics/proper_vertex_one_to_five_ls_intertwiners.py)과 [대응 테스트](../../tests/test_proper_vertex_one_to_five_ls_intertwiners.py)는 invariant basis의 직교성, $SU(2)$ 불변성, 두 계산법의 일치를 확인한다. 이 값은 “국소 벡터가 영벡터로 사라지지 않는다”는 제한된 사실을 말한다.

문헌에서 proper vertex의 Lorentzian 다중 simplex 문제는 계속 섬세한 과제다. [Engle--Zipfel (2015)](https://arxiv.org/abs/1502.04640)은 Lorentzian proper vertex의 비퇴화 경계 자료와 단일 vertex의 semiclassical 분석을 다루며, [Engle--Vilensky--Zipfel (2015)](https://arxiv.org/abs/1505.06683)은 한 4-simplex의 proper vertex가 단 하나의 Feynman term을 갖는 asymptotics를 다룬다. 두 결과 모두 이 장의 15개 국소 벡터를 전역으로 glue한 다섯-vertex proper EPRL amplitude나 multicell Hessian을 제공하지 않는다. 그래서 이 문헌을 이름만으로 마지막 다리의 증명으로 쓰지 않는다.

## 27.5 여기서 멈추지 않고 이어갈 정확한 순서

국소 LS 벡터는 globally glued spin network가 아니다. 독립 tetrahedron $SU(2)$ frame, future/outward normal의 선택, shared face에서 어느 쪽을 ket으로 읽고 어느 쪽을 bra로 읽을지, 그리고 Lorentzian boost 자료가 아직 빠져 있다. 따라서 식 (10)을 proper amplitude의 saddle 또는 Hessian으로 승격할 수 없다.

다음 증명은 아래 순서를 건너뛰지 않는다.

1. future/outward normal과 $SL(2,\mathbb C)/SU(2)$ coset lift를 구성한다.
2. 독립 $SU(2)$ tangent frame, Regge phase, shared-face bra--ket gluing을 구성한다.
3. $Y_\gamma$, proper projector, gauge fixing과 적분 measure를 실제로 넣는다.
4. 그 뒤에만 standard proper EPRL 다섯-vertex 합과 gauge-reduced Hessian을 계산한다.

이 순서는 앞 단계의 약점을 감추기 위한 가지치기가 아니라, 각 단계가 다음 단계의 입력이 되도록 증명의 방향을 고정한 것이다. 26장의 Euclidean classical quotient Schur와 이 장의 Lorentzian local boundary data는 서로 보완적인 준비물이다. 둘을 합쳐도 아직 양자 multi-vertex 진폭은 아니며, 바로 그 차이가 다음 계산에서 검증할 대상이다.

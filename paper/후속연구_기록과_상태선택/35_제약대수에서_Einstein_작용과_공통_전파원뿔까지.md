# 35. 제약 대수에서 Einstein 작용과 공통 전파 원뿔까지

2026-09-21. CE-CC1. 관계기하에서 동적 중력으로 가는 연결의 역유도와
양의 운동비용을 직접 동일시하는 후보의 제약면 반례.

## 0. 기존 결과와 이번 유도의 차이

[55장](../06_QFT_재설계/55_확장_상태_매장의_Einstein_변분_조건.md)과
[E54](../검증_원장/참조_양자_보존_원장.md)는 Einstein–Hilbert 작용을 공급하고
그 변분·고전 제약을 계산했다. [23장 R55](23_관계시계와_기하에서_물리작용까지의_조건.md)는
양의 공간 계량의 pullback만으로 Lorentz 시공간을 얻지 못함을 보였다.

여기서는 **제약 대수 → Hamiltonian의 계수 → Einstein 작용** 순서로 역유도한다.
동시에 양의 관계비용을 여섯 계량 성분의 운동항으로 직접 쓰는 후보를 검사한다.
이 역유도는 아래 작용류에 한정한 고전 정리다.

| 입력 전제 | 현재 지위 |
|---|---|
| 매끄러운 3차원 공간과 독립 canonical 변수 \(h_{ij},\pi^{ij}\) | 공급함; CE 관계변수에서 이 canonical 쌍을 얻는 사상은 미유도 |
| 국소성, 시간반전 짝성, 운동량의 ultralocal 이차식 | 선택한 최소 ansatz |
| 공간 미분 차수 2까지: 상수항과 \(^{(3)}R\)에 선형인 퍼텐셜 | 선택한 절단; 고차 곡률은 포함하지 않음 |
| 임의 lapse에 대한 Lorentzian 초곡면 변형 대수 | 목표 대수로 요구함; CE에서 그 대수 자체가 발생하는 증명은 아님 |
| 추가 제약·선호 절편·물질–계량 운동량 혼합이 없음 | 이번 정리의 제한 |

따라서 이번 결과를 시공간 차원·Lorentz 부호·양자 중력의 무가정 유도로 세지 않는다.
기존에 공급한 EH 작용의 조건부 변분도 그대로 보존한다.

## 1. 미정 계수를 남긴 최소 Hamiltonian

경계 없는 compact 공간 \(\Sigma\) 또는 부분적분 경계항이 사라지는 조건을 쓴다.
\(h_{ij}>0\), \(h=\det h_{ij}\), \(\pi^{ij}\)는 무게 1의 대칭 운동량 밀도이며
\(\pi=h_{ij}\pi^{ij}\)다.
\[
\{h_{ij}(x),\pi^{kl}(y)\}
=\delta_{(i}^{k}\delta_{j)}^{l}\delta^3(x-y).
\]
운동량 제약은 공간 미분동형사상을 생성하도록
\[
D[v]=\int_\Sigma\pi^{ij}\mathcal L_vh_{ij}
=-2\int_\Sigma v_j\nabla_i\pi^{ij}
\]
로 둔다. Hamiltonian 제약의 계수는 아직 정하지 않는다.
\[
\boxed{H[N]=\int_\Sigma N\left[
\frac a{\sqrt h}\left(\pi^{ij}\pi_{ij}-\omega\pi^2\right)
+\sqrt h\left(b\,{}^{(3)}R+c_0\right)\right].} \tag{35.1}
\]
\(a>0\), \(b,c_0,\omega\)는 상수다. \(a>0\)는 traceless 운동 방향의
부호를 고정한다. 미정 \(\omega\)가 trace 방향을 제어한다.
이 작용류 안에서는 이차 운동량 불변량이 두 개다.

요구하는 Lorentzian 대수는
\[
\{D[v],D[w]\}=D[[v,w]],\qquad
\{D[v],H[N]\}=H[\mathcal L_vN],
\]
\[
\boxed{\{H[N],H[M]\}=D[v],\quad
v^i=h^{ij}(N\partial_jM-M\partial_jN).} \tag{35.2}
\]
첫 두 식은 밀도와 공간 공변성에서 따른다. 마지막 식의 계수와 잔여항을 계산한다.

## 2. 괄호 계산과 필요충분조건

**정리 35-A.** 식 (35.1)이 같은 독립 변수에서 식 (35.2)를 추가 제약 없이
함수형 항등식으로 만족하는 필요충분조건은
\[
\boxed{\omega=\frac12,\qquad ab=-1.} \tag{35.3}
\]
\(a\)의 크기와 \(c_0\)는 이 조건으로 고정되지 않는다.

**증명.** 운동항의 운동량 미분은
\[
\frac{\delta H_T[N]}{\delta\pi^{ij}}
=\frac{2aN}{\sqrt h}(\pi_{ij}-\omega\pi h_{ij}).
\]
곡률 퍼텐셜의 계량 미분에서 lapse 미분을 포함하는 부분은
\[
\left.\frac{\delta H_R[N]}{\delta h_{ij}}\right|_{\nabla N}
=b\sqrt h(\nabla^i\nabla^jN-h^{ij}\Delta N).
\]
미분 없는 부분과 두 ultralocal 운동항의 괄호는 \(NM\) 반대칭화에서 소거된다.
남은 수축은
\[
\{H[N],H[M]\}=2ab\int_\Sigma
\left\{\pi^{ij}(M\nabla_i\nabla_jN-N\nabla_i\nabla_jM)
+(2\omega-1)\pi(M\Delta N-N\Delta M)\right\}.
\]
부분적분하면
\[
\boxed{\{H[N],H[M]\}
=-abD[v]+2ab(2\omega-1)\mathcal A[v],}
\tag{35.4}
\]
\[
\mathcal A[v]=\int_\Sigma\sqrt h\,v^i\partial_i(\pi/\sqrt h).
\]
이 형태는 운동량의 밀도 무게도 포함한다.

\(D[v]\)와 trace 기울기 항은 독립이다. 예컨대 평탄한 국소 좌표에서
trace는 일정하지만 \(\partial_i\pi^{ij}\)는 변하는 운동량을 택할 수 있고,
반대로 §4처럼 \(D_i=0\)이지만 trace 기울기는 비영인 운동량도 있다.
따라서 임의 lapse·운동량에서 원하는 대수가 성립하려면
\(-ab=1\), \(2ab(2\omega-1)=0\)이어야 한다.
역으로 이 두 조건을 대입하면 잔여항이 없어져 정확히 (35.2)가 된다. \(\square\)

\(b=0\)인 ultralocal 가지는 두 Hamiltonian 제약의 괄호가 0이므로
이번의 임의 공간 변형 대수를 표현하지 않는다.
\(\partial_i(\pi/\sqrt h)=0\) 같은 추가 제약을 넣어 잔여항을 없애는 것은
다른 제약계이며, 위 필요조건을 반박하지 않는다.

## 3. Einstein 작용의 역복원과 남는 상수

\[
K_{ij}=\frac{\dot h_{ij}-\mathcal L_{\vec N}h_{ij}}{2N}
\]
로 쓰면 Hamilton 방정식과 (35.3)은
\[
\pi^{ij}=\frac{\sqrt h}{a}(K^{ij}-Kh^{ij})
\]
를 준다. 이를 canonical 작용에 대입하는 Legendre 변환은
\[
S=\frac1a\int dt\,d^3x\,N\sqrt h
\left[{}^{(3)}R+K_{ij}K^{ij}-K^2-ac_0\right].
\]
\[
M^2=\frac2a>0,\qquad \Lambda=\frac{ac_0}{2}
\]
라 놓고 표준 경계항을 포함하면
\[
\boxed{S=\frac{M^2}{2}\int d^4x\sqrt{-g}(R^{(4)}-2\Lambda).} \tag{35.5}
\]
따라서 이 제한된 작용류에서는 EH 항을 출발점으로 넣지 않고 그 형태를 얻었다.
반면 \(M^2\), \(\Lambda\), 그리고 목표 변형 대수 자체는 자연값으로 선택되지 않았다.
공통 lapse의 시간 단위도 대수의 정규화를 정할 때 고정한 것이다.

고전 제약이 독립인 정칙 영역에서는 여섯 계량 쌍에서 네 first-class 제약과
그 gauge 방향을 제거하면 두 configuration 자유도가 남는다.
이 계산은 양자 연산자 순서·이상항·물리 내적의 검증이 아니다.

## 4. 양의 운동비용 후보의 실제 제약면 반례

운동량을 trace와 traceless 부분으로 나누면
\[
\pi^{ij}\pi_{ij}-\omega\pi^2
=\pi_T^{ij}\pi^T_{ij}+\left(\frac13-\omega\right)\pi^2.
\]
여섯 방향 모두 양수이려면 \(\omega<1/3\)이어야 하지만,
필요한 값은 \(\omega=1/2\)다. 정규 직교 대칭행렬 기저에서 필요한 이차형식은
\[
\operatorname{diag}(1,1,1,1,1,-1/2).
\tag{35.6}
\]
공간 계량 \(h_{ij}\)가 양수인 것은 GR과 양립한다. 여기서 다른 대상인 것은
**여섯 계량 성분의 운동 방향에 정의된 계량**이다.

부호 논의만으로 끝내지 않고 실제 제약면 위에서 확인하자.
\[
\Sigma=S^1_{2\pi}\times S^1_1\times S^1_1,\qquad
h_{ij}=\delta_{ij},\qquad
\pi^{ij}=\operatorname{diag}(0,\cos x,\sin x),
\]
\[
a=1,\quad b=-1,\quad \omega=0,\quad c_0=-1.
\]
선택한 단위에서 이 값들은 반례의 계수이며 관측 우주상수의 값이 아니다.
모든 점에서
\[
\mathcal H=\cos^2x+\sin^2x-1=0,\qquad
\mathcal D_i=-2\partial_j\pi^{ji}=0.
\]
두 lapse를 모두 양수로
\[
N=1,\qquad M=2+\sin x
\]
라 놓으면 \(v^x=\cos x\), 다른 성분은 0이다. 원하는 대수의 우변은 \(D[v]=0\)이다.
그러나 함수형 미분을 먼저 취하고 이 상태에 대입한 실제 괄호는
\[
\{H[1],H[2+\sin x]\}
=2\int_0^{2\pi}\cos x\,(-\sin x+\cos x)\,dx
=\boxed{2\pi\ne0}. \tag{35.7}
\]
\(y,z\) 적분의 길이는 각각 1이다. 원래의 lapse 이차미분 표현으로 직접 계산해도
같은 \(2\pi\)를 얻는다. 제약식을 먼저 0으로 둔 뒤 미분한 계산이 아니다.

따라서 이 데이터는 표시한 초기 제약을 모두 만족하지만, 이 제약들만으로는
임의 lapse 아래 first-class 보존이 성립하지 않는다.
추가 제약이나 lapse 제한을 얻으면 그 별도 이론을 다시 검사해야 한다.
이 반례를 수정된 모든 중력 이론의 불가능성으로 확대하지 않는다.

### 4.1 관계기하의 양성과 충돌하는 정확한 대상

양의 Fisher/Bures 이차형식 \(F\)를 실수 변수변환으로 직접 옮기면
\(J^TFJ\ge0\)이다. 가역이면 관성도 보존되므로 (35.6)의 음의 방향을 만들 수 없다.
따라서 그 양의 비용을 **미제약 여섯 계량 성분의 GR 운동항 자체**로 동일시하는
경로는 이번 ansatz와 변형 대수를 함께 만족시키지 못한다.

이는 양의 Hilbert 내적이나 양의 물질 운동항이 중력과 양립하지 않는다는 주장이 아니다.
제약으로 제거되는 conformal 방향의 부호를 물리적 음의 노름 graviton으로 읽지 않는다.
비국소 사상·추가 gauge 변수·복합 계량의 추가 제약·다른 canonical 표현은
직접 동일시와 다른 구성이다. CE의 관계변수에서 그러한 구성을 얻는 문제는 남는다.

## 5. 같은 대수는 물질의 시간·공간 운동계량도 묶는다

최소 결합하는 여러 관계장 \(\varphi^A\)와 운동량 \(p_A\)에 대해
\[
\mathcal H_m=\frac1{2\sqrt h}p_A A^{AB}(\varphi)p_B
+\frac{\sqrt h}{2}B_{AB}(\varphi)h^{ij}
\partial_i\varphi^A\partial_j\varphi^B+\sqrt h\,V(\varphi),
\]
\[
D_m[v]=\int v^ip_A\partial_i\varphi^A
\]
를 둔다. \(A,B\)는 대칭·양의 행렬로 허용하되 처음부터 서로 역행렬이라고
가정하지 않는다. 이번에는 비최소 곡률항과 중력 운동량 혼합을 제외한다.

**정리 35-B.** 중력과 같은 (35.2)의 총 대수가 성립하는 필요충분조건은
\[
\boxed{A^{AC}B_{CB}=\delta^A_{\ B}.} \tag{35.8}
\]

**증명.** \(\delta H_m[N]/\delta\varphi^A\)에서 lapse 미분을 포함하는 부분은
\(-\sqrt h B_{AB}h^{ij}(\partial_iN)(\partial_j\varphi^B)\)이고,
\(\delta H_m[N]/\delta p_A=NA^{AB}p_B/\sqrt h\)다.
\(A,B\)의 장 미분과 퍼텐셜 미분을 포함한 나머지는 \(NM\) 반대칭화에서 소거된다.
그러므로
\[
\{H_m[N],H_m[M]\}
=\int v^ip_A(AB)^A_{\ B}\partial_i\varphi^B.
\tag{35.9}
\]
최소 결합의 중력–물질 교차 괄호도 lapse 미분 없는 \(NM\) 항이라 소거된다.
임의 \(p_A,\partial_i\varphi^B\)에서 총 우변이 \(D_g[v]+D_m[v]\)가 되려면
\(AB=I\)여야 하며, 역도 즉시 성립한다. \(\square\)

이때 물질 작용은
\[
S_m=-\frac12\int\sqrt{-g}\,
B_{AB}(\varphi)g^{\mu\nu}\partial_\mu\varphi^A\partial_\nu\varphi^B
-\int\sqrt{-g}\,V(\varphi).
\]
주요 미분항의 특성 원뿔은 같은 \(g^{\mu\nu}\)로 정해진다.
반례로 \(A=I,B=\operatorname{diag}(1,2)\)는 둘 다 양수지만
두 번째 장에 \(p_2=1,\varphi^2=\sin x\), 위 같은 lapse를 쓰면
물질 괄호는 \(2\pi\), 요구되는 \(D_m[v]\)는 \(\pi\)다.
양성만으로 공통 전파 원뿔을 보장하지 못한다.

이 결과는 양의 관계기하가 쓰일 수 있는 위치도 보여준다.
양의 \(B_{AB}\)는 관계장의 target 운동계량으로 사용할 수 있고,
같은 시공간 대수가 그 시간·공간 계수를 묶는다.
하지만 \(B_{AB}\)의 구체적 함수와 \(V\), 게이지군·전하·입자 수는 선택하지 않는다.
기존 CE의 \(F(\varphi)R\) 가지는 운동량 혼합을 갖기 때문에 이 최소 결합
식을 그대로 적용할 수 없다. 별도의 canonical 변환과 정의역 확인이 필요하다.

## 6. 증거와 다음에 실제로 필요한 것

[코드](../../experiments/ce_constraint_closure_20260921/derive_closure.py)와
[결과](../../experiments/ce_constraint_closure_20260921/results.json)는
곡률 변분 수축·부분적분 두 형식·계수 해·Legendre 변환·이차형식의 관성·
제약면 반례와 다성분 물질 괄호를 정확한 기호 계산으로 대조한다.
이번 반례의 \(2\pi\)는 수치 허용오차에 의존하지 않는다.

이번에 연결한 것은 **목표 시공간 변형 대수 → 중력 작용의 형태 →
양의 관계비용과의 구분 → 같은 물질 전파 원뿔**이다.
남은 앞단은 CE 상태·관계변수에서 왜 그 canonical 구조와 대수가 나오는가이다.
남은 뒷단은 양자 제약의 이상 없는 닫힘, 상수·상태 선택과 실제 공동 관측이다.
이 단계의 조건부 중력 복원을 OBJ-01 전체의 완료로 세지 않는다.

## 선행 연구와 적용 경계

제약 표현에서 geometrodynamics를 복원하는 방향은 Hojman–Kuchař–Teitelboim의
[Geometrodynamics regained](https://doi.org/10.1016/0003-4916(76)90112-3)에 속한다.
다른 제약 선택을 통한 구성은 [Gomes의 연구](https://arxiv.org/abs/1310.1699)에서도 다룬다.
그 일반 방법이나 ADM 변환의 최초 발견을 주장하지 않는다.
여기서는 현재 CE의 직접 동일시 후보와 물질 계량에 필요한 제한을 명시하고,
정확한 제약면 반례와 함께 원고의 미완성 연결을 판정했다.

추가 second-class 제약을 포함하는 \(\lambda R\) 모형이 특정 조건에서 GR의
부분 gauge 고정과 동등할 수 있다는 [Bellorín–Restuccia의 결과](https://arxiv.org/abs/1004.0055)도
있다. 따라서 §4의 실패를 추가 제약을 갖는 모든 후보나 GR과 동등한 다른 표현의
실패로 확대하지 않는다. 그 대안은 새 제약·경계조건·자유도 장부를 갖추어 별도로 연결해야 한다.

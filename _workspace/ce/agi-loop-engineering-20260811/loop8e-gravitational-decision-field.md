# Loop 8E — evidence-sourced gravitational decision field

## 0. 판정

**[미완성: 구현 전 후보]** Loop 8D의 `갈등 → 선형 경계상승`을 후속
알고리즘으로 사용하지 않는다. 새 후보는 중력의 최소 구조인
`source → field geometry → motion → basin capture`를 결정공간에 옮긴다.
실제 중력과 신경활동이 같은 물리장이라는 주장이 아니라 인지 결정을 위한
유효모형이다.

## 1. 증거질량

**[정의]** 무차원 결정다양체를 \((\mathcal A,g_\theta)\), 행동 위치를
\(x_a\), PFC logit을 \(q_a\), Loop 8C 잔차를 \(\varphi_a\), 양의 정규화
kernel을 \(K_\sigma(x,x_a)\)라 한다. MD 상태 \(\theta\)가 metric을 정한다.

**[공리: 모델 선택]**

\[
m_a=\frac{\exp[(q_a+\eta_\varphi\varphi_a)/T_e]}
{\sum_b\exp[(q_b+\eta_\varphi\varphi_b)/T_e]},
\qquad
\rho(x)=\sum_a m_aK_\sigma(x,x_a).
\tag{G1}
\]

따라서 \(m_a>0\), \(\sum_am_a=1\)이다. residual은 행동을 직접 고르지
않고 source 질량만 이동시킨다.

## 2. 장의 변분원리

\(L_{g_\theta}=-\operatorname{div}_{g_\theta}\nabla_{g_\theta}\succeq0\),
screening \(\mu>0\), 결합 \(\kappa>0\), 공간평균 \(\bar\rho\)를 둔다.

**[공리: 모델 선택]**

\[
\mathcal F[\Phi]
=\frac12\int_{\mathcal A}
\left(\|\nabla\Phi\|_{g_\theta}^2+\mu^2\Phi^2\right)dV_g
+\kappa\int_{\mathcal A}\Phi(\rho-\bar\rho)dV_g.
\tag{G2}
\]

**[정리]** \(\mathcal A\)가 compact이고 self-adjoint 경계조건을 가지며
\(\mu>0\)이면 (G2)는 \(H^1(\mathcal A)\)에서 유일한 최소해를 갖고

\[
\boxed{(L_{g_\theta}+\mu^2)\Phi=-\kappa(\rho-\bar\rho)}
\tag{G3}
\]

를 만족한다.

**증명.** \(L_g+\mu^2I\)는 coercive positive-definite self-adjoint
operator다. 일차변분은 (G3)을 주고 strict convexity가 유일성을 준다.
\(\square\)

**[산출]**

\[
\Phi=-\kappa(L_{g_\theta}+\mu^2)^{-1}(\rho-\bar\rho).
\tag{G4}
\]

양의 증거질량은 행동 근처에 음의 potential well을 만든다. \(\mu=0\)
극한에서는 zero-mean source가 필요하고 해는 상수 gauge 자유도를 갖는다.

## 3. 결정은 낙하와 포획

**[공리: 모델 선택]** 행동 상태 \(X\)와 속도 \(V\)는

\[
dX=Vd\hat t,
\qquad
\nabla_VV=-\gamma V-\nabla_{g_\theta}\Phi(X)
+\sqrt{2\gamma T_d}\,\dot W_g
\tag{G5}
\]

를 따른다. 마찰과 잡음은 open-system 유효항이며 single-copy 보존작용에서
유도됐다고 주장하지 않는다.

source와 metric이 고정되고 \(T_d=0\)일 때

\[
H(X,V)=\frac12\|V\|_{g_\theta}^2+\Phi(X)
\tag{G6}
\]

라 하자.

**[정리]** 결정론적 구간에서

\[
\frac{dH}{d\hat t}=-\gamma\|V\|_{g_\theta}^2\le0.
\tag{G7}
\]

국소 최소점의 basin을 둘러싼 가장 낮은 saddle energy를 \(\Phi_s\)라 할 때
basin 안에서

\[
H(X,V)<\Phi_s
\tag{G8}
\]

가 성립하면 궤적은 다른 행동 basin으로 넘어갈 수 없다.

**증명.** saddle을 넘으려면 potential energy가 최소 \(\Phi_s\)여야 하고
kinetic energy는 비음수다. (G7)에 의해 총에너지는 증가할 수 없으므로
(G8)과 모순이다. \(\square\)

**[정의]** 첫 (G8) 성립 시점을 `capture time`으로 한다. 이는 외부의
`b=b_0+kC`가 아니라 source가 만든 well과 saddle에서 계산되는 자유경계다.
\(T_d>0\)에서는 영구포획 정리가 아니라 metastable capture이며 escape를
별도로 측정해야 한다.

## 4. 갈등 지연의 장 유도

**[정리: 이진 대칭]** \(x\mapsto-x\) 대칭인 결정공간에서 행동원이
\(x_+=a\), \(x_-=-a\)에 있고 kernel과 metric도 대칭이라 하자.
\(m_+=m_-\), \(X(0)=V(0)=0\), \(T_d=0\)이면

\[
\nabla\Phi(0)=0,\qquad X(t)=0.
\tag{G9}
\]

반면 \(m_+\ne m_-\)이고 Green kernel의 중심 미분이 0이 아니면 일반적으로
더 큰 질량 쪽으로 즉시 가속한다.

**증명.** 같은 질량의 두 source가 만드는 potential은 even function이므로
원점 미분이 0이다. 질량이 다르면 odd force 성분이 남는다. \(\square\)

**[산출]** 고갈등 지연은 `갈등을 감지해 경계를 올리는 명령`이 아니라
대칭 source의 힘 상쇄다. 명확한 증거에서는 질량 비대칭이 커져 capture가
빨라진다. 잡음이 있는 완전대칭에서는 선택은 느리고 무작위여야 한다.

**[산출: 국소 Gaussian 검사]** 평탄한 이진 좌표에서

\[
\Phi(x)=-m_+e^{-(x-a)^2/(2\sigma^2)}
-m_-e^{-(x+a)^2/(2\sigma^2)}
\]

이면 중심의 힘은

\[
-\Phi'(0)
=\frac{a}{\sigma^2}e^{-a^2/(2\sigma^2)}(m_+-m_-).
\tag{G10}
\]

따라서 초기 가속도의 부호는 추가 threshold 없이 질량차의 부호와 정확히
일치한다.

## 5. CE/뇌 결합과 한계

| 요소 | 장 모형 역할 | 지위 |
|---|---|---|
| PFC output | \(q_a\), 증거질량 logit | 모델 선택 |
| MD | \(g_\theta\), 행동원 사이 유효거리 | 미완성 생물학적 사상 |
| Loop 8C residual | \(\varphi_a\), source 질량 보정 | 합성 output 기반 모델 선택 |
| BG/STN | 전역 마찰 \(\gamma\) 또는 screening \(\mu\) | 미완성 |
| CE 중력 구조 | source가 field를 만들고 field가 motion을 정함 | 조직 원리 |

실제 Einstein tensor를 인지공간에 쓰지 않는다. 인지다양체에는 물리적
stress-energy 보존과 Lorentzian spacetime 사상이 없기 때문이다. (G2)–(G5)는
screened Newtonian 유효축약이고 실제 중력 동일시는 삭제한다.

## 6. 무차원 감사

기준 길이 \(L_0\), 시간 \(\tau_0\), 에너지 \(E_0\)로 정규화한 뒤 hat을
생략한다.

| 코어 인자 | 판정 | 조건 |
|---|---|---|
| \((q_a+\eta_\varphi\varphi_a)/T_e\) | 통과 | 분자와 \(T_e\) 동일 척도 |
| \(\mu^2\Phi\), \(L_g\Phi\) | 통과 | \(\mu\)를 \(L_0^{-1}\)로 정규화 |
| \(\kappa(\rho-\bar\rho)\) | 조건부 통과 | (G3) 좌변 척도에 맞춘 \(\kappa\) |
| \(\sqrt{2\gamma T_d}\) | 조건부 통과 | 정규화 시간·속도 사용 |
| \(H<\Phi_s\) | 통과 | 동일 무차원 에너지 |

차원 정합은 신경생물학적 정당성을 증명하지 않는다.

## 7. 최소 반증 실험

Loop 8B/8C checkpoint는 고정한다. 행동점을 \(\pm1\)에 놓은 1차원 공간에서
(G3)를 사전 고정 grid로 풀고 (G5)를 적분한다.

비교군은 실패한 Loop 8D linear-STN, equal-budget DDM, gravity field,
source-mass shuffle, flat metric 대 MD metric이다.

**[예측]** 다음이 동시에 필요하다.

- equal-mass deterministic symmetry 오차가 허용오차 안에서 0;
- mass asymmetry와 초기 가속 방향이 모든 seed에서 일치;
- 명확한 증거일수록 capture time이 단조 감소;
- 고갈등 accuracy/utility가 Loop 8D와 equal-budget DDM을 paired LCB로 초과;
- source shuffle에서 효과 소실;
- capture 이후 escape/flip rate가 사전 상한 이하;
- PFC/MD/residual trace bit identity와 수치 안정성 통과.

grid 수렴성, kernel 폭, screening, 잡음과 capture tolerance는 결과 전에 별도
사전등록한다. 사후 선택하면 중력장 가설은 검증되지 않은 것이다.

# 억압 계수 \(\epsilon=e^{-1}\) 후보 분석

이 문서는 \(e^{-1}\)이 정확히 나타나는 서로 다른 수학 문제와 CE의 물리
계수를 동일시하는 선택을 분리한다.

## 1. 세 계수의 정의

**[정의]**

- \(\epsilon_*\): CE에서 선택할 후보 보편 억압 계수
- \(\epsilon_{\rm obs}\):
  \[
  \epsilon_{\rm obs}:=
  \frac{\Omega_{\Lambda,0}-\Omega_{m,0}}
       {\Omega_{\Lambda,0}+\Omega_{m,0}}
  \]
  단, \(\Omega_{\Lambda,0}+\Omega_{m,0}\ne0\)이다.
- \(\epsilon_{\rm mass}\): 특정 질량 모형에서
  \[
  m_{\rm eff}=m_0(1-\epsilon_{\rm mass})
  \]
  로 정의한 질량 억압 계수. 여기서 \(m_0\ne0\)은 기준 질량이다.

**[공리]**

\[
\epsilon_* = \epsilon_{\rm obs} = \epsilon_{\rm mass}
\]

라는 동일시는 정의나 아래 정리들의 결론이 아니라 강한 물리 사상이다.

## 2. Euclidean minimum의 조건부 산출

**[정리: 반고전 근사]**
유한차원 cutoff에서 양의 Hessian을 가진 고립 비퇴화 minimum의
Laplace 기여는

\[
K_j\simeq A_j e^{-S_{E,j}/\hbar}
\]

꼴이며 \(A_j\)에는 fluctuation determinant와 국소 measure 정규화가
들어간다.
\(\Delta S_E:=S_{E,2}-S_{E,1}\)로 두면 두 saddle의 비는

\[
\frac{K_2}{K_1}
\simeq
\frac{A_2}{A_1}
\exp\!\left[-\frac{S_{E,2}-S_{E,1}}{\hbar}\right].
\]

**[산출]** \(\Delta S_E/\hbar=1\)과 \(A_2/A_1=1\)을 별도로 가정하면
비는 \(e^{-1}\)이다. 일반 비에서는 determinant prefactor가 남는다.
[증명과 정의역](핵심_정리_증명.md#laplace-saddle)
영모드·gauge orbit·음의 모드가 있는 saddle에는 별도의 collective
coordinate, gauge fixing 또는 지정한 thimble이 필요하다.

## 3. 고전 secretary problem

**[정리]** 서로 다른 전순위를 가진 \(n\)개 후보가 균등 무작위 순열로
도착하고, 상대순위만 관측하며, recall 없이 한 번만 선택하는 고전
best-choice 문제를 생각하자. 처음 \(r\)개를 버리고 이후 첫 record를 택하면

\[
P_n(r)=\frac{r}{n}\sum_{j=r+1}^{n}\frac1{j-1}.
\]

\(r_n\)을 이 확률을 최대화하는 "처음 버리는 후보 수"라 하면 고전
secretary 정리에 의해

\[
\lim_{n\to\infty}\frac{r_n}{n}=\frac1e,
\qquad
\lim_{n\to\infty}P_n^{\rm opt}=\frac1e.
\]

이 정리는 해당 선택 문제의 결과다. 경로적분 억압이나 우주론 계수와의
동일시는 제공하지 않는다.
[증명](핵심_정리_증명.md#secretary-limit)

## 4. \(x^x\)의 최소점

**[정리]** \(x>0\)에서

\[
\frac{d}{dx}\log(x^x)=\log x+1
\]

이므로

\[
\operatorname*{arg\,min}_{x>0}x^x=e^{-1},
\qquad
\min_{x>0}x^x=e^{-1/e}.
\]

최소점의 위치를 CE 계수로 읽는 일은 별도 **[공리]**다.

## 5. logistic ODE

**[공리]** 유효 동역학을

\[
\dot\epsilon=r\epsilon(1-\epsilon)-k\epsilon,
\qquad 0<k<r
\]

로 둔다.

**[산출]** 고정점은

\[
\epsilon=0,
\qquad
\epsilon_*=1-\frac{k}{r}
\]

이다.

**[정리]** \([0,1]\)은 순방향 불변이고 모든
\(\epsilon(0)>0\)인 해는 양의 고정점으로 수렴한다. 명시적 해는

\[
\epsilon(t)=
\frac{\epsilon_*}
{1+(\epsilon_*/\epsilon(0)-1)e^{-(r-k)t}}.
\]

[증명](핵심_정리_증명.md#logistic-flow)

\(\epsilon_*=e^{-1}\)을 얻으려면

\[
\frac{k}{r}=1-e^{-1}
\]

을 별도로 선택해야 한다. logistic 형식이나 이 비율은 앞선 정리에서
도출되지 않는다.

## 6. 평탄 dust+\(\Lambda\) 우주의 조건부 산출

**[공리]** 공간적으로 평탄하고 현재 성분이 pressureless matter와
cosmological constant뿐이라고 두면

\[
\Omega_{m,0}+\Omega_{\Lambda,0}=1.
\]

**[산출]** 위 정의와 평탄 closure로부터

\[
\Omega_{\Lambda,0}=\frac{1+\epsilon_{\rm obs}}2,
\qquad
\Omega_{m,0}=\frac{1-\epsilon_{\rm obs}}2.
\]

또한 \(a_0=1\), \(H_0>0\)인 expanding Big-Bang branch에서 GR
Friedmann 식

\[
E(a)^2=\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}
\]

을 쓰고, dust가 분리 보존되며 \(\Lambda\)가 상수라고 둔다.
\(0<\Omega_{m,0},\Omega_{\Lambda,0}<1\)이면 표준 FLRW 적분으로

\[
H_0t_0=
\frac{2}{3\sqrt{\Omega_{\Lambda,0}}}
\operatorname{arsinh}
\sqrt{\frac{\Omega_{\Lambda,0}}{\Omega_{m,0}}}
\]

을 얻는다. 즉

\[
H_0t_0=
\frac{2}{3\sqrt{(1+\epsilon_{\rm obs})/2}}
\operatorname{arsinh}
\sqrt{\frac{1+\epsilon_{\rm obs}}{1-\epsilon_{\rm obs}}}.
\]

\(\epsilon_{\rm obs}\)를 자료로 정했다면 이 식들은 조건부 재표현이지 독립
우주론 예측이 아니다.
[증명](핵심_정리_증명.md#dust-lambda-age)

## 7. 남은 과제 **[미완성]**

- \(\epsilon_*\)를 산출하는 공변 작용과 경계조건
- saddle prefactor를 포함한 실제 진폭비
- \(\epsilon_*\), \(\epsilon_{\rm obs}\), \(\epsilon_{\rm mass}\) 동일시의 동역학
- dust+\(\Lambda\)를 넘어선 섭동·성장률과 독립 likelihood
- continuum 경로공간의 topology, prior support와 recovery 조건

경로공간 정리를 적용하기 위한 정확한 조건과 no-go는
[pathspace closure checklist](../9_등호이전/05c_pathspace_closure_checklist.md)와
[action/topology package](../9_등호이전/05f_CE_action_topology_package.md)에 둔다.

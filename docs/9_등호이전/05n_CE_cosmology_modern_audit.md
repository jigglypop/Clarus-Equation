# 05n. CE 우주론 연결의 형식 경계

## 0. 범위

유한 Gibbs·Gamma 계산은 그 자체로 우주론 forward model이 아니다. 이
문서는 보존 가능한 고정점 수학, 관측값 대입의 지위와 실제 우주론에 필요한
닫힘 조건을 분리한다.

## 1. 지수 고정점의 완전한 구조

**[정의]** 무차원 \(D\geq0\)에 대해
\[
x=e^{-D(1-x)},\qquad x\in(0,1]
\]
를 생각한다.

**[정리]**

1. \(0\leq D\leq1\)이면 해는 \(x=1\) 하나뿐이다.
2. \(D>1\)이면 \(x=1\) 외에 유일한 해
   \(q(D)\in(0,1/D)\)가 하나 더 있다.
3. \(D>1\)에서 \(q(D)\)는 반복
   \(x_{n+1}=e^{-D(1-x_n)}\)의 국소 안정 고정점이고 \(x=1\)은
   불안정하다.

**증명.** 고정점 식은
\[
h_D(x):=\log x+D(1-x)=0
\]
과 동치다. \(h_D''(x)=-x^{-2}<0\), \(h_D(1)=0\)이고
\(h_D(x)\to-\infty\) as \(x\downarrow0\)다.

\(D\leq1\)이면 \(h_D'(x)=x^{-1}-D\geq0\)이므로 해는 1뿐이다.
\(D>1\)이면 유일한 최대점 \(x=1/D\)에서
\[
h_D(1/D)=D-1-\log D>0
\]
이고 strict concavity로 \((0,1/D)\)에 해가 정확히 하나 있다.
반복함수 \(F_D(x)=e^{-D(1-x)}\)의 도함수는
\(F_D'(x)=DF_D(x)\)이므로 고정점에서 \(F_D'(x)=Dx\)다.
\(Dq(D)<1\), \(F_D'(1)=D>1\)이어서 안정성 결론이 따른다.
\(\square\)

**[산출]** 특정 무차원 \(D\)를 넣어 \(q(D)\)를 수치로 계산하는 것은 이
정리의 산출이다. \(D\)의 물리값, \(q(D)\)를 물질 분율이나 측정확률에
대응시키는 단계는 각각 **[공리: 외부 입력]**과
**[공리: 물리 사상]**이다.

## 2. Fixed-point 수치와 우주론 파라미터

다음 세 문장은 서로 다른 층이다.

1. **[정리]** 1절 정의역에서 고정점의 개수와 안정성이 결정된다.
2. **[산출]** 선택한 \(D\)에서 수치해 \(q(D)\)를 구할 수 있다.
3. **[미완성]** \(q(D)\)가 특정 우주론 밀도분율이라는 물리 사상.

세 번째 문장에는 적어도

\[
q(D)\longmapsto
\varepsilon_i(a),\ p_i(a),\
T_{\mu\nu}^{(i)}
\]

를 주는 공변 모형이 필요하다. 한 무차원 수가 관측 중심값과 가깝다는
사실만으로 이 사상이 정의되지는 않는다.

## 3. Mode benchmark의 지위

[05m_CE_mode_decomposition_audit.md](05m_CE_mode_decomposition_audit.md)의
독립 Gamma benchmark는
\[
\Phi_N\sim
\operatorname{Gamma}\!\left(
\frac N2,\frac{2m}{N}
\right)
\]
를 **[공리: 모델 선택]**으로 두었을 때
\[
\frac{\mathbb Ee^{-\Phi_N}}{e^{-m}}
=
\exp\!\left[
m-\frac N2\log\!\left(1+\frac{2m}{N}\right)
\right]
\]
을 정확히 준다.

허용 상대오차 \(\delta\)를 외부 자료에서 택하면
\[
m-\frac N2\log\!\left(1+\frac{2m}{N}\right)
\leq\log(1+\delta)
\]
는 이 benchmark 안의 조건부 **[산출]**이다. 이는 임의의 \(\Phi\)
분포에 대한 보편적인 mode 수 하한이 아니다. 특히 평균 \(m\)만으로
분산, 상관 또는 \(N\)을 식별할 수 없다.

## 4. 우주론 forward model의 최소 자료

고정점이나 residual 분율을 우주론과 비교하려면 다음 연쇄를 닫아야 한다.

### 4.1 배경

**[공리: 우주론 모형]** 무차원 scale factor \(a\), 성분별 에너지 밀도
\(\varepsilon_i(a)\), 압력 \(p_i(a)\), 곡률상수
\(\kappa\) (\([\kappa]=L^{-2}\))와
\(H_0\)를 지정하면
\[
H^2(a)
=
\frac{8\pi G}{3c^2}\sum_i\varepsilon_i(a)
-\frac{\kappa c^2}{a^2}
\]
를 풀 수 있다. CE 성분을 넣으려면 그 보존식 또는 상호작용 source를
함께 정해야 한다.

### 4.2 섭동

거리 관측만으로 성장과 lensing이 정해지지 않는다. 적어도
\[
\delta_i,\quad
c_{s,i}^2,\quad
\pi_i,\quad
\mu(a,k),\quad
\eta(a,k)
\]
중 선택한 이론에 필요한 섭동 closure를 제시해야 한다. Background
\(H(a)\)만 맞춘 결과를 structure-growth 예측으로 읽지 않는다.

### 4.3 초기우주와 관측연산자

CMB 또는 BBN 비교에는 recombination, photon--baryon sound horizon,
neutrino sector와 primordial initial condition이 필요하다. BAO, SNe,
lensing, growth 자료에는 각각의 observable map, nuisance parameter와
공분산을 포함한 likelihood가 필요하다.

이 자료 없이 한 density ratio를 여러 관측량의 동시 예측으로 확장하지
않는다.

## 5. 조건부 forward 계산

[05o_CE_residual_cosmology_forward_model.md](05o_CE_residual_cosmology_forward_model.md)
은 평탄 FLRW와 선택한 유효 방정식상태 아래 \(H(z)\), 거리와 선형성장을
계산하는 조건부 모형이다. 그 계산식이 정확하다는 것과 CE가 그
방정식상태를 유도한다는 것은 별개다.

관측 비교의 형식은
\[
\Delta O=O^{\rm model}-O^{\rm obs},
\qquad
\chi^2=\Delta O^{\mathsf T}C^{-1}\Delta O
\]
로 쓸 수 있지만, 자료 vector \(O^{\rm obs}\), covariance \(C\),
selection function과 nuisance marginalization은 **[경험식]** 입력이다.

## 6. 현재 보존되는 것과 남은 것

보존되는 수학:

- \(x=e^{-D(1-x)}\)의 정의역별 해 개수와 안정성
- 지정 Gamma-mode 모형의 정확한 Laplace transform
- 지정 FLRW closure 아래의 배경·거리·성장 forward equation

남은 물리 사상:

- \(D\)와 microscopic action의 관계
- 고정점 \(q(D)\)와 stress-energy 성분의 관계
- CE 성분의 배경·섭동 방정식과 초기조건
- 독립 likelihood에 대한 사전 고정 예측

관측 불일치가 조건부 고정점·Gamma 정리를 거짓으로 만들지는 않는다.
반대로 그 정리의 참이 물리 사상을 증명하지도 않는다.

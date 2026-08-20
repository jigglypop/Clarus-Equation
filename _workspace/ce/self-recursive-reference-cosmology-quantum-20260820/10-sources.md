# 무한 자기재귀 참조함수 1차 출처 원장

Status: COMPLETE  
Date: 2026-08-20

## 1. 분지과정

Galton과 Watson의 원 논문은 family extinction의 세대 재귀 문제를
도입했다. 현대 표기로 필요한 핵심은 offspring 확률생성함수 (G)에 대해

\[
q_n=G^{\circ n}(0),\qquad q_{\rm ext}=\lim_{n\to\infty}q_n
\]

이고, 이 극한이 (G(q)=q)의 최소 비음근이라는 점이다. 따라서
고정점 방정식만 적는 것보다 초기값 (0)과 합성열을 함께 적어야 한다.

- G. Watson and F. Galton, “On the Probability of the Extinction of
  Families,” *Journal of the Anthropological Institute* **4** (1875),
  138--144, [DOI](https://doi.org/10.2307/2841222),
  [원문 스캔](https://galton.org/cgi-bin/searchImages/search/essays/pages/galton-1874-jaigi-family-extinction_2.htm).
- S. Lalley, *Branching Processes*, University of Chicago lecture notes,
  [PDF](https://galton.uchicago.edu/~lalley/Courses/312/Branching.pdf).
  이 자료는 (P(Z_n=0)=G^{\circ n}(0)), 단조극한과 최소 고정점 정리를
  현대 표기로 명시한다. 동료심사 논문이 아닌 강의노트라는 출처 등급은
  유지한다.

## 2. 유한차원 양자채널

유한차원 CPTP 채널 \(\mathcal E\)의 고정상태 존재, 고정상태의 유일성,
모든 초기상태에서의 mixing은 서로 다른 명제다. 반복
\(\mathcal E^n(\rho_0)\)의 수렴에는 고정점 부분공간뿐 아니라 주변
스펙트럼을 검사해야 한다. 특히 고정상태 외에 절댓값 1인 고유값이 있으면
비감쇠 진동이 남을 수 있다. unitary conjugation은 CPTP이면서도 일반적으로
mixing이 아닌 직접 반례다.

- D. Burgarth et al., “Ergodic and Mixing Quantum Channels in Finite
  Dimensions,” *New Journal of Physics* **15**, 073045 (2013),
  [arXiv:1210.5625](https://arxiv.org/abs/1210.5625),
  [DOI](https://doi.org/10.1088/1367-2630/15/7/073045).

이 출처는 양자채널 반복의 fixed point, invariant subspace, peripheral
spectrum과 mixing 조건을 다룬다. 이는 CE의 특정 quantum-to-branching
사상을 승인하는 출처가 아니라, 그 사상 전에 통과해야 할 표준 기준선이다.

## 3. 우주론 동역학

우주론에서 late-time attractor는 물리 방정식에서 유도한 autonomous flow의
고정점과 선형 안정성을 뜻한다. 예를 들어 scalar field와 barotropic fluid
모형에서는 Friedmann 제약, 보존식, scalar 방정식과 무차원 정규화 변수를
먼저 정한 뒤 parameter 영역별 fixed point와 안정성을 분류한다. 이는
임의의 계산 반복 (x_{n+1}=F(x_n))을 우주 시간으로 읽는 것과 다르다.

- E. Copeland, A. Liddle and D. Wands, “Exponential potentials and
  cosmological scaling solutions,” *Physical Review D* **57**, 4686--4690
  (1998), [arXiv:gr-qc/9711068](https://arxiv.org/abs/gr-qc/9711068),
  [DOI](https://doi.org/10.1103/PhysRevD.57.4686).

이 논문은 특정 scalar--fluid 모형의 1차 출처다. 결과를 임의의 CE 모형에
그대로 이식하지 않으며, 정당한 cosmological fixed-point 분석에 필요한
변수·제약·방정식·parameter 영역의 기준으로만 사용한다.

## 4. 출처에서 고정되는 공통 게이트

다음 다섯 문장은 별개로 검사해야 한다.

1. (T)의 고정점이 존재한다.
2. 여러 고정점 중 어떤 해를 고르는 selection rule이 있다.
3. 선택된 영역에서 고정점이 유일하다.
4. 지정 초기값의 반복 또는 물리 흐름이 그 고정점으로 수렴한다.
5. 그 고정점이 해당 물리계의 상태·관측량으로 admissible하다.

Galton--Watson 계열은 `초기값 0 → 최소근`이라는 선택 규칙을 제공한다.
양자채널은 주변 스펙트럼 조건, 우주론은 작용/보존식에서 유도한 flow와
제약면 안정성이 추가로 필요하다. `self-reference`라는 이름만으로 다섯
게이트 어느 것도 자동 통과하지 않는다.

## 5. 신경사건 분지와 point-process

신경 avalanche를 branching process로 근사하는 출발점은 관측된 전극
event의 후속 event 수다. 이는 raw signed synaptic weight를 offspring
matrix로 읽는 근거가 아니다.

- J. Beggs and D. Plenz, “Neuronal Avalanches in Neocortical Circuits,”
  Journal of Neuroscience 23, 11167--11177 (2003),
  [DOI](https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003).
  이 논문은 avalanche 통계가 임계 branching approximation과 양립함을
  보고한다. 개별 causal parent assignment나 \(A=|W|\)를 제공하지 않는다.
- A. Corral López, V. Buendía and M. Muñoz,
  “Excitatory-inhibitory branching process,” Physical Review Research 4,
  L042027 (2022),
  [DOI](https://doi.org/10.1103/PhysRevResearch.4.L042027).
  억제 node를 명시적으로 포함하면 단일 비음수 one-type branching에는
  없는 phase가 생긴다. inhibition을 음의 offspring로 넣는 단순화는
  허용되지 않는다.
- A. Hawkes, “Spectra of some self-exciting and mutually exciting point
  processes,” Biometrika 58, 83--90 (1971),
  [DOI](https://doi.org/10.1093/biomet/58.1.83).
- P. Brémaud and L. Massoulié, “Stability of Nonlinear Hawkes Processes,”
  Annals of Probability 24, 1563--1588 (1996),
  [DOI](https://doi.org/10.1214/aop/1065725193).
  multivariate history kernel과 nonlinear intensity의 안정성은 kernel
  적분 또는 Lipschitz majorant의 spectral bound를 요구한다. 무한 kernel
  support는 무한 memory를 가질 수 있으므로 runtime tick만으로 유한차원
  Markov state가 자동 생기지 않는다.

따라서 뇌 bridge는 두 route를 구분한다.

1. 독립 Poisson offspring genealogy가 실제로 정의된 경우의 정확한
   extinction fixed point
2. signed E/I history를 가진 nonlinear Hawkes intensity와 그 안정성 gate

두 route는 추가 조건 없이 같은 식이 아니다.

## 6. 양자 instrument에서 고전 기록으로

- E. Davies and J. Lewis, “An Operational Approach to Quantum Probability,”
  Communications in Mathematical Physics 17, 239--260 (1970),
  [DOI](https://doi.org/10.1007/BF01647093).

양자층과 BrainRuntime 사이의 허용 경계는 instrument outcome이다.

\[
p(y\mid\rho)=\operatorname{Tr}\mathcal I_y(\rho),\qquad
\rho_y=\frac{\mathcal I_y(\rho)}{p(y\mid\rho)},\qquad
\sum_y\mathcal I_y\ \text{trace preserving}.
\]

뇌 입력은 \(u=\psi(y)\) 같은 고전 record feature다. nonselective CPTP
channel만 주어지면 outcome genealogy가 없으며, 서로 다른 instrument가
같은 nonselective channel을 만들 수 있으므로 branching matrix는
식별되지 않는다.

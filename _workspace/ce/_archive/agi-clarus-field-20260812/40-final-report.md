# 클라루스장: 유계 뇌–우주 장 연구 primitive의 형식화와 구현

Status: COMPLETE

최종 판정: **조건부 수학 정리와 그 정리를 따르는 연구용 baseline 구현은 완료했다. 전체 $p^*$ 자기수렴, 생물학적 뇌 모사, 우주 물리 이론, AGI 달성 주장은 성립하지 않는다.**

## 1. 초록

[모델 선택] 이 연구는 유한 그래프의 국소 기억, 경성 사건 게이트, 감쇠·확산 스칼라장을 결합해 뇌의 선택적 기억과 우주의 공간적 전파를 하나의 계산 모형으로 표현한다. [조건부 정리] 유계 Lipschitz readout, 메모리–장 분리, 단위공 쓰기, 경성 게이트 아래에서 기억과 장의 전역 유계성, 장의 비음성, 닫힌 게이트의 항등성, 사건 단위 안정성이 성립한다. [조건부 정리] i.i.d. 외생 게이트와 공통 재생 쓰기 및 비원자 임계 조건을 더하면 세 상의 시간평균 점유율이 초기조건과 무관한 극한으로 수렴한다. [예측] 목표 점유율 $p^*=(0.0487077,0.2623,0.6891)$ 전체의 자기수렴은 현재 장에서 나오지 않으며, 고정 임계 toy는 활성률이 외생 입력률을 추적했다. [구현 산출] 정확한 상수-source 장 적분, 경성 latch, 단위공 투영, 무차원 상 판독, bounded HRR readout을 구현했고 관련 검사 묶음에서 17, 28, 71, 34, 72개 검사가 각각 통과했다. [한계] 이 결과는 연구 primitive의 소프트웨어·수학 검증이며 생물학, 우주론 또는 AGI의 경험적 검증이 아니다.

## 2. 서론

이 절은 클라루스장이 무엇을 통합하고 무엇을 통합하지 않는지 정한다. [모델 선택] 그래프 노드의 벡터 상태는 국소 기억을 나타내고, 스칼라장 $phi$는 그래프를 따라 확산하며 감쇠하는 전역 조절량을 나타낸다. 신경계에서는 선택적 쓰기, 안정한 latch, 확산성 조절 신호에 대한 계산적 유비로 읽을 수 있다. 우주 프레임워크에서는 국소 자유도와 공간 결합장의 유비로 읽을 수 있다. 두 대응은 모두 [물리 사상 가설]이며, 실제 뉴런·시냅스·시공간의 법칙이라는 주장이 아니다.

[산출] 선행 V14가 보인 무손실 슬롯과 bilinear readout의 충분성은 상태 저장과 결합 판독을 분리할 근거를 준다. 그러나 V14 route L 원형은 상태 의존 HRR 쓰기와 항상 양수인 sigmoid 게이트를 사용한다. 1차원에서 그 갱신은 $h^+=(1+gv)h$가 될 수 있으므로 지수 발산 반례를 갖는다. 따라서 본 연구는 route L 원형을 상속하지 않고, 경성 게이트와 유계 쓰기를 갖는 별도 baseline을 구성한다.

[형식 지위] 본 결과의 핵심은 CF-1, CF-2, CF-3, CF-5의 조건을 명확히 한 데 있다. [예측] 세 상의 특정 비율 $p^*$는 정리로 승격하지 않는다. [구현 산출] 코드는 이 구분을 공개 인증서에도 기록한다.

## 3. 정의와 표기

이 절은 정리의 정의역과 모든 핵심 기호를 고정한다. $G=(V,E,W)$는 $N<\infty$개 노드를 가진 비음 가중치의 대칭 연결 그래프다. $L$은 $G$의 정규화 라플라시안이고 $L\succeq0$이다. 노드 $i$의 기억은 $s_i\in\mathbb R^w$, 장은 $\phi_i\in\mathbb R_{\ge0}$이다. 모든 구현 변수는 기준 척도로 정규화된 무차원량이다.

유계 readout $r:\mathbb R^w\to[0,R]$은 식 (1)로 고정한다.

$$
r(s_i)=\min(\lVert s_i\rVert_2,R).
\tag{1}
$$

식 (1)은 비음이고 1-Lipschitz이며 $R$로 유계다. 장의 연속시간 방정식은 식 (2)다.

$$
\dot\phi=-(\kappa L+\lambda I)\phi+r(s),
\qquad \kappa\ge0,\quad \lambda>0.
\tag{2}
$$

여기서 $kappa$, $lambda$, tick 길이 $Delta t$, $R$은 정규화된 무차원 설정값이다. 기억의 이산 갱신은 식 (3)이다.

$$
\hat g_i=g_i\mathbf 1[g_i>\theta_g],
\qquad
s_i^+=(1-\hat g_i)s_i+\hat g_i\Pi_{B_1}(\tilde s_i).
\tag{3}
$$

$g_i\in[0,1]$는 쓰기 점수, $	heta_g\in[0,1]$는 경성 임계, $Pi_{B_1}$은 닫힌 단위공 투영이다. 구현은 $hat g_i=0$인 행을 산술 갱신하지 않고 그대로 복사한다.

[정의] 상 지표는 active, structural, frozen의 서로소 삼분할이다. active는 $hat g_i>0$인 경우다. 비활성 노드는 무차원 장 점수 $lambda\phi_i/R$가 $	heta_s$보다 크면 structural이고, 아니면 frozen이다. 시간 창의 세 상 분율을 $pi=(\pi_A,\pi_S,\pi_F)$로 쓰며 $pi_A+\pi_S+\pi_F=1$이다.

[정의] prediction-error 점수는 기준 척도 $x_0>0$를 사용해 식 (4)처럼 구성할 수 있다.

$$
g_i=\sigma\!\left(a\left\lVert\frac{x_i-\hat x_i}{x_0}\right\rVert_2^2+b\right).
\tag{4}
$$

$a,b$와 식 (4)의 sigmoid 인자는 무차원이다. 식 (4)는 부호 반전에 불변인 연성 점수만 만든다. 실제 latch의 열림 여부는 식 (3)의 경성 임계가 정한다.

## 4. 공리

이 절은 증명에 필요한 선택을 숨기지 않고 분류한다.

1. **D1 — 유계 스칼라 readout [모델 선택].** 벡터 기억은 식 (1)의 비음·유계·Lipschitz 스칼라를 통해서만 장을 구동한다.
2. **S1 — 상태와 장의 분리 [모델 선택].** 라플라시안은 $phi$에만 작용하며 $s$ 갱신에는 직접 들어가지 않는다.
3. **S2 — 게이트 경유 유계 쓰기 [모델 선택].** 기억 쓰기는 식 (3)으로만 일어나며 후보는 단위공에 속한다.
4. **S3 — 경성 게이트 [모델 선택].** $g_i\le\theta_g$이면 유효 게이트가 정확히 0이다.
5. **A-E1 — 외생 입력 [확률 공리].** 입력열 $(x_t)$은 i.i.d.다.
6. **A-E2R — 공통 재생 쓰기 [확률 공리].** 게이트와 쓰기값 $	ilde s_i(t)=\psi_i(x_t)$는 현재 $(s_t,\phi_t)$와 무관하다. 노드마다 열림 확률 $ho_i>0$이고 열릴 때 $hat g_i\ge g_{\min}>0$이다. 같은 입력으로 결합한 두 궤적은 같은 쓰기값을 받는다.
7. **A-E3 — 비원자 임계 [확률 공리].** 정상 법칙은 $g=\theta_g$와 $lambda\phi/R=\theta_s$ 경계에 질량을 두지 않는다.
8. **N1 — 공통 현재 잡음 [확률 공리].** CF-5에서 현재 잡음의 법칙은 신호/노이즈 계급과 신호 부호에 조건부 동일하고 현재 부호와 독립이다.
9. **N2 — 예측가능한 적응 임계 [확률 공리].** 적응 임계는 현재 표본을 보기 전의 과거 정보로 정해진다.

A-E1–A-E3은 전체 구현이 자동으로 보장하는 성질이 아니다. [구현 산출] 인증서는 CF-3의 유효 범위를 `iid-exogenous-common-write+nonatomic-thresholds`로 명시한다.

## 5. 정리와 증명

이 절은 유계성, 사건 스케줄 안정성, 조건부 점유율 평형, 선형 게이트의 한계를 증명한다.

### 5.1 CF-1: 결합계의 유계성과 양성

**정리 CF-1 [조건부 정리].** D1과 S1–S3 아래에서 유계 입력과 $phi(0)\ge0$에 대해 결합계의 해는 모든 유한 시간에 유일하게 정해진다. 기억은 식 (5), 장은 식 (6)의 전역 경계를 만족하고, 장의 비음성이 보존되며, 닫힌 게이트의 기억은 항등 갱신을 한다.

$$
\lVert s_i(t)\rVert_2\le
\max\{\lVert s_i(0)\rVert_2,1\}.
\tag{5}
$$

$$
\lVert\phi(t)\rVert_2\le
\max\left\{\lVert\phi(0)\rVert_2,\frac{\sqrt N R}{\lambda}\right\}.
\tag{6}
$$

**증명.** $A=\kappa L+\lambda I$라 놓으면 $A\succeq\lambda I$다. 한 tick 동안 $s$가 고정될 때 변분상수법은 식 (7)을 준다.

$$
\phi(t+h)=e^{-Ah}\phi(t)+A^{-1}(I-e^{-Ah})r(s(t)).
\tag{7}
$$

$\lVert e^{-Ah}\rVert_2\le e^{-\lambda h}$이고 $\lVert r(s)\rVert_2\le\sqrt N R$이므로 식 (7)을 반복하거나 적분형에 적용하면 식 (6)이 나온다. $-A$의 비대각 원소는 비음이므로 $-A$는 Metzler 행렬이다. 따라서 $e^{-Ah}$는 성분별 비음이며, $r(s)\ge0$과 함께 $phi(t)\ge0$를 보존한다. 기억 갱신은 현재 상태와 단위공 원소의 볼록결합이므로 식 (5)가 귀납적으로 성립한다. 닫힌 게이트에서는 $hat g_i=0$이므로 수학적으로 $s_i^+=s_i$이고, 구현은 기존 행을 직접 복사하므로 비트 패턴도 보존한다. □

[경계] 식 (6)은 2-노름 인증서다. 정규화 라플라시안의 행합을 사용하는 일반적인 $\ell_\infty$ 증명은 성형 그래프 중심에서 성립하지 않는다.

### 5.2 CF-2: 사건 단위 안정성

**정리 CF-2 [정리].** 닫힘 tick이 등거리이고 열림 tick의 후보가 단위공에 있으며 $g\ge g_{\min}>0$이면, 동일 스케줄의 두 궤적 사이 오차는 열림 사건 수 $N(t)$에 대해 식 (8)을 만족한다.

$$
\lVert e(t)\rVert
\le(1-g_{\min})^{N(t)}\lVert e(0)\rVert
+\varepsilon+\frac{\bar\eta}{g_{\min}}.
\tag{8}
$$

$\varepsilon$은 두 쓰기 후보의 최대 차이고, $\bar\eta$는 열림 사건마다 추가되는 결함의 상계다.

**증명.** 닫힘 tick은 오차 노름을 보존한다. 열림 사건 번호를 $k$로 쓰면 삼각부등식과 볼록 갱신으로

$$
e_{k+1}\le(1-g_{\min})e_k+g_{\min}\varepsilon+\bar\eta
\tag{9}
$$

를 얻는다. 식 (9)의 기하급수 합을 계산하면 식 (8)이 나온다. 후보 차이만 있고 사건 결함이 없으면 $\sup_t\lVert e(t)\rVert\le\max\{\lVert e(0)\rVert,\varepsilon\}$이므로 시간 길이와 열림 빈도에 무관한 비확대가 성립한다. □

**따름정리 CF-2L [조건부 정리].** 열림이 현재 오차와 독립인 Bernoulli$(\rho)$이고, 열림과 닫힘 오차가 각각

$$
e_{t+1}=(1-g)e_t+g\varepsilon+\eta_o,
\qquad
e_{t+1}=e_t+\eta_c
\tag{10}
$$

로 정확히 갱신되면 정상 평균은 식 (11)이다.

$$
\mathbb E[e_\infty]
=\varepsilon+\frac{\eta_o}{g}
+\frac{(1-\rho)\eta_c}{\rho g}.
\tag{11}
$$

**증명.** 두 사건에 대한 조건부 평균을 취하면 계수 $1-\rho g$인 일차 재귀가 되고, 그 고정점을 풀면 식 (11)이 나온다. □

[경계] $1/\rho$ 발산은 식 (10)의 독립성, 갱신 순서, 비음 결함을 둔 스칼라 모형의 정리다. 임의 스케줄에서는 평균 빈도만으로 긴 닫힘 간격을 제어하지 못하므로 일반 장의 정리로 확장하지 않는다.

### 5.3 CF-3: 조건부 점유율 평형

**정리 CF-3 [조건부 정리].** CF-1의 조건과 A-E1–A-E3 아래에서 시간평균 점유율 $pi(T)$은 초기조건과 무관한 극한 $\bar\pi$로 거의 확실하게 수렴한다.

**증명.** 같은 i.i.d. 입력열로 두 초기조건을 결합한다. A-E2R 때문에 닫힘 tick에서는 상태 차이가 보존되고 열림 tick에서는 같은 $\psi_i(x_t)$가 쓰이므로

$$
\lVert s_i(t)-s_i^\dagger(t)\rVert
\le(1-g_{\min})^{N_i(t)}
\lVert s_i(0)-s_i^\dagger(0)\rVert.
\tag{12}
$$

대수의 법칙으로 $N_i(t)/t\to\rho_i>0$이므로 식 (12)는 0으로 간다. 장의 차이는 감쇠율 $lambda$를 가진 선형 필터가 Lipschitz 입력 차이로 구동되는 식이므로 역시 0으로 간다. 유계 불변집합에서 과거로부터의 공통 재생 사건을 사용한 반복은 초기집합의 지름을 0으로 보내며, i.i.d. 입력의 가측 함수인 유일 정상 해를 만든다. 정상 과정은 i.i.d. shift의 factor이므로 에르고딕이다. Birkhoff 정리와 A-E3을 적용하면 상 지표의 시간평균도 모든 초기조건에서 같은 $\bar\pi$로 수렴한다. □

**반례 CF-3C [반례].** 외생 게이트만으로 CF-3은 성립하지 않는다. 허용된 쓰기 후보를 $\tilde s_i=s_i$로 두면 게이트가 열려도 $s_i^+=s_i$다. 초기값 0과 1은 영원히 분리된다. $r(s)=s$이면 두 장은 각각 0과 $1/\lambda$로 수렴한다. $0<\theta_s<1/\lambda$에서 structural/frozen 점유율은 초기조건에 따라 달라진다. 따라서 공통 재생 쓰기 또는 전체 결합 사상의 별도 joint contraction 조건이 필요하다. □

[미완성] 게이트나 쓰기값이 현재 $(s,\phi)$에 내생적으로 의존하는 완전 결합계의 정상 법칙 존재·유일성과 초기조건 독립성은 증명하지 않았다.

### 5.4 CF-5: 선형 게이트의 부호 대칭 한계

**정리 CF-5 [조건부 정리].** 신호가 $x=z\xi+\eta$, $z\in\{-1,+1\}$이고 노이즈가 $x=\eta$이며 N1이 성립한다고 하자. 선형 게이트 $u^\top x>c$의 양·음 신호 검출률을 $p_+,p_-$, 노이즈 열림률을 $q$라 하면 식 (13)이 성립한다.

$$
\min(p_+,p_-)\le q\le\max(p_+,p_-).
\tag{13}
$$

**증명.** $m=u^\top\xi$와 생존함수 $S(v)=\Pr(u^\top\eta>v)$를 두면 $p_\pm=S(c\mp m)$이고 $q=S(c)$다. $c$는 $c-|m|$와 $c+|m|$ 사이에 있고 $S$는 비증가이므로 식 (13)이 나온다. □

**따름정리 CF-5A [조건부 정리].** 정적 임계에서 $p_+,p_-\ge1-\delta$와 $q\le\delta$를 동시에 요구하면 $delta\ge1/2$다. N2까지 만족하는 적응 임계에서는 시간평균에 대해 $\bar q\ge\bar p_++\bar p_--1$이고 같은 요구가 $delta\ge1/3$을 강제한다.

[반례] 잡음 법칙이 신호 부호에 의존하면 식 (13)은 깨질 수 있다. $u=\xi=1$, $c=0$에서 $z=+1$의 잡음을 $-0.5$, $z=-1$의 잡음을 2로 두고 노이즈에 그 혼합분포를 사용하면 $p_+=p_-=1$이지만 $q=1/2$다. 따라서 N1과 N2는 정리의 필수 범위다.

## 6. 스칼라 고정점과 세 상 목표의 지위

이 절은 증명된 상수 하나와 경험적으로 놓인 두 성분을 분리한다. $A_d=4/(e\pi)^{4/3}$와 $D_{\mathrm{eff}}=3+A_d(1-A_d)=3.1766443715$를 외부 입력으로 두면 식 (14)의 내부 고정점이 존재한다.

$$
B_a(a)=e^{-D_{\mathrm{eff}}(1-a)},
\qquad a=B_a(a).
\tag{14}
$$

**정리 CF-$a$ [조건부 정리].** $D_{\mathrm{eff}}>1$에서 식 (14)은 경계 고정점 1 외에 유일한 내부 고정점 $a^*=0.0487077473\ldots$을 가지며, $|B_a'(a^*)|=D_{\mathrm{eff}}a^*=0.154727\ldots<1$이다.

**증명.** $F(a)=B_a(a)-a$는 엄격 볼록이고 $F(0)>0$, $F(1)=0$이다. $F'(0)=D_{\mathrm{eff}}e^{-D_{\mathrm{eff}}}-1<0$이고 $F'(1)=D_{\mathrm{eff}}-1>0$이므로 $F$는 $(0,1)$에서 한 번 내려갔다가 올라가며 경계근 1 이전에 정확히 한 내부근을 갖는다. 내부근에서의 도함수는 식 (14)를 사용해 $B_a'(a^*)=D_{\mathrm{eff}}a^*$이고 수치가 1보다 작다. □

[경험식 공리] 나머지 성분 $(0.2623,0.6891)$은 현재 무입력 유도로 나오지 않는다. [삭제] 이전 3-simplex 사상은 실제 고정점이 $(0.048708,0.796565,0.154727)$이고 목표와 최대노름 거리 0.5344이므로 활성 정본에서 제거했다. [예측] 전체 $p^*$ 자기수렴은 추가 적응 메커니즘과 사전등록 killing test가 필요한 가설이다. 고정 임계 toy에서 입력 신호율 $0.049,0.120,0.300$은 활성 점유율 $0.0488,0.1207,0.2994$로 그대로 전파됐다.

[GO-후보] 별도 경로 R-A1은 식 (14)의 잔차를 임계 적응에 사용해 $a^*$를 set-point 상수 없이 재현했다. 이 경로는 내생 임계를 도입해 A-E2R을 깨므로 현재 baseline과 CF-3 증명에는 포함하지 않는다. [가설] Poisson binding 그래프의 비거대 성분을 active와 동일시하는 R-A3도 별도 물리 사상 검증이 필요하다.

## 7. 구현 산출

이 절은 정리에서 코드로 옮긴 범위만 기록한다. [구현 산출] `ClarusField`는 유한 대칭 연결 그래프를 검사하고, $A=\kappa L+\lambda I$의 고유분해로 식 (7)의 상수-source 정확 tick을 계산한다. [구현 산출] 기억 후보는 단위공에 투영되며 닫힌 행은 직접 복사된다. [구현 산출] circular convolution은 `bounded_hrr_bind`에서 단위공 readout으로만 계산되고 recurrent 상태 전이에는 들어가지 않는다.

[모델 선택] 기본값은 $lambda=0.25$, $kappa=1$, $Delta t=1$, $R=1$, $	heta_g=0.5$, $	heta_s=0.25$다. 이 값들은 무차원 구현 기본값이며 관측에서 얻은 상수나 CE 필연 산출이 아니다. $p^*$는 설정, loss, 게이트 또는 phase 분류에 삽입하지 않았다.

구현 위치는 다음과 같다.

- 핵심 모듈: `reality_stone/python/reality_stone/clarus/clarus_field.py`
- 공개 API: `reality_stone/python/reality_stone/clarus/__init__.py`
- 검사: `tests/test_clarus_field.py`, `tests/test_dimensionless.py`
- 결정론적 예시: `examples/agi/clarus_field_demo.py`
- 정본 상태와 코드 지도: `docs/7_AGI/1_AGI.md`, `docs/7_AGI/12_Equation.md`, `docs/7_AGI/18_CodeMap.md`

## 8. 수치 검증과 내부 비교

이 절은 구현 정합성 검사를 이론·과제 성능과 분리한다. 다음 표의 결과는 모두 내부 회귀검사이며 AGI 또는 생물학 점수가 아니다.

| 범위 | 결과 | 판정 범위 |
|---|---:|---|
| 클라루스장 focused | 17 passed | 장·게이트·공개 API |
| 무차원 + 장 | 28 passed | 무차원 식 등록 포함 |
| CE core slice | 71 passed, 2 warnings | bootstrap·dimensionless·layer A·bridge·field |
| runtime/public slice | 34 passed, 2 warnings | runtime contract 호환성 |
| local-cloud compatibility | 72 passed | V10/V13 계보 비회귀 |
| 변경 Python 정적 검사 | All checks passed | 문법·스타일 |

[수치 산출] CF-1/2 fixture에서 4,000 tick 동안 닫힌 기억의 비트 패턴이 보존됐고, $\sup\lVert\phi\rVert_2=11.3031\le11.4286$이었다. [수치 산출] A-E2R fixture의 두 원거리 초기조건은 같은 입력에서 합류했고, 40,000 tick 점유율 차는 $2.031\times10^{-5}$였다. [음성 대조] 서로 다른 입력률에서 활성률이 입력률을 추적했으므로 전체 $p^*$ 자기수렴 증거는 얻지 못했다. [구현 산출] 8-node, 32-tick 예시는 최대 기억 노름 0.9999998953, 장 노름 9.4243143777, 평균 점유율 $(0.125,0.765625,0.109375)$을 냈다. 이 한 번의 예시는 smoke 결과다.

기존 CE 검증 하네스는 bootstrap 잔차 $2.08\times10^{-17}$, 차원 검사 7/7, scorecard 11 PASS와 1 CAUTION을 기록해 전체 상태를 CAUTION으로 남겼다. 이 하네스의 $D_{\mathrm{eff}}=3.17776$은 본 run의 $3.1766443715$와 다른 원천을 사용하므로 두 스칼라 수치를 동일시하지 않는다.

## 9. 현재 AGI 연구 단계

이 절은 저장소의 구현 성숙도와 과제 증거를 구분한다. [상태 판정] 현재 저장소는 AGI가 아니라 수학·런타임·합성 과제 연구 prototype이다.

| 계보 | 구현·검증 상태 | 과제 판정 |
|---|---|---|
| BrainRuntime A–F | A–E 계층과 실행 가능한 F-loop 배선 | 소프트웨어 정합성; 생물학·AGI 검증 아님 |
| STDP | 구현 경로와 감사 존재 | 효능 NO-EFFECT, held-out guard 실패 |
| V9 | 수학·finite controller·runtime 구현 | 256-seed STOP; 정확도 0.3457 대 monolithic 0.6116 |
| V10 | local/cloud synthetic mechanism | 64+64 seed의 좁은 조건부 GO |
| V11 | 강한 recurrent/OOD 비교 | 14 gate 중 10 STOP; compute-matched Elman-3에 열세 |
| V12 | stable bilinear 시도 | ABANDONED, scored result 0 |
| V13 계보 | 구현·단위검사 완료 | 16-seed 과제 모두 STOP; horizon/composition 실패 |
| V14 | lossless slot·bilinear 충분성 및 linear no-go | 정식 구현·scored test SKIPPED; route L은 toy GO-후보뿐 |
| Clarus field | CF-1/2/3/5 조건부 수학과 baseline 구현 | 과제 채점 전; $p^*$ 자기수렴 거부 |

[생물학 비교] CloudCell의 C. elegans 판정은 0/4, human proxy는 0/3으로 실패했다. GFP 대조는 이전 local temporal-memory 신경 해석을 기각했다. 따라서 “생물 뇌를 본떴다”는 현재 구조적 영감 수준이다. [우주 프레임워크] 유한 그래프 확산장은 우주 구조의 계산 유비만 제공한다. 연속 시공간 극한, 로런츠/게이지 공변성, 물리 단위, 관측 데이터와의 likelihood, 중력 backreaction은 구현하거나 검증하지 않았다.

## 10. 관측 비교

이 run은 외부 관측값을 인용하지 않았으므로 관측 비교 레인은 `SKIPPED`다. 내부 합성 fixture와 저장소 벤치 결과는 8절과 9절에 분리해 기록했다. 따라서 본 보고서는 우주 관측 또는 신경 데이터에 대한 적합도 판정을 내리지 않는다.

## 11. 미완성 과제와 한계

이 절은 다음 승격 전에 필요한 작업을 명시한다.

- [미완성] 내생 게이트·상태 의존 쓰기를 포함한 완전 결합계의 정상 법칙 존재·유일성 증명.
- [예측] 전체 $p^*$ 자기수렴의 사전등록 16-seed killing test. 신호율 변화, 셔플 입력, 무작위 게이트, $D'$ 스캔을 포함해야 한다.
- [미완성] bounded slot + HRR readout을 실제 장기 horizon·composition 과제에서 강한 recurrent baseline과 compute-matched 비교.
- [미완성] 뇌 경로에는 spike timing, 세포 유형, 국소 가소성, neuromodulator 시간척도와 실제 신경 데이터가 필요하다.
- [미완성] 우주 경로에는 연속체 극한, 좌표·게이지 대칭, 보존법칙, 차원 복원, 독립 관측 likelihood가 필요하다.
- [한계] CF-2의 안정성은 과제 해결 능력이나 학습 능력을 뜻하지 않는다. CF-3의 평형 존재는 목표값 $p^*$를 선택하지 않는다. 회귀검사 통과는 AGI 행동 증거가 아니다.

열린 P0는 없다. 감사에서 발견된 P1-1부터 P1-4는 각각 거짓 3-simplex 정리 삭제, readout 고정, R-A1 분리, CF-3 전제 강화와 route L 상속 금지로 해소했다. 다음 연구의 우선순위는 현재 baseline을 그대로 고정한 과제 채점이며, 결과를 본 뒤에만 내생 적응 또는 spiking/continuum 확장을 별도 분기로 도입하는 것이다.

## 12. 재현성

이 절은 코드와 수학 검산을 다시 실행하는 경로를 제공한다. 저장소 루트는 `C:/Users/dongh/OneDrive/Desktop/Clarus-Equation`이고 run 디렉터리는 `_workspace/ce/agi-clarus-field-20260812`다. route toy 이외의 본 구현 검사는 결정론적 fixture를 사용하며, 확률 검산의 시드와 표본 수는 각 artifact 스크립트에 고정돼 있다.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_clarus_field.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest tests/test_dimensionless.py tests/test_clarus_field.py -q -p no:cacheprovider
.venv\Scripts\python.exe examples/agi/clarus_field_demo.py
.venv\Scripts\python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf1_cf2.py
.venv\Scripts\python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf3.py
.venv\Scripts\python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf4.py
.venv\Scripts\python.exe _workspace/ce/agi-clarus-field-20260812/artifacts/verify_cf5.py
```

전체 명령과 환경 예외는 `31-validation.md`에 기록했다. `%TEMP%` 접근권한으로 실패한 최초 runtime/public 실행은 검증된 workspace-local `--basetemp`로 같은 검사들을 다시 실행해 34 passed를 얻었으며, 환경 setup 실패와 코드 실패를 분리했다.

## 13. 참조

외부 문헌과 외부 데이터는 사용하지 않았다. 내부 근거는 `00-contract.md`, `11-math.md`, `12-routes.md`, `20-audit.md`, `30-implementation.md`, `31-validation.md`와 그 파일들이 지시하는 artifacts 및 정본 문서다.

Status: COMPLETE

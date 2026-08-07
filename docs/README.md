# Clarus Equation 문서 안내

Clarus Equation(CE)은 가능한 경로·후보·상태가 선택되기 전의 분포와, 그 분포가 한 결과로 수렴하는 과정을 하나의 무차원 언어로 기술하려는 연구 프로그램이다.

이 저장소의 문서는 완성된 단일 이론 논문이 아니다. 순수 수학, 물리량과의 연결 규칙, 수치 현상론, 공학·뇌·AI 응용이 함께 있는 연구 아카이브다. 따라서 각 주장은 반드시 `Exact`, `Selection`, `Bridge`, `Phenomenology`, `Open`, `Open test` 중 어느 층인지 확인해서 읽어야 한다.

> Runtime integration history (2026-05-09): 당시 실행 예제는
> `reality_stone/examples/unified_clarus_demo.py`로 통합되었고, Clarus runtime은
> `reality_stone/python/reality_stone/clarus`로 이동했으며 과거 최상위
> `clarus/` package는 제거되었다. 이후 예제가 다시 확장되었으므로 현재
> 실행 파일 목록은 9절을 우선하고, 오래된 `examples/...`·`scripts/...`
> 참조는 역사 기록일 수 있음을 확인한다.

## 문서 사용 규칙

- **처음 읽기**:
  [코어 독자 가이드](코어_독자_가이드.md)에서 환기구 비유와
  `밝혀진 것 / 조건부인 것 / 열린 것`을 수식 전에 확인한다.
- **입력(캘리브레이션)**: 관측값을 사용해 CE 내부 파라미터를 고정하는 단계
- **출력(예측)**: 고정된 파라미터로 새로운 관측량을 계산하는 단계
- **비교(검증)**: 출력이 관측과 어느 정도 일치하는지 평가하는 단계
- **가정(모형 선택)**: 평탄성, 성분 분해, 무차원화 스케일 등 물리 모형을 닫기 위해 추가로 채택하는 조건

독자 가이드 다음에는 코어의 현재 강화 과정인
[`0_검증과감사/CORE_STRENGTHENING_LOOP.md`](0_검증과감사/CORE_STRENGTHENING_LOOP.md)를
따른다. 이 문서는 생존함수, 자기·타공간 재귀, 가지 안정성,
Hodge type closure, \(D_{\mathrm{eff}}\)의 반례군과 실행 gate를 함께 관리한다.

세부 오일러 항등식 기반 읽기 규칙은 2절에서 정리한다.

## 1. 한눈에 보는 핵심 논리

CE의 계산 사슬은 다음처럼 읽는 것이 가장 안전하다.

```text
후보 경로 또는 상태
        ↓
무차원 비용·결함·작용
        ↓
지수형 생존/Gibbs 가중
        ↓
자기재귀 고정점 또는 저온 농축
        ↓
하나의 manifest 결과 선택
        ↓
물리량·관측량과의 Bridge
        ↓
데이터 비교와 반증
```

이를 수식으로 압축하면 두 계열이 중심이다.

첫째, \(S:\mathbb R_{\ge0}\to(0,1]\)가 비자명하고 측정가능하며,
독립적인 비음수 비용이 더해질 때 생존 가중치가 곱해진다는 조건을 둔다.

$$
S(D_1+D_2)=S(D_1)S(D_2),\qquad S(0)=1.
$$

그러면 \(-\log S\)에 대한 Cauchy 함수방정식으로

$$
S(D)=e^{-\lambda D}
$$

가 되며 range와 비자명성에서 \(\lambda>0\)이다. \(\lambda=1\)은 깊이 단위를
unit-rate로 잡는 정규화/Poisson 선택이지 위 함수방정식이 정한 수치가 아니다.
\(S(D)=e^{-D}\)의 함수형 유일성은 이 전체 조건 아래의 정리지만, 독립
곱셈 조건과 unit-rate를 실제 물리에 적용하는 것은 `Selection` 또는
`Bridge`다.

둘째, 선택된 분율이 다시 유효 비용을 바꾸는 자기재귀 조건을 둔다.

$$
x=e^{-(1-x)D_{\mathrm{eff}}}.
$$

이 고정점은 Lambert $W$ 함수와 수치 solver로 다룰 수 있다. 그러나 해 $x$를 바리온 분율, 활성 뉴런 비율, 답변 생존분율 같은 특정 관측량으로 읽는 순간 별도의 `Bridge`가 필요하다.

## 2. 오일러 항등식의 역할

CE는

$$
e^{i\pi}+1=0
$$

을 모든 물리량을 직접 산출하는 만능 공식으로 사용하지 않는다. 문서군에서는 이를 다음 다섯 연산을 묶는 **최소 생성 문법**으로 읽는다.

| 기호 | 1차 역할 | CE에서의 예와 문서 기능 |
|---|---|---|
| $e$ | 접힘·감쇠·생존·Gibbs 가중 | $S(D)=e^{-\lambda D}$; $\lambda=1$은 단위 선택 |
| $\pi$ | 주기·위상공간 정규화 | $1/(2\pi)$; 결합 sum rule은 `Selection` |
| $i$ | Lorentzian 진폭의 위상·간섭 | $e^{iS/\hbar}$; Born 확률 readout은 별도 열린 `Bridge` |
| $1$ | 정규화·완전 상태 | $S(0)=1$, $e^0=1$; 고정점 수렴은 도함수로 별도 검사 |
| $0$ | 영점·선택·manifest 조건 | 잔차 $r=0$, $d(d-3)=0$ 같은 분기 선택 규칙 |

핵심 원칙은 먼저 **무차원 코어 식**을 고정하고, **선택 규칙**과
**브리지 규칙**으로 표준모형·우주론 변수에 연결한 뒤, 마지막에만
**스케일 승격**으로 차원 있는 물리량을 복원하는 것이다. 따라서 차원 있는
질량이나 에너지 값은 오일러 항등식에서 곧바로 나오지 않는다.
`m_\phi`, `M_{\text{CE}}`, `v_{\text{EW}}/M_{\text{Pl}}` 같은 양도
무차원 비율과 기준 스케일을 결합해 읽는다.

## 3. 주장 등급

| 등급 | 의미 | 필요한 검증 |
|---|---|---|
| `Exact` | 정의·항등식·정리로 닫힌 부분 | 관측값 없이 수학적으로 검산 |
| `Selection` | 분기·정규화·물리 가지 선택을 명시하면 닫힘 | 대안과 선택 조건 제시 |
| `Bridge` | 수학적 대상을 실제 물리량과 연결 | 독립 유도 또는 실험 gate |
| `Phenomenology` | 유효모형·보정·관측 readout 포함 | 데이터와 holdout 비교 |
| `Open` | 아직 통과 항목으로 세지 않음 | 추가 증명 또는 재설계 |
| `Open test` | 아직 관측되지 않은 사전등록 예측 | 실제 실험 결과 |

코드가 실행되고 residual이 작다는 사실은 구현의 정확성을 검증한다. 그것만으로 물리적 `Bridge`가 증명되는 것은 아니다.

### 3.1 마스터 문서와 문서 계층

| 문서 | 내용 |
|---|---|
| [경로적분.md](경로적분.md) | CE 계산 체인의 최신 원천. 정의, 브리지, 스케일 승격, 통합 예측을 담은 메인 문서 |
| [상수.md](상수.md) | 대표 수치와 항목 지위의 최종 집계표. 저장소 전체의 상수·관측량 표기는 이 문서를 우선한다 |
| [axium.md](axium.md) | 공리, 기호, 문서 레벨 규약을 고정하는 프레이밍 문서 |

CE 문서는 네 층으로 읽는 것이 가장 안전하다.

| 층 | 질문 | 대표 문서 | 역할 |
|---|---|---|---|
| 공리층 | 무엇을 기본 전제로 둘 것인가 | [axium.md](axium.md) | 기호, 공리, 주장 범위 고정 |
| 유도층 | 공리에서 어떤 식이 실제로 내려오는가 | [경로적분.md](경로적분.md) | 정의, 가정, 근사, 정량 유도 |
| 집계층 | 어떤 관측량이 어디서 나오는가 | [상수.md](상수.md) | 대표 수치와 상수·관측량의 최종 집계 |
| 응용층 | 유도된 구조를 어디에 적용할 것인가 | `2_경로적분과_응용/`, `4_공학적_활용/`, `5_유도/`, `6_뇌/` | 우주론 외 확장, 벤치마크, 구조 유비 |

### 3.2 현재 정리 원칙과 레거시 표기

- 전수 교정의 최신 변경 계약은
  [`0_검증과감사/FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md`](0_검증과감사/FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md)다.
- 최신 Track A 입력과 파생 수치는
  [`0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)을
  단일 원천으로 삼고, `verify_numeric_consistency.py`로 재계산한다.
- `경로적분.md`는 계산 체인의 최신 원천이다.
- `상수.md`는 대표 수치와 항목 지위의 최종 정본이다.
- `axium.md`는 공리, 기호, 주장 범위를 고정하는 프레이밍 문서다.
- `1_강의/`는 설명용 강의 노트이며, 코어보다 강한 결론으로 읽지 않는다.
- `2_경로적분과_응용/`, `4_공학적_활용/`, `5_유도/`, `6_뇌/`는 최신 정본과 호환되도록 다시 썼지만, 대부분 `Bridge` 또는 `Phenomenology` 층이다.
- 응용 문서의 수치 벤치마크는 코어의 직접 `Exact` 결론이 아니라, 추가 가정과 실험 제약을 포함한 해석으로 읽는다.
- 우주론 대표값은 최신 Track A의 명시적 분할식 아래
  \(R=0.3782386966\)과
  \((\Omega_b,\Omega_{DM},\Omega_{DE})=(0.0486382585,0.2610881744,0.6902735671)\)로 읽는다.
  과거 $R=0.38063$ 벡터는 이 식의 반올림값이 아니므로 현재 대표값이 아니다.
- 같은 manifest에서 Hubble readout 산술은
  \(H_0(0)=66.802746\), \(H_0(1)=72.702371\)
  \(\mathrm{km\,s^{-1}Mpc^{-1}}\)로 고정한다. 이는 readout ansatz의
  산술 검산이지 단일 사전 예측이 아니다.
- 13-component DESI DR2 covariance snapshot에는 서로 다른 두
  background·\(r_d\) 분기가 있다. external-input 분기는 위 late-time
  3-sector vector와 \(r_d=147.09\,\mathrm{Mpc}\)를 쓰고, 4-sector
  Eisenstein--Hu `Selection`은 \(\Omega_b\)를 유지한 채
  \(\Omega_{\rm rad}=9.192332266\times10^{-5}\)를 먼저 배정하고 남은 dark
  sector를 \(R_{\rm CE}\)로 다시 나눈 뒤
  \(r_d=151.50842877\,\mathrm{Mpc}\)를 쓴다. 두 고정 분기 모두 이
  BAO-only partial gate에서 `REJECT`다.
- canonical \(Z_2\) portal benchmark는
  \(\operatorname{BR}_{\rm inv}=0.77082222\)로 PDG 2026이 열거한
  ATLAS direct 한계 \(0.107\)을 통과하지 못한다. 올바른 CP-even scalar
  kernel의 유한질량 \(g-2\) 진단은
  \(162.55198\times10^{-11}\)이며, 옛 \(135\times10^{-11}\)은 폐기한다.

레거시 문서의 `정리/가정/식별/현상론` 표기는 위 등급으로 다시 해석한다.

- `A3a`는 자기일관 고정점 식이며 `Selection` 또는 `Exact` 층에 가깝다.
- `A3b`는 관측 가능한 선택 측도와 바리온 분율을 연결하는 `Bridge` 층이다.
- `\alpha_s^{1/d}`와 \(s_A^2:=4\alpha_s^{4/3}\) 체인은 registered
  output이며, 물리적 weak-angle scheme과의 동일시는 `Bridge` 층이다.
- 블랙홀, 뇌, AGI, LLM 문서는 코어 정전의 직접 증명이라기보다 `Phenomenology` 또는 구조적 응용 층으로 읽는다.

등급별 근거 위치는 `Exact`는 `axium.md`·`경로적분.md` 초반,
`Selection`은 `경로적분.md` 15절·`1_강의/`, `Bridge`는
`경로적분.md` 3절·`2_경로적분과_응용/10_공리_정당화.md`,
`Phenomenology`는 `경로적분.md` 후반·`상수.md`·응용 문서를 먼저 본다.

## 4. 대표 예제

### 4.1 자기재귀 고정점

최신 Track A의 $D_N=3.1779129995$를 사용하면 선택된 저분율 가지의 고정점은

$$
x=e^{-(1-x)D_{\mathrm{eff}}}
\quad\Longrightarrow\quad
x=0.0486382585
$$

이다. Lambert-$W$, bracketed solver와 역함수
$D=-\ln x/(1-x)$가 수치 허용오차 안에서 같은 값을 준다.

이 단계는 지정 $D_N$에서의 정확한 고정점 계산이다. 현재 우주의
$\Omega_b$와의 연결은 경로 수가 아니라 B2의 공변 에너지 가중
stress readout으로 수행한다.

검산:

```powershell
$env:PYTHONPATH = "reality_stone/python"
python reality_stone/python/reality_stone/clarus/bootstrap_solver.py
python -m pytest tests/test_bootstrap_solver.py -q
```

관련 문서:

- [3_상수/3_부트스트랩.md](3_상수/3_부트스트랩.md)
- [0_검증과감사/BRIDGE_B2_DERIVATION.md](0_검증과감사/BRIDGE_B2_DERIVATION.md)
- [9_등호이전/05_CE_브리지.md](9_등호이전/05_CE_브리지.md)

### 4.2 암흑물질과 암흑에너지

지정한 B2+A3b 후보 모형은 고정점 출력과 그 여집합을 먼저 나눈다.

$$
\Omega_{\mathrm{total}}
=
\underbrace{x}_{\text{A3a output}}
+
\underbrace{(1-x)}_{\text{complement}},
\qquad x\overset{\mathrm{A3b\ Bridge}}{\longleftrightarrow}\Omega_b.
$$

고정점 해 $x=0.0486382585$를 B2 에너지 readout으로 바리온 분율과 연결하면

$$
\Omega_{\mathrm{dark}}=1-\Omega_b=0.9513617415
$$

가 된다. 이는 A3b 식별 아래의 조건부 총량이며, 암흑물질과 암흑에너지를
아직 동역학적으로 구분하지 않는다.

dark-split ansatz는 여집합을 다음 두 후보 성분으로 읽는다.

| 성분 | 후보 해석 | 검증해야 할 동역학 |
|---|---|---|
| 암흑에너지 $\Omega_\Lambda$ | Clarus 장의 진공·0차 배경 성분 | 균질한 배경과 가속 팽창 |
| 암흑물질 $\Omega_{DM}$ | 진공 위에 남은 집합적 요동·응축 성분 | 중력적으로 뭉치며 구조 성장에 참여 |

두 성분의 비를

$$
R\equiv\frac{\Omega_{DM}}{\Omega_\Lambda}
$$

로 두면 최신 Track A의 명시적 분할 functional은

$$
\boxed{R=\alpha_sD_N(1+x\delta_N)=0.3782386966}
$$

이다. 이는 $\alpha_sD_N$ 선도항에 $x\delta_N$으로 정의한 neutral
feedback을 붙인 하나의 고정된 현상론 모형이다. 그러면 전체 암흑 부문을
다음처럼 분할한다.

$$
\Omega_\Lambda
=
\frac{1-x}{1+R}
=0.6902735671,
$$

$$
\Omega_{DM}
=
\frac{(1-x)R}{1+R}
=0.2610881744.
$$

세 성분은 정규화를 만족한다.

$$
\Omega_b+\Omega_{DM}+\Omega_\Lambda=1.
$$

또한

$$
\frac{\Omega_{DM}}{\Omega_b}\simeq5.368
$$

이므로 이 현상론 분할식은 약 5.37의 비를 조건부로 재현한다. 생존 고정점
\(x\)와 dark-split ratio \(R\)를 실제 우주 성분에 연결하는 공변 동역학과
공동 likelihood가 통과하기 전에는 이를 “왜 약 5배인가”의 완전한 설명으로
세지 않는다.

#### 암흑물질의 정체

현재 split ansatz의 암흑물질 후보는 특정 WIMP 하나가 아니라

$$
\Omega_{DM}
=
\text{Clarus 장 진공 위의 집합적 요동 성분}
$$

로 표기한 집합 성분이다. 이는 현상론적 식별이지 미시적 정리나 특정 입자의
검출 선언이 아니다. 정확한 $Z_2$는 홀수-$\Phi$ vertex와 단일-$\Phi$ 붕괴를 금지하지만,
Higgs portal $-\lambda_{HP}|H|^2\Phi^2$가 있으면 EWSB 뒤
$h\Phi^2$ vertex가 남아 Higgs 매개 핵반동은 일반적으로 0이 아니다.
따라서 직접검출 판정은 $\lambda_{HP}$, pole mass와 핵자 form factor를
넣은 산란단면적으로 수행한다.

$m_{\rm light}=m_p\delta_N^2\simeq29.6992\ \mathrm{MeV}$는 우주 전체
암흑물질 입자의 확정 질량이 아니다. 집합적 장 요동을 국소 스칼라 pole로
표현할 때 생기는 입자언어 `Bridge` 후보다.

#### 암흑에너지의 정체

dark-energy 후보는 Clarus sector의 vacuum-like background 성분으로 읽는다.
이 stress가 실제로 \(p\simeq-\rho\)를 만족하는지와 섭동 안정성은 공변
작용에서 검정한다. 밀도 분할값 \(\Omega_\Lambda\)와, 왜 절대 에너지
밀도가 매우 작은가 하는 우주상수 문제는 서로 다른 질문이다.

- $\Omega_{DE}=0.6902735671$: 이 분할 모형에서 전체 에너지 중 차지하는 무차원 분율
- $\rho_\Lambda$: 실제 진공 에너지의 절대 스케일

문서에는 de Sitter 엔트로피와 위상면적을 이용해 $\rho_\Lambda^{1/4}\sim2.24\ \mathrm{meV}$를 읽는 홀로그래피 후보가 있다. 이 부분은 미시적 양자중력 유도가 남아 있으므로 `Exact`가 아니라 `Bridge/Phenomenology`다.

#### 현재 판정과 역사

암흑 부문 분할 규칙은 다음 순서로 발전했다.

1. 초기에는 $R\simeq\alpha_s\pi$를 사용했다.
2. 이미 고정된 유효 깊이를 재사용하는 $R_{\mathrm{LO}}=\alpha_sD_N$로 교체했다.
3. 최신본은 식에 없는 수치 보정을 버리고
   $R=\alpha_sD_N(1+x\delta_N)$ 자체의 계산값 $0.3782386966$을 사용한다.

내부 정규화는 닫히지만, “진공 성분 = 암흑에너지”와 “집합적 요동 =
암흑물질”의 미시적 동일시는 아직 독립적으로 증명되지 않았다. 따라서 현재
공식 판정은 둘 다 `Phenomenology`다. 더구나 현행 external-input
background의 13-component DESI DR2 BAO-only full-cov partial gate는
\(\chi^2=40.20145\), dof \(=13\),
\(p=1.2828\times10^{-4}\)로 `REJECT`다. 밀도비 하나의 근접성과 은하 halo,
렌즈, CMB 성장률을 동시에 재현하는 것은 서로 다른 검증 문제다.

관련 문서:

- [경로적분.md](경로적분.md) 3절
- [3_상수/3_부트스트랩.md](3_상수/3_부트스트랩.md)
- [3_상수/7_우주론.md](3_상수/7_우주론.md)
- [2_경로적분과_응용/02_에스컬레이터.md](2_경로적분과_응용/02_에스컬레이터.md)

#### 코어 보강과 남은 간극

- $P_{\text{count}}$와 $\Omega_b$를 직접 동일시하지 않는다. B1--B4와
  A3b를 함께 채택해 renormalized stress tensor의 에너지 가중 측도,
  관측 초곡면, sector projector, feedback depth를 모두 고정한 경우에만
  $x=\langle E_b\rangle/\langle E_{\rm tot}\rangle=\Omega_b$가 조건부로 닫힌다.
- `\alpha_s^{1/d}` 및 \(s_A^2:=4\alpha_s^{4/3}\) 체인은 registered
  matching ansatz다. \(s_A^2\)를 특정 on-shell, \(\overline{\rm MS}\) 또는
  effective weak angle로 보내는 scheme map 전에는 정밀 일치 score를 주지
  않는다. 식 선택에 참고한 EW 값도 같은 revision의 holdout으로 재사용하지 않는다.
- `N_c = d`는 구조적 유비를 넘어 동역학적 동일성으로 닫히지 않았다.
- 뇌 문서의 항목은 `6_뇌/05_실험근거.md`의 실험 게이트에서 `supported`로 판정될 때만 `Phenomenology`에서 `Bridge`로 격상된다(등급 기준: `6_뇌/05_실험근거/01_판정기준과핵심주장.md` 1절의 supported/bridge/testable). 격상 근거는 실험 게이트이지 닫힘 가정 자체가 아니다. 부트스트랩 닫힘 가정 (C1)-(C3)은 `1_강의/A_연역적_유도.md` 2.5절에, 의식의 (C1)-(C3) 자기일관 조건은 `7_AGI/7_Consciousness.md` 1절에 별도로 정의된다(서로 다른 (C1)-(C3) 집합이며 `05_실험근거`에는 정의되지 않는다). AGI/게임 밸런스는 여전히 응용 가설층.
- 공학/블랙홀 문서의 수치는 최신 정본과 양립하도록 정리되었지만, 대부분 설계 벤치마크 또는 유효 모델 수준이다.
- A1의 보통 경로적분 Hessian은 아직 공변
  \(\mathcal K_{\mu\nu}\)나 stress tensor로 닫히지 않았다. field-space
  connection, gauge/ghost, renormalization과 metric-variation matching의
  통과 조건은
  [A1/Q0 공변 작용 루프](0_검증과감사/A1_Q0_COVARIANT_ACTION_LOOP.md)를
  따른다.

### 4.3 전자약-강력 결합 브리지

대표적인 물리 브리지 후보는

$$
s_A^2:=4\alpha_s^{4/3}
$$

이다. 여기서 \(s_A^2\)는 우선 CE neutral-mixing output이고, 물리적
\(\hat s_Z^2\) 또는 \(\bar s_\ell^2\)와의 동일시는 RG·threshold·scheme
conversion이 필요한 `Bridge`다. 지수와 prefactor도 Yang--Mills 작용에서
유도되지 않았으므로, 중심값 근접성을 독립 검증으로 세지 않는다.

산술 재현(물리적 scheme 검증 아님):

```powershell
python examples/physics/alpha_s_closure_gate.py
```

이 스크립트는 Track B의 낮은 양의 근을 숨기고 “zero free parameters”를
출력하는 legacy/noncanonical diagnostic이다. 물리 validation이나 최신
acceptance에 쓰지 않는다. 현행 두-branch 산술은
`docs/0_검증과감사/verify_numeric_consistency.py`가 검증한다.

관련 문서:

- [3_상수/2_혼합매개변수.md](3_상수/2_혼합매개변수.md)
- [2_경로적분과_응용/10_공리_정당화.md](2_경로적분과_응용/10_공리_정당화.md)

### 4.4 입력·조건부 출력·공동 likelihood의 분리

flavour와 인플레이션은 다음 네 역할을 섞지 않는다.

| 항목 | 현행 역할 | 검증 계약 |
|---|---|---|
| CKM/PMNS | 하나의 Yukawa·질량 texture를 대각화한 unitary-matrix benchmark | 원소별 사후 보정식의 개별 PASS를 폐기하고, 전체 matrix·invariant를 global-fit covariance로 공동 평가 |
| \(A_s=2.10\times10^{-9}\) | 외부 normalization input | finite-\(\xi\) potential의 \(\lambda_4=1.3434991\times10^{-10}\)를 고정; 입력값을 성공 건수로 재집계하지 않음 |
| \(N_*=57.1999\) | reheating-history benchmark input | \(N_*\)를 지정한 뒤 exact Einstein-frame 적분을 실행 |
| \(n_s,r\) | 조건부 출력 | \(n_s=0.9661711385,\ r=0.0043456103\); 독립 spectrum holdout과 reheating completion 필요 |

구 `ckm_vcb_nlo_gate.py`와 `primordial_spectrum_readout_gate.py`는
후보식을 기록한 legacy diagnostic이며 현행 acceptance가 아니다. 현행 산술과
문서 계약은 다음 명령으로 검사한다.

```powershell
python docs/0_검증과감사/verify_numeric_consistency.py
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

첫 명령은 canonical manifest의 정의·산술과 동결 likelihood의 \(p\)-value를
검산한다. 둘째 명령은 H1·링크·delimiter·표·제어문자, 제한된 semantic
sentinel과 canonical 산술을 전수 검사한다. 이 명령이 flavour global fit,
Boltzmann likelihood 또는 모든 작용의 물리적 정당성을 자동 증명한다고
해석하지 않는다.

### 4.5 등호 이전의 후보 선택

`9_등호이전/`은 여러 후보가 아직 하나의 등호로 확정되기 전의 상태를 다룬다. 유한 후보 $x_i$와 결함 에너지 $E_i$가 있을 때

$$
p_\beta(x_i)
=
\frac{e^{-\beta E_i}}
{\sum_j e^{-\beta E_j}}
$$

를 두면 $\beta\to\infty$에서 최소 결함 후보로 질량이 농축된다. 이 구조는 유한공간에서는 엄밀하며, 연속·비콤팩트 경로공간으로 확장할 때 coercivity, tightness, Gamma 수렴 조건이 필요하다.

관련 문서:

- [9_등호이전/01_공리와증명.md](9_등호이전/01_공리와증명.md)
- [9_등호이전/02c_Gamma수렴과Gibbs농축.md](9_등호이전/02c_Gamma수렴과Gibbs농축.md)
- [9_등호이전/04_PreEq_범주.md](9_등호이전/04_PreEq_범주.md)

### 4.6 LLM 답변 검증

LLM 응용에서는 여러 답변 후보를 등호 이전 상태로 보고, 근거와 모순되는 claim에 높은 defect를 부여한다. 낮은 잔차를 가진 답변만 manifest시키고, 비선택 잔차는 오류 분석에 남긴다.

검산:

```powershell
python examples/pre_eq/llm_manifest_verifier.py
python examples/pre_eq/claim_residual_verifier.py
python examples/pre_eq/claim_residual_benchmark.py benchmarks/eval/faithbench_all.jsonl
```

관련 문서:

- [4_공학적_활용/10_PreEq_LLM_manifest_verifier.md](4_공학적_활용/10_PreEq_LLM_manifest_verifier.md)
- [4_공학적_활용/11_CE_claim_residual_verifier_formula.md](4_공학적_활용/11_CE_claim_residual_verifier_formula.md)

### 4.7 Clarus 장과 pole bridge

문서의 “Clarus boson”은 코어에서 독립 입자를 먼저 가정한 것이 아니다. Clarus 장의 2점 상관함수 pole 또는 역상관길이를 입자언어로 읽는 `Bridge`다.

$$
m_{\rm light}=m_p\delta_N^2\simeq29.6992\ {\rm MeV}
$$

는 이 pole readout의 최신 질량 ansatz다. 해당 질량창의 신호만으로 field
identity가 확정되지는 않으며 production coupling·width·residue도 함께
필요하다. 신호 부재는 지정한 국소 스칼라 readout을 제약한다.

검산:

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

`examples/physics/clarus_boson_search_gate.py`는 아직
\(\alpha_s=0.11789\), \(m=29.65\,\mathrm{MeV}\)를 쓰는 legacy gate라 최신
29.6992 MeV 정본의 acceptance 명령으로 사용하지 않는다.

## 5. 현재 검증 상태

가장 보수적인 현재 판정은 다음과 같다.

| 영역 | 현재 상태 |
|---|---|
| 생존함수의 함수형 | 정규성·곱셈 조건 아래 `Exact/Selection` |
| 자기재귀 고정점과 solver | 수학·코드 수준 `Exact` |
| $d=3$ Hodge closure | 지정 oriented metric type-closure 아래 선형대수 `Exact`; 자연에 적용하는 조건은 `Selection` |
| $P_{\mathrm{survive}}\leftrightarrow\Omega_b$ | `Bridge` |
| 결합상수·혼합각 | 주로 `Bridge` |
| DM/DE·초기 스펙트럼·Hubble readout | 주로 `Phenomenology/Open` |
| 고정-background DESI DR2 13-component BAO-only partial gate | \(\chi^2=40.20145\), dof \(=13\), \(p=1.2828\times10^{-4}\): `REJECT`; CMB·SN·growth full joint는 `Open` |
| Clarus field pole | `Open test` |
| 공학·뇌·AGI | `Phenomenology/Open test` |
| PreEq 유한수학 | 상당 부분 `Exact under assumptions` |
| PreEq의 실제 CE 경로공간 승격 | `Selection/Bridge/Open` 혼합 |

정합성 관점에서는 무차원 코어가 비교적 안정적이고, 선택 규칙은 대체로
정합하며, 브리지 규칙이 핵심 검증 구간이다. 뇌·AGI·공학 응용은 직접
증명보다 구조 유비로 읽는다. 즉 CE 전체를 이미 `Exact`로 완전 폐쇄된
이론으로 읽는 것은 과도하다.

최신 판정은 다음 문서를 우선한다.

- [0_검증과감사/PROOF_STATUS_MATRIX.md](0_검증과감사/PROOF_STATUS_MATRIX.md)
- [0_검증과감사/PROOF_VALIDATION_LEDGER.md](0_검증과감사/PROOF_VALIDATION_LEDGER.md)
- [0_검증과감사/미해결_난제_목록.md](0_검증과감사/미해결_난제_목록.md)

## 6. 문서의 발전 역사

CE 문서군의 역사는 다음 단계로 이해할 수 있다.

### 1단계: 압축 생성 문법

초기에는 오일러 항등식의 $0,1,e,\pi,i$를 차원·정규화·감쇠·주기·위상의 최소 기호로 읽고, 여러 물리 상수를 하나의 구조로 묶으려는 시도에서 출발했다.

이 시기의 흔적은 `1_강의/`, `axium.md`, `경로적분.md`의 초기 장에 남아 있다. 설명력은 강하지만, 생성 문법과 직접 물리 증명의 경계가 흐린 문장이 일부 존재한다.

### 2단계: 경로 생존과 부트스트랩

핵심이 $S(D)=e^{-D}$와

$$
x=e^{-(1-x)D_{\mathrm{eff}}}
$$

로 압축되었다. 여기서 차원 선택, 생존분율, 바리온·암흑성분 readout, 결합상수 관계가 하나의 계산 사슬로 정리되었다.

### 3단계: 상수·우주론 확장

입자 질량, CKM/PMNS, 인플레이션, 암흑에너지, $H_0$ 등으로 적용 범위가 넓어졌다. 많은 수치 후보와 gate가 이 단계에서 생겼다.

동시에 raw 식 실패 뒤 보정식이나 관측 readout을 선택하는 사례도 늘었다. 현재 문서에서는 이런 항목을 `Exact`가 아니라 `Bridge` 또는 `Phenomenology`로 분류한다.

### 4단계: 검증과 감사

“맞는 숫자”와 “증명된 식”을 구분하기 위해 증명 상태표와 검산 ledger가 도입되었다. Newton Jacobian, 관측 기준 충돌, $V_{cb}$ LO 실패, $A_s$ raw 실패 등이 숨겨지지 않고 기록되었다.

현재 문서군의 신뢰도를 지탱하는 핵심은 이 감사 체계다.

### 5단계: 뇌·AGI·공학 응용

경로 선택과 잔류장 개념이 sparse runtime, sleep/replay, hallucination gate, graph residual, Riemann attention으로 확장되었다.

이 단계의 문서는 CE 코어의 직접 증명이 아니라, CE 구조를 계산 시스템에 투영한 응용 연구로 읽어야 한다.

### 6단계: 등호 이전 수학

가장 최근의 큰 확장은 `9_등호이전/`이다. “한 값이 왜 선택되는가”를 유한 후보공간, Gibbs 농축, Markov/Kleisli 구조, tropical limit, Gamma 수렴으로 다시 세웠다.

이 축은 CE의 물리 브리지를 자동 증명하지는 않지만, 후보 선택과 잔차 보존을 다루는 가장 엄밀한 수학 기반을 제공한다.

## 7. 권장 읽기 순서

### 빠르게 개념만 볼 때

1. 이 README
2. [axium.md](axium.md)
3. [상수.md](상수.md)
4. [0_검증과감사/PROOF_STATUS_MATRIX.md](0_검증과감사/PROOF_STATUS_MATRIX.md)

### 수학 사슬을 볼 때

1. [9_등호이전/README.md](9_등호이전/README.md)
2. [9_등호이전/01_공리와증명.md](9_등호이전/01_공리와증명.md)
3. [9_등호이전/02_연속공간과측도.md](9_등호이전/02_연속공간과측도.md)
4. [9_등호이전/05_CE_브리지.md](9_등호이전/05_CE_브리지.md)
5. [경로적분.md](경로적분.md)

### 물리 예측을 감사할 때

1. [상수.md](상수.md)
2. [3_상수/](3_상수/)
3. [0_검증과감사/PROOF_VALIDATION_LEDGER.md](0_검증과감사/PROOF_VALIDATION_LEDGER.md)
4. [0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json](0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)
5. [0_검증과감사/run_full_consistency_gate.ps1](0_검증과감사/run_full_consistency_gate.ps1)

### AI·런타임을 볼 때

- `6_뇌/05_실험근거.md`는 뇌·AGI 연결의 실험 gate다.
- `7_AGI/12_Equation.md`는 런타임 기호를 모으지만, 상단의 5계층 stack만 canonical runtime spec으로 읽는다.
- canonical stack은 `kernel dynamics`, `coupling / geometry`, `mode update`, `hippocampus / replay`, `global runtime summary`로 고정한다.
- 이 다섯 층의 수학적 정의가 `Exact/Selection`이어도 뇌 대응이 들어가면 지위는 `Bridge`를 넘지 않는다.
- 실험 gate에서 `supported`인 항목은 최대 `Bridge`, `bridge`인 항목은 그대로 `Bridge`, `hypothesis`인 항목은 `Phenomenology`로만 사용한다.
- `hypothesis`를 성능 보장이나 “증명된 뇌 등가성”으로 올리지 않는다.

1. [7_AGI/12_Equation.md](7_AGI/12_Equation.md)
2. [7_AGI/14_BrainRuntimeSpec.md](7_AGI/14_BrainRuntimeSpec.md)
3. [7_AGI/17_AgentLoop.md](7_AGI/17_AgentLoop.md)
4. [7_AGI/18_CodeMap.md](7_AGI/18_CodeMap.md)
5. [7_AGI/21_STDP_Efficacy_Audit.md](7_AGI/21_STDP_Efficacy_Audit.md)
6. [4_공학적_활용/10_PreEq_LLM_manifest_verifier.md](4_공학적_활용/10_PreEq_LLM_manifest_verifier.md)

| 새 개념 | 먼저 볼 문서 | 해석 레벨 |
|------|------|------|
| runtime Clarus cell / bitfield | `7_AGI/12_Equation.md` | software kernel dynamics; 발견된 생물학적 assembly가 아님 |
| neural Clarus assembly / instruction 후보 | `6_뇌/10_신경프로그래밍언어_역공학.md` | task-code·표현 interface 골격 `YES`; stationary·switching·common-successor·state-hierarchy proxy와 language identification `NO`; 다음은 다영역 local-DSL/interface·최소 충분 회로·실제 합성·인과 gate |
| neural transition-noise / diffusion probe | `6_뇌/10_신경프로그래밍언어_역공학.md` §19 | 시험한 여섯 family 중 local affine+isotropic Gaussian proxy가 model-relative winner; Gaussianity 자체, state·time·anisotropic 지문, 300ms stationary single-OU scale consistency는 `NO`; 생물학적·생성형·score·인과·spatial graph 기전도 식별되지 않음 |
| geometry coupling / graph | `7_AGI/2_Architecture.md`, `6_뇌/05_실험근거.md` | coupling / geometry |
| wake / NREM / REM | `7_AGI/3_Sleep.md`, `6_뇌/05_실험근거.md` | mode update |
| hippocampus / replay | `7_AGI/3_Sleep.md`, `6_뇌/05_실험근거.md` | hippocampus / replay |
| sparse lifecycle / snapshot | `7_AGI/12_Equation.md` | global runtime summary |
| self / identity 해석 | `7_AGI/12_Equation.md` | `Phenomenology` 우선 |

### 질문별 빠른 탐색 지도

| 질문 | 먼저 볼 문서 | 이어서 볼 문서 |
|---|---|---|
| 문서군 전체 완성도와 다음 정리 순서는 무엇인가 | [0_검증과감사/FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md](0_검증과감사/FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md) | [0_검증과감사/문서_전체_완성도_감사.md](0_검증과감사/문서_전체_완성도_감사.md) |
| 무엇이 증명·브리지·미해결인가 | [0_검증과감사/PROOF_STATUS_MATRIX.md](0_검증과감사/PROOF_STATUS_MATRIX.md) | [0_검증과감사/미해결_난제_목록.md](0_검증과감사/미해결_난제_목록.md) |
| CE의 최소 공리가 무엇인가 | [axium.md](axium.md) | [경로적분.md](경로적분.md) 3절 |
| 오일러 항등식이 어떤 역할을 하는가 | [axium.md](axium.md) | [경로적분.md](경로적분.md) 15.14절 |
| 바리온·DM·DE가 어떻게 유도되는가 | [경로적분.md](경로적분.md) 3절 | [상수.md](상수.md) 3층 |
| 우주론 수식들이 무엇을 의미하는가 | [3_상수/9_우주론_수식_의미와_후보.md](3_상수/9_우주론_수식_의미와_후보.md) | [3_상수/7_우주론.md](3_상수/7_우주론.md) |
| 상수들이 어디서 나오는가 | [상수.md](상수.md) | `3_상수/` 하위 문서 |
| 어떤 부분이 정리이고 어떤 부분이 가정인가 | [axium.md](axium.md) | [경로적분.md](경로적분.md) 19절 |
| 차원 있는 물리량은 어떻게 읽어야 하는가 | [경로적분.md](경로적분.md) | [상수.md](상수.md) 7–8층 |
| 계산 gate에서 무차원성을 어떻게 검사하는가 | [참조/무차원_감사_수학.md](참조/무차원_감사_수학.md) | `reality_stone/python/reality_stone/clarus/dimensionless.py` |
| 등호 이전 후보 상태는 어떻게 형식화하는가 | [9_등호이전/README.md](9_등호이전/README.md) | [9_등호이전/01_공리와증명.md](9_등호이전/01_공리와증명.md) |
| 우주론 밖 응용은 어디까지 연결되었는가 | `2_경로적분과_응용/`, `4_공학적_활용/` | `5_유도/`, `6_뇌/` |

## 8. 폴더 지도

| 폴더 | 역할 |
|---|---|
| [0_검증과감사/](0_검증과감사/) | 현재 증명 등급, 검산 기록, 미해결 문제 |
| [1_강의/](1_강의/) | 연역·귀납 설명과 다섯 생성 기호 |
| [2_경로적분과_응용/](2_경로적분과_응용/) | 코어에서 물리 문제로 가는 유도·브리지 |
| [3_상수/](3_상수/) | 입자·우주론 수치, Hubble readout 감사 |
| [4_공학적_활용/](4_공학적_활용/) | 공학, graph residual, LLM 검증 |
| [5_유도/](5_유도/) | 유체·단백질·암흑에너지·블랙홀 파생 모델 |
| [6_뇌/](6_뇌/) | 뇌 동역학, 수면, 공개 데이터, 생명 종별 gate 기록 |
| [7_AGI/](7_AGI/) | runtime, memory, sleep, agent loop, OOD 사양 |
| [8_리만/](8_리만/) | Mellin-Riemann Attention과 positional encoding |
| [9_등호이전/](9_등호이전/) | 후보분포, manifest, 범주론, 경로공간 승격 |
| [참조/](참조/) | 형식적 수학, 무차원 감사, 보조 유도 |

## 9. 실행 코드 위치

현재 실행 가능한 예제는 하나로 제한되어 있지 않다.

| 위치 | 내용 |
|---|---|
| `examples/physics/` | 상수·우주론·field-pole·proof gate |
| `examples/pre_eq/` | manifest verifier, claim residual, RAGTruth/FaithBench |
| `examples/agi/` | STDP 진단과 효능 benchmark |
| `reality_stone/examples/` | Reality Stone/Clarus 통합 데모 |
| `tests/` | Clarus 수학·런타임·응용 회귀 테스트 |

기본 검증:

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
.venv\Scripts\python.exe -m pytest tests/test_bootstrap_solver.py tests/test_dimensionless.py tests/test_core_axioms.py tests/test_bridge_gates.py -q -p no:cacheprovider
```

`tests/scorecard.py`와 `tests/run_validation.py`는 각각 legacy
\(\alpha_s=0.11789\) 및 \(m_\phi\simeq29.648\,\mathrm{MeV}\) baseline을
포함하므로 최신 Track A acceptance와 분리한다.
`examples/physics/horizon_entropy_lift_derivation_gate.py`도 현
checkout에서는 \(\alpha_s=0.11789\)를 고정해 구 Hubble readout을 내므로
canonical acceptance runner가 아니다. 최신 문서 수치는 위 manifest
verifier가 재계산한다.

일부 오래된 문서에는 현재 checkout에서 제거된 `scripts/...`,
`examples/ai/...`, `examples/physics/evolution/...`, `data/evolution/...`,
`quant/...` 경로가 남아 있다. 이런 참조는 현재 구현이 아니라 역사적 실험
기록 또는 추가 예정 코드일 수 있으므로, 실제 파일 존재 여부와 문서의
`recorded result`/`reproducible local gate` 등급을 먼저 확인해야 한다.

## 10. 현재 결론과 남은 질문

현재 CE를 가장 강하고 정직하게 표현하면 다음과 같다.

> CE는 지수형 생존·Gibbs 가중과 자기재귀 선택을 공통 코어로 삼고, 이를 물리·생명·AI에 연결하는 여러 Bridge를 시험하는 연구 프로그램이다.

현재 비교적 안정적인 것은 조건부 수학과 solver다. 가장 중요한 미해결 문제는 다음이다.

1. 경로 독립성과 곱셈 생존 조건의 미시적 물리 유도
2. 생존분율과 바리온 분율 사이의 독립적인 경로공간 Bridge
3. 결합상수 지수와 loop projector의 게이지 이론 유도
4. 사후 선택이 아닌 사전등록·holdout 기반 우주론 검증
5. Clarus 장 자체와 국소 field-pole readout의 반증 조건 분리
6. 유한 PreEq 정리를 실제 비콤팩트 물리 경로공간으로 승격
7. 뇌·AGI 응용에서 baseline 대비 독립적인 성능 우위 입증

새로운 식을 추가하기 전에 [검증과 감사](0_검증과감사/)에 판정 근거와 반증 조건을 먼저 기록하는 것이 이 문서군의 운영 원칙이다.

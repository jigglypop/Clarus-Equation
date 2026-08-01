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

첫째, 독립적인 비용이 더해질 때 생존 가중치가 곱해진다는 조건을 둔다.

$$
S(D_1+D_2)=S(D_1)S(D_2),\qquad S(0)=1.
$$

연속성 또는 측정가능성을 함께 두면

$$
S(D)=e^{-\lambda D}
$$

가 되고, 무차원 단위 규약으로 $\lambda=1$을 선택하면 $S(D)=e^{-D}$를 얻는다. 함수형의 유일성은 조건부 수학 정리지만, 독립 곱셈 조건을 실제 물리에 적용하는 것은 `Selection` 또는 `Bridge`다.

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
| $e$ | 접힘·감쇠·생존·Gibbs 가중 | $S(D)=e^{-D}$ 같은 무차원 감쇠 규칙 |
| $\pi$ | 주기·위상공간 정규화 | $1/(2\pi)$, $\alpha_{\text{total}}=1/(2\pi)$, 위상 평균 |
| $i$ | 간섭·위상·분지 상쇄 | 경로적분, Born readout, 분지 간 상쇄 |
| $1$ | 정규화·완전 상태 | $S(0)=1$, $e^0=1$, 고정점 수렴의 기준선 |
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

- `경로적분.md`는 계산 체인의 최신 원천이다.
- `상수.md`는 대표 수치와 항목 지위의 최종 정본이다.
- `axium.md`는 공리, 기호, 주장 범위를 고정하는 프레이밍 문서다.
- `1_강의/`는 설명용 강의 노트이며, 코어보다 강한 결론으로 읽지 않는다.
- `2_경로적분과_응용/`, `4_공학적_활용/`, `5_유도/`, `6_뇌/`는 최신 정본과 호환되도록 다시 썼지만, 대부분 `Bridge` 또는 `Phenomenology` 층이다.
- 응용 문서의 수치 벤치마크는 코어의 직접 `Exact` 결론이 아니라, 추가 가정과 실험 제약을 포함한 해석으로 읽는다.
- 우주론 대표값은 현재 3계층 관성 보정 패키지인
  $\Omega_b=0.04865$, $\Omega_\Lambda\approx0.6891$,
  $\Omega_{DM}\approx0.2623$, $R\approx0.38063$으로 읽는다.

레거시 문서의 `정리/가정/식별/현상론` 표기는 위 등급으로 다시 해석한다.

- `A3a`는 자기일관 고정점 식이며 `Selection` 또는 `Exact` 층에 가깝다.
- `A3b`는 관측 가능한 선택 측도와 바리온 분율을 연결하는 `Bridge` 층이다.
- `\alpha_s^{1/d}`와 `\sin^2\theta_W=4\alpha_s^{4/3}` 체인은 현재 핵심 정본 기준으로 `Bridge` 층이다.
- 블랙홀, 뇌, AGI, LLM 문서는 코어 정전의 직접 증명이라기보다 `Phenomenology` 또는 구조적 응용 층으로 읽는다.

등급별 근거 위치는 `Exact`는 `axium.md`·`경로적분.md` 초반,
`Selection`은 `경로적분.md` 15절·`1_강의/`, `Bridge`는
`경로적분.md` 3절·`2_경로적분과_응용/10_공리_정당화.md`,
`Phenomenology`는 `경로적분.md` 후반·`상수.md`·응용 문서를 먼저 본다.

## 4. 대표 예제

### 4.1 자기재귀 고정점

$D_{\mathrm{eff}}\simeq3.178$을 사용하면 물리 가지의 고정점은

$$
x=e^{-(1-x)D_{\mathrm{eff}}}
\quad\Longrightarrow\quad
x\simeq0.0486466333
$$

이다. Newton 방법과 bracketed solver가 같은 해를 주며 수치 residual은 약 $2.1\times10^{-17}$이다.

이 단계에서 고정점 계산은 `Exact/Selection`에 가깝다. 이를 현재 우주의 $\Omega_b$로 동일시하는 마지막 단계는 `Bridge`다.

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

CE는 우주의 성분을 먼저 “선택되어 $d=3$ 표준모형 자유도로 나타난 부분”과 “선택되지 않은 Clarus 장의 암흑 부문”으로 나눈다.

$$
\Omega_{\mathrm{total}}
=
\underbrace{x}_{\Omega_b}
+
\underbrace{(1-x)}_{\Omega_{\mathrm{dark}}}.
$$

고정점 해 $x\simeq0.04865$를 바리온 분율과 연결하면

$$
\Omega_{\mathrm{dark}}=1-\Omega_b\simeq0.95135
$$

가 된다. 이 단계는 “현재 우주의 약 95%가 아직 암흑 부문에 있다”는 총량을 정할 뿐, 암흑물질과 암흑에너지를 아직 구분하지 않는다.

CE는 Clarus 장의 암흑 부문을 다음 두 성분으로 해석한다.

| 성분 | CE 해석 | 동역학적 역할 |
|---|---|---|
| 암흑에너지 $\Omega_\Lambda$ | Clarus 장의 진공·0차 배경 성분 | 균질한 배경과 가속 팽창 |
| 암흑물질 $\Omega_{DM}$ | 진공 위에 남은 집합적 요동·응축 성분 | 중력적으로 뭉치며 구조 성장에 참여 |

두 성분의 비를

$$
R\equiv\frac{\Omega_{DM}}{\Omega_\Lambda}
$$

로 두면 선도차수 CE 규칙은

$$
R_{\mathrm{LO}}=\alpha_sD_{\mathrm{eff}}\simeq0.37463
$$

이다. 해석은 “유효 차원마다 결합세기 $\alpha_s$만큼 진공 요동이 축적된다”는 연장성 가정이다. 바리온이 세 게이지 층에 주는 피드백을 포함한 현재 대표값은

$$
R_{\mathrm{3layer}}\simeq0.38063
$$

이다. 그러면 전체 암흑 부문을 다음처럼 분할한다.

$$
\Omega_\Lambda
=
\frac{1-x}{1+R}
\simeq0.68907,
$$

$$
\Omega_{DM}
=
\frac{(1-x)R}{1+R}
\simeq0.26228.
$$

세 성분은 정규화를 만족한다.

$$
\Omega_b+\Omega_{DM}+\Omega_\Lambda=1.
$$

또한

$$
\frac{\Omega_{DM}}{\Omega_b}\simeq5.39
$$

이므로 “암흑물질이 보통물질보다 약 5배 많은 이유”를 별도 입자 개수의 우연이 아니라, 생존 고정점 $x$와 암흑 부문 요동비 $R$의 결합으로 설명한다.

#### 암흑물질의 정체

현재 CE가 직접 주장하는 암흑물질은 특정 WIMP 하나가 아니라

$$
\Omega_{DM}
=
\text{Clarus 장 진공 위의 집합적 요동 성분}
$$

이다. 정확한 $Z_2$ 집합 가지에서는 단일 입자 핵반동이 선도차수에서 나타나지 않는다고 읽는다. 일반적인 WIMP식 탄성 산란이 발견되면 이 가지는 반증된다.

$m_\phi\simeq29.65\ \mathrm{MeV}$는 우주 전체 암흑물질 입자의 확정 질량이 아니다. 집합적 장 요동을 국소 스칼라 pole로 표현할 때 생기는 입자언어 `Bridge` 후보다.

#### 암흑에너지의 정체

암흑에너지는 Clarus 장의 요동이 없는 진공 배경 성분으로 읽는다. 밀도 분할값 $\Omega_\Lambda$와, 왜 그 절대 에너지 밀도가 매우 작은가 하는 우주상수 문제는 서로 다른 질문이다.

- $\Omega_\Lambda\simeq0.6891$: 전체 에너지 중 차지하는 무차원 분율
- $\rho_\Lambda$: 실제 진공 에너지의 절대 스케일

문서에는 de Sitter 엔트로피와 위상면적을 이용해 $\rho_\Lambda^{1/4}\sim2.24\ \mathrm{meV}$를 읽는 홀로그래피 후보가 있다. 이 부분은 미시적 양자중력 유도가 남아 있으므로 `Exact`가 아니라 `Bridge/Phenomenology`다.

#### 현재 판정과 역사

암흑 부문 분할 규칙은 다음 순서로 발전했다.

1. 초기에는 $R\simeq\alpha_s\pi$를 사용했다.
2. 이미 유도된 유효 차원을 재사용하는 $R_{\mathrm{LO}}=\alpha_sD_{\mathrm{eff}}$로 교체했다.
3. 바리온이 $U(1)$, $SU(2)$, $SU(3)$ 층에 주는 관성 피드백을 포함해 $R_{\mathrm{3layer}}\simeq0.38063$으로 정제했다.

수치 정합은 좋지만, “진공 성분 = 암흑에너지”와 “집합적 요동 = 암흑물질”의 미시적 동일시는 아직 독립적으로 증명되지 않았다. 따라서 현재 공식 판정은 둘 다 `Phenomenology`다. 특히 밀도비를 맞추는 것과 은하 halo, 렌즈, CMB 성장률을 동시에 재현하는 것은 별도의 검증 문제다.

관련 문서:

- [경로적분.md](경로적분.md) 3절
- [3_상수/3_부트스트랩.md](3_상수/3_부트스트랩.md)
- [3_상수/7_우주론.md](3_상수/7_우주론.md)
- [2_경로적분과_응용/02_에스컬레이터.md](2_경로적분과_응용/02_에스컬레이터.md)

#### 코어 보강과 남은 간극

- `P_{\text{survive}} \leftrightarrow \Omega_b`는 유클리드 등분배 정리 + 부트스트랩 자기일관성 + Cauchy 유일성으로 정당화되었으며, `Selection` 근접 `Bridge`로 격상되었다. 잔여 불확도: 상호작용 보정 $O(\alpha_s/(2\pi)) \sim 2\%$.
- `\alpha_s^{1/d}` 및 `\sin^2\theta_W = 4\alpha_s^{4/3}` 체인은 수치적으로 강하며, 독립 실험 검증 경로(FCC-ee, 격자 QCD, EIC)가 `상수.md`에 정리되었다. 코어 기준으로는 `Bridge`.
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
\sin^2\theta_W=4\alpha_s^{4/3}
$$

이다. 수치 정합은 강하지만, 왜 지수가 정확히 $4/3$이어야 하는지에 대한 독립적인 게이지 이론 유도는 아직 완전히 닫히지 않았다. 따라서 현재 등급은 `Bridge`다.

검산:

```powershell
python examples/physics/alpha_s_closure_gate.py
```

관련 문서:

- [3_상수/2_혼합매개변수.md](3_상수/2_혼합매개변수.md)
- [2_경로적분과_응용/10_공리_정당화.md](2_경로적분과_응용/10_공리_정당화.md)

### 4.4 물리 예측과 사후 readout의 구분

$V_{cb}$, $V_{us}$, $A_s$, $n_s$ 같은 항목은 단순 원식이 바로 관측을 맞힌 경우와, loop 보정·projection·전이수 선택을 받아 통과한 경우를 구분해야 한다.

예를 들어 $V_{cb}$의 LO 식은 엄격한 기준에서 실패하지만,

$$
|V_{cb}|=\alpha_s^{3/2}\left(1+\frac{\delta}{2\pi}\right)
$$

라는 NLO projector 후보는 관측 범위에 들어온다. 현재 판정은 `Exact`가 아니라 조건부 `Phenomenology`다.

검산:

```powershell
python examples/physics/ckm_vcb_nlo_gate.py
python examples/physics/primordial_spectrum_readout_gate.py
python tests/scorecard.py
```

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
m_\phi=m_p\delta^2\simeq29.65\ {\rm MeV}
$$

는 이 pole readout의 사전등록값이다. 해당 질량창의 신호는 pole bridge를 지지하지만, 신호 부재는 우선 국소 스칼라·포탈 readout을 제약한다. 코어 장 자체의 반증 조건은 별도로 명시해야 한다.

검산:

```powershell
python examples/physics/clarus_boson_search_gate.py
```

## 5. 현재 검증 상태

가장 보수적인 현재 판정은 다음과 같다.

| 영역 | 현재 상태 |
|---|---|
| 생존함수의 함수형 | 정규성·곱셈 조건 아래 `Exact/Selection` |
| 자기재귀 고정점과 solver | 수학·코드 수준 `Exact` |
| $d=3$ Hodge 선택 | `Selection` |
| $P_{\mathrm{survive}}\leftrightarrow\Omega_b$ | `Bridge` |
| 결합상수·혼합각 | 주로 `Bridge` |
| DM/DE·초기 스펙트럼·Hubble readout | 주로 `Phenomenology/Open` |
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
4. `tests/scorecard.py`
5. `examples/physics/`

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
| 문서군 전체 완성도와 다음 정리 순서는 무엇인가 | [0_검증과감사/문서_전체_완성도_감사.md](0_검증과감사/문서_전체_완성도_감사.md) | [참조/정합성_검증.md](참조/정합성_검증.md) |
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
$env:PYTHONPATH = "reality_stone/python"
python -m pytest tests -q
python tests/scorecard.py
python tests/run_validation.py
```

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

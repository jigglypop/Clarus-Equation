Status: COMPLETE

# 독립 수학·수치 검산

## 범위와 판정 원칙

- 기준 스냅샷은 현재 작업트리다. `RBE/`는 사용자 범위 변경에 따라 코드, 테스트,
  결론 모두에서 제외했다.
- 정본 대조는 `docs/axium.md`, `README.md`,
  `docs/2_경로적분과_응용/00_검증_규약.md`,
  `docs/참조/무차원_감사_수학.md`, `docs/참조/정합성_검증.md`,
  `docs/7_AGI/18_CodeMap.md`로 제한했다.
- 수치 통과는 구현 회귀의 증거일 뿐 정리의 증명이나 물리 사상의 정당화가 아니다.
  외부 웹과 과거 CE run의 상세 결론은 사용하지 않았다.

## C1--C5 판정

| Claim | 실제 지위 | P 등급 | 판정 |
|---|---|---:|---|
| C1 | 미완성 | P1 | 수학 모듈의 책임은 대체로 분리되어 있으나, 공개 집합이 실패한 선택 import를 `None`으로 남길 수 있고 최신 V18b/L3--L8 코어가 코드 지도에 없다. |
| C2 | 미완성 | P0 | 낮은 고정점·등록된 V16/V17 궤도 같은 좁은 정리는 보존되지만, 공개 허용 정의역 전체에는 잘못된 부호를 고르는 완전한 반례와 단위·경계 공백이 있다. |
| C3 | 미완성 | P1 | 집중 테스트 242개는 통과하지만 아래 반례 7종을 방어하지 않는다. 문자열 휴리스틱 무차원 검사는 거짓 양성을 낸다. |
| C4 | 미완성 | P1 | 준비된 `.venv`에서는 재현되지만 `sympy`가 패키지 의존성에 없고, 무차원 검사 본 실행은 오류가 있어도 0으로 끝나며 소스 트리에 보고서를 쓴다. |
| C5 | 산출 | P0 | P0 1건, P1 7건, P2 3건을 아래처럼 재현했다. 보안 결함은 이 수학 레인에서 확정하지 않았다. |

## 정의 → 공리 → 정리 → 산출 복원

| 대상 | 정의역·독립 전제 | 보존 가능한 지위 | 코드 대조 |
|---|---|---|---|
| Poisson 고정점 | $D>1$, $q\in[0,1]$, 최소해 선택 | 최소 소멸 고정점과 $Dq<1$은 정리; $q\mapsto\Omega_b$는 공리 | 정본은 최소 소멸확률을 명시한다(`docs/axium.md:272-300`, `docs/axium.md:351-365`). 레거시 solver는 이를 “survival”로 표기한다(`bootstrap_solver.py:4-7`). |
| 무차원 코어 | exp/log/확률/고정점 인자가 무차원 | 차원 규칙은 정의·공리; 개별 차원 계산은 산출 | 정확 유리 차원 도구는 존재하지만(`dimensionless.py:140-237`), 별도 registry checker는 식의 차원을 계산하지 않는다(`dimensionless_checker.py:462-529`). |
| V15 finite metric | 각 노드의 SPD $g_i$, 동일 단위의 변위·기준 길이 | 유한 이차 길이와 graph cost는 정의·산출 | continuum, curvature, irreversible dynamics를 거짓으로 둔 certificate 경계는 정직하다(`unified_metric.py:797-830`). |
| V16 metric flow | $x\ne0$, $c>0$, $0<\eta\le1$ | 정확산술에서 SPD 보존과 $p_+=p^{1-\eta}c^\eta$는 조건부 정리 | 구현식은 문서화된 rank-one congruence와 일치한다(`covariant_metric_flow.py:7-16`, `covariant_metric_flow.py:441-493`). |
| V17 signed cue | identity 또는 선언한 $\operatorname{diag}(GL(d),1)$ 궤도, unit public reference | 이 좁은 등록 fixture의 비용 $(2,4)$는 산출 | 코드 지도도 extra oriented memory임을 인정한다(`docs/7_AGI/18_CodeMap.md:701-721`). 공개 생성기의 전체 SPD 정의역으로 확대하면 아래 P0 반례가 생긴다. |
| V18b delayed credit | action $a\in\{-1,1\}$, reward $r\in\{0,1\}$가 “정답 여부”라는 외부 의미 | $\tilde y=a(2r-1)$은 그 공리 아래 정리; 업데이트는 산출 | reward의 correctness 의미는 코드가 검증할 수 없는 환경 공리다(`delayed_linear_credit.py:147-156`, `delayed_linear_credit.py:319-336`). 일반 delayed credit 주장은 certificate에서 거부한다. |
| L3--L8 finite host | 유리수 등록 상태·등록 flux·고정 horizon/bit | 등록된 한 단계 값은 산출; open-set·autonomy·brain 주장은 미완성 | exact `Fraction` 계산과 테스트의 “machine pass is not theorem” 경계는 적절하다. 다만 공개 상태 정의역과 $\sigma$ 의존성은 아래와 같이 닫히지 않았다. |

## 단계별 독립 계산

### 1. 고정점과 다른 해 가지

$$
f(q)=q-e^{-D(1-q)},\qquad D=3.17776.
$$

독립 이분법으로 낮은 해는 $q_0=0.048646633337214085$, 잔차는
$6.94\times10^{-18}$이다. 동시에 $q=1$도 정확한 해다. 낮은 해는
$Dq_0<1$로 국소 안정하고 $q=1$은 $D>1$로 불안정하다. 저장소 안의 더 좁은
`bootstrap_fixed_points`도 두 가지를 모두 열거한다(`core_axioms.py:326-344`).
그러나 공개 Newton 함수는 초기값 1에서 1을 반환하고 검증기는 이를
`equation_satisfied=True`로 받는다(`bootstrap_solver.py:54-95`,
`bootstrap_solver.py:222-240`). 잔차만으로 최소 가지를 증명하지 못한다.

선언된 bridge를 정의에서 다시 계산하면
$\sin^2\theta_W=0.23122206826075514$,
$\delta=0.17775842340997383$,
$D=3.1777584234099736$이다. solver의 반올림 상수와 차이는
$1.57659\times10^{-6}$, 낮은 해의 차이는 $8.63\times10^{-8}$이다. 이는 현재
$10^{-9}$ 잔차 판정과 다른 종류인 입력 반올림이며 별도 장부가 필요하다.

### 2. 무차원 감사

| 코어 인자 | 차원 벡터 | 상태 | 숨은 전제 |
|---|---:|---|---|
| $(1-q)D$ | $(0,0,0,0)$ | 무차원 | $q,D$가 확률·평균 자손수라는 정의 |
| V15 $d_g^2/\ell_0^2$ | $(0,0,0,0)$ | 조건부 무차원 | metric cost와 reference scale이 같은 길이 단위 |
| V16 $\log(p/c)$ | $(0,0,0,0)$ | 조건부 무차원 | $p,c>0$이고 같은 squared-cost 단위 |
| V17 $(C_{wrong}-C_{correct})/C_{correct}$ | $(0,0,0,0)$ | 조건부 무차원 | 두 비용의 공통 단위와 양의 분모 |
| V18b $a(2r-1)$ 및 $\eta\tilde y e$ | $(0,0,0,0)$ | 합성 모형에서 무차원 | $a,r,\eta,e$를 순수수로 채택 |

V15/V16 코드는 단위를 타입으로 보존하지 않고 양수·finite만 검사하므로 위 상태는
런타임 정리가 아니라 입력 공리다. 더 심각하게 registry checker는 식에 `/`가 있으면
기대 차원이 무차원이라는 이유만으로 PASS할 수 있다(`dimensionless_checker.py:516-522`).
반례 `mass/time`도 `PASS`였다.

### 3. 경계·대칭·공통 모수 섭동

| Probe | 독립 결과 | 무너지는 범위 |
|---|---|---|
| V17 off-orbit SPD | $G=\begin{pmatrix}50.5&-49.5\\-49.5&50.5\end{pmatrix}$, cue $s=+1$은 prewrite $p=2$라 허용되지만 terminal 비용이 $(-,+)=(4,200)$이고 $-1$을 선택 | 모든 공개 허용 SPD 상태에서 signed cue가 정확하다는 넓은 API 주장. identity/embedded-chart 궤도 정리는 보존. |
| V15 node-scale 교체 | 한 노드가 $10^{308}I$이면 다른 노드의 antisymmetry $2\times10^{100}$가 허용되고 $I$로 조용히 평균됨 | “각 노드에서 symmetric 입력”이라는 검증 범위(`unified_metric.py:206-220`). |
| V16 공통 metric scale | 비용비가 항상 4인데 $(10^{-16},4\cdot10^{-16})$는 tie, $(1,4)$는 unique | tie 집합의 양의 공통 scale 불변성; `max(1,cost)`가 숨은 단위 1을 도입(`covariant_metric_flow.py:512-536`). |
| L8 $\sigma$ A/B 교체 | 같은 $I=1$에서 $\sigma=0,1$의 sensor/action 다음 상태가 동일 | 공개 $H\to H$에서 $\sigma$가 transition에 참여한다는 넓은 읽기. 구현은 `named_i`만 사용(`universe_life_kernel.py:848-875`). |
| HybridState 직접 경계 | `HybridState(2,0,0)`이 생성됨 | “cube state”의 $[0,1]^3$ 정의역. 검사는 `from_values`에만 있음(`universe_life_kernel.py:141-160`). |

### 4. 교차 예측, dof와 look-elsewhere

- V16/V17/V18b와 L3--L8 수치는 등록된 합성 fixture의 산출이다. 테스트 수를 관측
  자유도나 독립 예측 수로 세지 않는다.
- `tests/run_validation.py`의 scorecard는 전체 23행 중 12행만 채점하고 외부 입력·
  exact/reference·open test를 분모에서 제외한다. 11 PASS, 1 CAUTION이지만 여러 후보식에
  대한 global covariance와 look-elsewhere correction은 허용 문서에 없으므로 전역
  유의도는 미완성이다.
- `proof_completion_attempt.py`는 LO $V_{cb}$ $-6.58\sigma$, tree $V_{us}$
  $+9.84\sigma$, raw $A_s$ $+197.80\sigma$의 obstruction을 보존하고 대체 readout을
  candidate로만 둔다. 이 후보 선택을 독립 예측으로 승격하지 않는다.

## P0/P1/P2 발견과 최소 수정 범위

### P0

1. **M-P0-1 — V17 공개 정의역의 완전한 wrong-sign 반례.**
   `make_state_from_metric`은 임의 SPD를 받고(`homogeneous_signed_cue.py:154-160`),
   `write_cue`는 $p=2$만 확인한다(`homogeneous_signed_cue.py:197-220`). 위 $G$는 이
   조건을 만족하지만 `readout`이 cue와 반대 부호를 고른다(`homogeneous_signed_cue.py:240-256`).
   최소 수정은 생성기를 identity/선언 chart 궤도로 제한하거나 그 궤도의 완전한
   불변식을 검사하는 것이다. 코드 지도의 비용 $(2,4)$ 문장은 등록 궤도로만 좁혀야
   하며, 좁은 정리 자체는 삭제하지 않는다.

### P1

1. **M-P1-1 — 무차원 checker의 거짓 양성과 non-failing main.** 실제 차원 환경 없이
   문자열로 PASS를 정하고, 실행 시 5 parse error가 있어도 종료 0이다. 또한 소스 폴더에
   `dimensionless_audit.txt`를 쓴다(`dimensionless_checker.py:484-529`,
   `dimensionless_checker.py:600-618`). expression AST와 symbol dimension registry로
   실제 벡터를 계산하고 ERROR 시 nonzero로 끝내야 한다.
2. **M-P1-2 — 최소 고정점 가지를 API가 강제하지 않는다.** Newton 초기값 1 반례와
   extinction/survival 표기 충돌을 닫으려면 $[0,1/D]$ bracket 및 $Dq<1$ 검사를 공개
   solver 계약에 넣고 물리 readout은 공리로 분리해야 한다.
3. **M-P1-3 — V15 symmetry tolerance가 노드 전역 scale에 오염된다.** tolerance를 노드별
   scale로 계산하고 antisymmetric residual을 각 노드에서 검사해야 한다.
4. **M-P1-4 — V16 tie tolerance의 숨은 단위 1.** 비용 단위를 명시적으로 정규화하거나
   순수 상대/ULP 판정을 사용해 양의 공통 scale 아래 minimizer 집합을 보존해야 한다.
5. **M-P1-5 — finite host의 공개 정의역·의존성 공백.** `HybridState.__post_init__`에서
   cube를 강제하고, `internal_kernel`의 도메인을 등록 집합 $S$로 제한하거나 $\sigma$의
   전이 역할을 명시해야 한다.
6. **M-P1-6 — 공개 API가 import 실패를 `None`으로 숨긴다.** 신규 수학 이름은 먼저
   `None`으로 선언되고(`clarus/__init__.py:69-97`), `ImportError`를 삼킨 뒤
   `__all__`에 남는다(`clarus/__init__.py:230-267`, `clarus/__init__.py:516-544`).
   선택 기능 proxy 또는 명시적 import error가 필요하다.
7. **M-P1-7 — 설치·지도 불일치.** `dimensionless_checker.py:20`의 `sympy`는 두
   `pyproject.toml` 의존성 목록에 없다(`pyproject.toml:6-12`,
   `reality_stone/pyproject.toml:9-14`). 코드 지도는 V17에서 끝나지만 현재 V18b/L3--L8
   공개 코어가 존재한다.

### P2

1. `docs/7_AGI/18_CodeMap.md:303`의 식
   $4/(e^{4/3}\pi^{4/3})$은 직접 계산하면 $0.2291575578$인데 표에는 `0.1726...`이다.
2. `bootstrap_solver.py:31`은 값 `0.00005` 옆 주석을 `±0.0005`로 적어 10배 불일치한다.
3. Layer-A 기준 슬라이스는 `runtime.py:395-402`의 sparse invariant 경고 2개를 낸다.
   일반 실행은 통과하지만 `-W error`에서는 1 실패다.

## 동적 검증 결과

| 명령군 | 종료 | 결과 | 수학적 해석 |
|---|---:|---|---|
| focused current math/core pytest, `-W error` | 0 | 242 passed, warning 0 | 등록 회귀 통과; 반례 부재의 증명 아님 |
| CE baseline pytest, 일반 | 0 | 61 passed, warning 2 | sparse CSR 경고 분리 |
| 같은 baseline, `-W error` | 1 | 60 passed, 1 failed | warning hygiene 실패, 산술 불일치 아님 |
| `tests/run_validation.py` | 0 | bootstrap residual $2.08\times10^{-17}$; scorecard 23/12, 11 PASS, 1 CAUTION; dimension 7/7; OVERALL CAUTION | 물리 사상·look-elsewhere를 닫지 않음 |
| `bootstrap_solver.py` | 0 | Newton/Brent 차이 $1.25\times10^{-13}$; SciPy SKIP | 낮은 가지의 수치 일치만 확인 |
| `proof_completion_attempt.py` | 0 | obstruction 3개와 candidate readout 분리 | candidate는 경험식/미완성 |
| `artifacts/math_counterexamples.py` | 0 | P0/P1 반례 7종 재현 | 정의역·대칭·scale·다른 가지 감사 |

## 재현 명령과 scratch

Scratch: `artifacts/math_counterexamples.py`

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='reality_stone/python'
.\.venv\Scripts\python.exe '_workspace/ce/repository-code-analysis-20260815/artifacts/math_counterexamples.py'
```

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='reality_stone/python'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -W error -q tests/test_bootstrap_solver.py tests/test_dimensionless.py tests/test_unified_metric.py tests/test_covariant_metric_flow.py tests/test_homogeneous_signed_cue.py tests/test_delayed_linear_credit.py tests/test_universe_life_kernel.py tests/test_l3_ne2_open_set.py tests/test_l3_nonlinear_las.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py tests/test_l7_region_loop.py tests/test_l8_internal_kernel.py
```

직접 `dimensionless_checker.py`를 실행하면 소스 트리에 보고서를 쓰므로 read-only 재현에는
사용하지 않는다. 위 scratch가 거짓 양성을 부작용 없이 재현한다.

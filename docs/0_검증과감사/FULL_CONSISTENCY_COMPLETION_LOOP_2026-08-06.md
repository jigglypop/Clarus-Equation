# CE 전체 정합성 완성 루프 — 2026-08-06

## 1. 목적과 완료의 뜻

이 문서는 2026-08-06 전수 감사에서 발견된 수학·물리·수치·문서
불일치를 최신 정본에 실제로 반영하는 변경 계약이다. 여기서 `완료`는
상태 라벨만 바꾸는 일이 아니다.

1. 직접 반례가 있는 식은 올바른 식과 증명으로 교체한다.
2. 같은 기호가 다른 타입을 가리키지 않게 한다.
3. 입력, calibration, prediction, holdout을 식별 가능한 장부로 고정한다.
4. 물리 주장을 유지하려면 그 결론을 내는 작용·경계조건·관측량을 함께
   제시한다.
5. 단위 테스트, 항등식 검산, 관측 likelihood를 서로 다른 gate로 둔다.
6. 어떤 결론도 통과하지 않은 gate를 통과한 것으로 기록하지 않는다.

거짓 명제를 `Exact`라고 유지하는 것은 완성이 아니라 모순의 은폐다.
따라서 이 루프는 기존 목표를 가능한 한 보존하되, 목표를 지지하는
정확한 대체 정리나 모형을 문서 안에 구축하는 방식으로 닫는다.

## 2. 정본 우선순위

충돌 시 다음 순서를 적용한다.

1. 이 완성 루프와 `PROOF_STATUS_MATRIX.md`
   및 `OBSERVATIONAL_BASELINE_2026-08-06.md`
2. `axium.md`의 타입·작용 규약
3. `경로적분.md`의 교정된 계산 사슬
4. `상수.md`의 입력·출력 provenance 표
5. 분야별 세부 문서
6. `1_강의/`, `참조/`의 역사적 설명

낮은 우선순위 문서가 높은 우선순위 문서보다 강한 결론을 내리면 높은
우선순위의 정확한 정리와 식으로 자동 해석한다.

## 3. 전역 타입 계약

| 기호 | 타입 | 단위/정의 | 금지되는 자동 동일시 |
|---|---|---|---|
| $\phi(x)$ | 공변 EFT의 독립 물리 scalar | 4D canonical scalar는 질량차원 1 | $\Phi_H$, $R$, 임의 데이터장 |
| $\Phi_H[\gamma,\eta]$ | Hessian probe readout | 경로좌표·measure convention 의존 | local scalar, stress tensor |
| $q$ | 유클리드 smoothing 대상장 | 응용별 명시 | 물리 입자 |
| $R$ | Ricci scalar | 질량차원 2 | Hessian 자체 |
| $x=\varepsilon^2$ | 낮은 생존 고정점 | 무차원 | $1-x$, $\alpha_s$ |
| $y=1-x$ | 생존 여집합 | 무차원 | $x$ |
| $\Omega_b$ | 임계밀도 대비 바리온 밀도 | 무차원 | $\omega_b=\Omega_bh^2$ |
| $\alpha_1$ | 지정 정규화의 hypercharge coupling | scheme·scale 필수 | $\alpha_{em}$ |
| $\alpha_s$ | QCD coupling | scheme·scale 필수 | $x$ 또는 $\Omega_b$ |

## 4. 입력·출력 계약

| 양 | canonical 역할 | 필수 provenance |
|---|---|---|
| $d=3$ | 채택한 물리 차원/Selection | Hodge-type 모형선택과 관측 사실을 분리 |
| $\alpha_s(M_Z)$ | Track A에서는 입력, Track B에서는 $\alpha_{em}(M_Z)$ 입력 아래 조건부 출력 | $\overline{\rm MS}$, $M_Z$, 데이터 릴리스 |
| $\alpha_{em}(M_Z)$ | Track B의 외부 SM-running 결과 | threshold와 HVP covariance |
| $v_{\rm EW},M_Z,m_p,M_{\rm Pl}$ | 서로 다른 차원 readout의 calibration anchors | 어느 공식에 어느 anchor가 쓰였는지 기록 |
| $H_0,r_d,T_{\rm CMB}$ | 우주론 forward model 입력 또는 holdout | 같은 likelihood에서 역할 고정 |
| $x$ | 지정 $D_{\rm eff}$에서 푼 수학적 출력 | 가지와 solver interval 기록 |
| $\Omega_b$ | energy-biased survivor observable을 쓴 동역학 모형의 출력 | stress/transfer/Boltzmann bridge 필요 |

`외부 입력 1개`와 `조정 파라미터 0개`는 다른 문장이다. 앞으로는 두
수를 동시에 기록한다.

## 5. 수학 완성 루프

### M1. 부트스트랩 변수

$$
x=e^{-D(1-x)},\qquad
D(x)=-\frac{\ln x}{1-x}.
$$

여집합 $y=1-x$를 쓰면

$$
D(y)=-\frac{\ln(1-y)}{y}.
$$

두 식을 같은 변수 이름으로 쓰지 않는다. 일반화
$x=e^{-\kappa D(1-x)}$의 역산은
$\kappa=-\ln x/[D(1-x)]$다.

### M2. 경로확률과 energy-weighted measure

Gamma shape $a$와 threshold $t$에서

$$
P_a(t)=\frac{\gamma(a,t)}{\Gamma(a)},\qquad
Q_a(t)=\frac{\gamma(a+1,t)}{\Gamma(a+1)},
$$

이며 정확히

$$
P_a(t)-Q_a(t)=\frac{t^ae^{-t}}{\Gamma(a+1)}.
$$

따라서 사건확률과 energy-weighted mass는 일반적으로 다르다. CE에서
물질화 분율로 쓸 양은 원래 path count가 아니라, 지정 stress-energy
functional $E[\gamma]\ge0$로 정의한

$$
P_E(A)=
\frac{\int_A E[\gamma]e^{-S_E[\gamma]/\hbar}\,\mathcal D\gamma}
{\int E[\gamma]e^{-S_E[\gamma]/\hbar}\,\mathcal D\gamma}
$$

다. $P_E(A)=\Omega_b$가 되려면 $A$가 baryon-number를 운반하는 최종
상태 sector이고, 다른 sector로의 energy transfer를 포함한 unitary 또는
CPTP evolution, 재가열·BBN·Boltzmann 진화를 통과해야 한다.

### M3. Lorentzian과 Euclidean PDE

Lorentzian scalar는 globally hyperbolic 배경, Cauchy data와 energy
estimate로 푼다. Euclidean 문제만 coercive Sobolev functional과
elliptic maximum principle을 사용한다. 무감쇠 Klein--Gordon 에너지는
보존되고, 감소정리는 $+\Gamma\partial_t\phi$가 명시된 경우에만 쓴다.

## 6. 물리 완성 루프

### Q0. 공변 EFT

`axium.md`의 최신 최소 EFT를 공통 출발점으로 한다. Hessian readout과
독립 scalar를 분리하고, stress tensor는 metric variation으로만
정의한다. 고차미분 smoothing은 유클리드 functional에 두며 Lorentzian
기본 작용에 자동 삽입하지 않는다.

### Q1. 결합상수

- Track A: PDG 2026 QCD review Eq. (9.25)의
  \(\alpha_s^{\overline{\rm MS}}(M_Z)=0.1180\pm0.0009\)를 calibration
  input으로 두고 \(s_A^2:=4\alpha_s^{4/3}\)를 **CE registered
  output**으로 계산한다. \(s_A^2\)를 on-shell,
  \(\overline{\rm MS}\), effective leptonic 약혼합각 중 하나와 비교하려면
  RG·threshold·scheme map을 먼저 닫는다.
- Track B: 독립적으로 계산된 \(\alpha_{em}(M_Z)\)를 입력하고 sum rule과
  CE 관계를 함께 풀어 두 양의 \(\alpha_s\) 근을 모두 보존한다. branch
  selection과 물리적 \(s_W^2\) scheme 식별은 별도 조건이다.

관측된 \(s_W^2\)를 넣어 \(\alpha_s\)를 얻은 뒤 같은 각을 다시 검증하는
경로, 또는 scheme이 다른 약혼합각끼리 오차 점수를 매기는 경로는 사용하지
않는다.

### Q2. flavour

CKM/PMNS는 관측량마다 보정인자를 붙이지 않고, 하나의 Yukawa/질량
행렬을 대각화해 unitary matrix 전체와 invariant를 공동 산출한다.
개별 원소 공식은 이 공동 construction의 근사식일 때만 유지한다.

### Q3. strong CP

CP-even real singlet만으로 $\bar\theta$를 0으로 만들지 않는다. 해결
목표를 유지하는 완성 branch는 shift symmetry를 가진 pseudoscalar
$a/f_a$와 $G\widetilde G$ 결합 또는 명시적 UV CP/Nelson--Barr 질량
구조를 작용에 추가하고, vacuum minimization과 nEDM bound를 함께
검증한다.

### Q4. 우주론

background density만 맞추지 않고 같은 parameter vector로 CMB, BAO,
SN, growth의 covariance likelihood를 계산한다. 진공형 stress와
pressureless clustering stress를 별도 장/상태로 유도하지 못한 경우
tree/loop라는 이름만으로 DE/DM을 식별하지 않는다.

## 7. 응용 완성 루프

- Casimir 계산은 $E/A$와 총 $E=A(E/A)$를 분리한다.
- 효율은 한 cycle의 모든 입력·reset·손실 에너지를 같은 단위로 합산한다.
- 초전도 mediator는 band cutoff 아래에서 integrate out한 Wilson
  coefficient로 Eliashberg/Keldysh 식에 들어간다.
- AGI 확률 map은 simplex를 보존하도록 모든 성분을 정규화하고 실제
  Jacobian bound를 증명한다.
- Riemann attention은 하나의 canonical phase convention, real spectral
  weight 또는 conjugation 조건으로 Hermiticity를 보장한다.

## 8. 반복 통과조건

구조·의미·수치 전수 gate는 다음처럼 실행한다. 이 명령은
[`CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)을
읽어 [`verify_numeric_consistency.py`](verify_numeric_consistency.py)도 함께 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

| 구분 | 실제 검사 | 범위와 통과조건 |
|---|---|---|
| A1 자동 구조 | PowerShell 전수 gate | H1·내부 링크·delimiter·C0 control·GFM 표·cross-line inline math·prose raw TeX 오류 0 |
| A2 자동 수치 | `verify_numeric_consistency.py` | manifest에서 bootstrap, Track-A, density, portal, scalar kernel, finite-\(\xi\) inflation, 고정 \(\chi^2\)의 \(p\)-값을 재계산해 tolerance 통과 |
| A3 자동 의미 sentinel | `verify_document_contract.py` | canonical 값·역할, stale active snapshot, AGI 안정성/의식/수면 과장, 완료 장부 drift 0 |
| M1 수동 타입·작용 감사 | 정본 action/type 문서와 분야별 식 대조 | Lorentzian/Euclidean, field/readout, 질량차원, EOM·stress·pole·vertex의 필요한 가정이 문서에 명시 |
| M2 수동 provenance·통계 감사 | 입력/fit/calibration/holdout과 covariance 문장 대조 | 같은 값을 입력과 예측으로 중복 계상하지 않고 post-hoc 결과를 holdout으로 부르지 않음 |
| E1 경험적 gate | 실제 공개/압축 데이터에 대한 구현 결과 | 통과 여부를 그대로 기록. 현 fixed-background DESI 두 패키지와 canonical Higgs-portal benchmark는 `REJECT` |

A1--A3은 문서와 manifest의 회귀를 막지만 일반 정리 증명기나 full
likelihood 재실행기가 아니다. 특히 A2의 우주론 \(p\)-값 검산은 장부에
고정된 \(\chi^2\)에서 재계산하며 DESI likelihood 자체를 다시 fit하지
않는다. M1--M2의 범위와 증거 문서는 최종 실행 장부에 열거한다.

## 9. 변경 장부

| 루프 | 대상 | 상태 |
|---|---|---|
| L0 | 정본·타입·입력 계약 | `IMPLEMENTED` |
| L1 | $\chi$/heat/Poisson 분리와 최소 공변 EFT | `IMPLEMENTED` |
| L2 | A3b·bootstrap·PDE 수학 교체 | `IMPLEMENTED` |
| L3 | 결합상수·입자·우주론 재구성 | `IMPLEMENTED` |
| L4 | 공학·AGI·Riemann 재구성 | `IMPLEMENTED` |
| L5 | 전 문서 동기화·자동검사 | `IMPLEMENTED` |

여기서 `IMPLEMENTED`는 해당 문서 계약과 수정 루프가 실제 파일에
반영되었다는 뜻이다. 열린 Bridge가 증명되었다거나 경험적 gate가
통과했다는 뜻이 아니다. 따라서 fixed-background DESI와 canonical
Higgs-portal benchmark의 `REJECT` 판정은 강등하거나 숨기지 않고 그대로
유지한다.

## 10. 최종 실행 장부

### 10.1 문서·수치 gate

| gate | 실행 결과 | 판정 범위 |
|---|---|---|
| A1 구조·수식 문법 | Markdown 212개 검사 통과 | H1, 내부 링크, delimiter, control character, 표, cross-line inline math, prose raw TeX |
| A2 정본 수치 | `verify_numeric_consistency.py` 통과 | bootstrap, Track A, density, portal, scalar kernel, finite-$\xi$ inflation, 고정 우주론 통계의 재계산 |
| A3 제한 의미 계약 | `verify_document_contract.py` 통과 | 정본 값·역할, 활성 구형값, 완료 장부, AGI 안정성·의식·수면 sentinel |
| M1 작용·타입 감사 | 완료 | `axium.md`, `경로적분.md`, `MATHEMATICAL_PHYSICS_ISSUES.md`, `Q0_0_Q0_3_MINIMAL_MANIFEST.md`, `5_유도/` 및 분야별 문서를 대조 |
| M2 provenance·통계 감사 | 완료 | PDG 2026 기준선, $s_A^2$ scheme 경계, $A_s$ 입력 역할, Hubble post-hoc calibration, portal·고정 우주론 `REJECT`를 명시 |

M1--M2는 전 문서에 계약을 반영한 수동 의미 감사이며 일반 정리의
기계증명이나 원시 관측자료의 full-likelihood 재분석은 아니다.

### 10.2 코드 회귀 검사

| 검사 | 결과 |
|---|---|
| 핵심 bootstrap·차원·공리·Bridge gate | `48 passed` |
| fusion·Clarus·Casimir·portal·cosmology 선택 검사 | `248 passed` |
| 저장소 전체 `pytest`, writable basetemp | `1653 passed, 13 skipped, 24 failed` |

전체 저장소의 24개 잔여 실패는 다음 네 묶음으로 분리했다.

1. `test_external_field_to_matter.py` 14개: 추적 benchmark의 내용은
   LF 정규화 hash와 일치하지만 Windows checkout의 CRLF byte hash가
   고정 기대값과 다르다.
2. local-memory verifier 5개: `artifacts/agi/` 아래 preregistration,
   confirmatory, proof 산출물이 저장소에 없다.
3. origin-life 3개: 추적 artifact와 builder/verifier의 artifact
   version이 어긋나고 reconstruction-loop status artifact가 없다.
4. GPT-2 manifold 2개: Hugging Face GPT-2 다운로드 연결이 강제로
   종료되어 재시도도 완료되지 못했다.

이 네 묶음은 `docs/` 바깥 코드·artifact 또는 외부 네트워크 경계에
있으므로 이번 문서 쓰기 범위에서 결과를 조작하지 않았다. 특히
`reality_stone` runtime의 구형 상수도 문서에서 `CANONICAL_DRIFT`로
격리했으며 runtime이 동기화됐다고 주장하지 않는다.

### 10.3 변경·복구 경계

- 변경 범위는 `docs/`이며 commit과 push는 수행하지 않았다.
- 검증용으로 만든 `docs/.pytest_tmp_full_20260806`은 최종 기록 후
  삭제했다.
- 파일별 변경은 `git diff -- docs`로 검토할 수 있고, 기존 사용자
  변경과 문서 밖 untracked 파일은 건드리지 않았다.

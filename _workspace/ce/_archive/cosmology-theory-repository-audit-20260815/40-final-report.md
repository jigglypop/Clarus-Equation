Status: COMPLETE

# CE 우주론 이론·코드 전수 감사 최종 보고서

## 초록

이번 감사는 현재 dirty checkout에서 우주론 주장에 도달하는 71개 파일,
26,602줄과 116개 테스트 함수를 추적했다. 결론은 **CE를 현재 상태에서
검증된 우주론 또는 독립 예측 이론으로 승인할 수 없다**는 것이다. 다만
$D>1$ Poisson 고정점 정리, 외부 경계조건을 명시한 평탄 FLRW/CPL/GR
계산, 같은 Planck convention의 Friedmann--de Sitter entropy 항등식은
보존된다. 밀도 사상, Hubble readout, cumulative 성장, primordial
projector와 우주상수 절대척도 경로는 각각 독립 공리·완전 반례·사후 선택
또는 동일 입력의 재표현 때문에 예측 지위를 갖지 못한다. 관측 레인에서는
잘못된 DESI 요약값과 식별 불가능한 hybrid baseline을 발견했고, 현재
독립 future holdout은 없다. 따라서 이번 run에서 새로 승인되는
`[예측]`은 **0개**이며 제품 소스는 수정하지 않았다.

## 1. 최종 판정

이 절은 계약의 C1--C6을 이론 지위와 코드 상태로 분리한다. 형식 감사의
`Gate: PASS`는 반례 있는 부모 주장을 활성 결론에서 모두 제외해 감사가
모순 없이 닫혔다는 뜻일 뿐, 이론·출판·release 승인이라는 뜻이 아니다.

| 계약 명제 | 실제 지위 | 판정 |
|---|---|---|
| C1 고정점 | `[정리]`과 선택한 $D$의 `[산출]` | **보존** |
| C2 $q_{\rm ext}$와 밀도분율의 동일시 | `[공리: 물리 사상]`; DM/DE 분할은 `[경험식]` | 유도·예측 표기 **제외** |
| C3 평탄 FLRW/CPL/GR | `[공리]` 아래 배경·거리·기본 성장은 `[산출]` | 조건부 **보존**; cumulative branch 제외 |
| C4 원시 스펙트럼, $H_0$, 우주상수 | 항등식 일부만 `[정리]`; 나머지는 `[경험식]` 또는 `[미완성]` | 독립 절대척도 예측 **없음** |
| C5 관측 판별 | 이미 본 자료의 `[경험식]` 진단 | confirmatory 증거 **없음** |
| C6 테스트 | 등록 구현 계약의 `[산출]` | 물리 이론의 증명이 아님 |

현재 살아 있는 결론은 네 묶음뿐이다. 첫째, 고정점의 두 양의 가지와 작은
가지의 유일성·국소 안정성이다. 둘째, 선택한 무차원 $D$에서의 수치근이다.
셋째, 공급된 평탄 우주론 경계조건에서의 표준 forward calculation이다.
넷째, convention을 고정한 entropy 항등식과 SPD가 별도로 보장된
covariance 이차형식이다.

## 2. 정의와 독립 전제

이 절은 서로 다른 수학 층과 물리 층이 한 등식으로 합쳐지는 것을 막는다.

**[정의]** $q_{\rm ext}\in(0,1]$는 Poisson branching toy model의 최소
소멸확률이고 $D>1$은 무차원 평균 자손수다. $a>0$는 scale factor,
$z=a^{-1}-1$, $E(a)=H(a)/H_0$이며 모든 지수·로그·확률 인자는
무차원이어야 한다.

**[공리: 물리 사상]** $q_{\rm ext}$를 현재 바리온 밀도분율
$\Omega_b=\rho_b(t_0)/\rho_{\rm crit}(t_0)$와 동일시하는 선택은 고정점
정리에서 나오지 않는다. 고정점 식에는 시간, 임계밀도, stress tensor,
frame 또는 관측 연산자가 없기 때문이다. 암흑물질과 암흑에너지의 분할도
별도 covariant action과 섭동 방정식 없이는 `[경험식]`이다.

**[공리: 우주론 branch]** 평탄성, GR, CPL 형태, $H_0$, $r_d$,
$\sigma_{8,0}$, $T_{\rm CMB}$, $N_{\rm eff}$, 재결합 history와 초기조건은
외부 입력 또는 모형 선택이다. 이 전제 아래 계산된 거리·나이·성장은
조건부 산출이지 CE 코어의 무입력 예측이 아니다.

## 3. 보존되는 고정점 정리

이 절은 저장소에서 독립적으로 완결되는 가장 강한 수학 결과를 증명한다.

**[정리]** $D>1$이면

$$
q=\exp[-D(1-q)]
\tag{1}
$$

은 $q=1$ 외에 $(0,1/D)$에 정확히 하나의 근 $q_*$를 가지며,
$Dq_*<1$이므로 고정점 반복에서 국소 안정하다.

**증명.** $h(q)=\log q+D(1-q)$로 두면 식 (1)의 근은 $h(q)=0$의
근이다. $h''(q)=-q^{-2}<0$이므로 $h$는 엄격히 오목하고,
$h'(q)=q^{-1}-D$이므로 유일한 최대점은 $q=1/D$이다. 또한
$h(0^+)=-\infty$, $h(1)=0$이며

$$
h(1/D)=D-1-\log D>0
\tag{2}
$$

이다. 따라서 $(0,1/D)$에 한 근이 있고 엄격한 오목성 때문에 그 근과
$q=1$ 이외의 근은 없다. 고정점 사상의 도함수는 근에서 $Dq$이므로 작은
근에서는 $Dq_*<1$이다. $□$

**[산출]** exact electroweak snapshot을 사용한
$D=3.1777584234099736$에서는
$q_*=0.04864671964402817$이다. legacy 코드의 반올림된
$D=3.17776$에서는 solver와 독립 계산이
$q_*=0.048646633337\ldots$로 일치한다. 반면 상수 원장의 `0.0487`은
표시 정밀도의 snapshot이며 계약의 $10^{-12}$ exact output으로 사용할 수
없다.

## 4. 조건부로 보존되는 우주론 계산

이 절은 CE 고유 예측과 표준 방정식의 올바른 구현을 구분한다.

**[산출]** 공급된 평탄 FLRW/CPL/GR branch에서 residual forward model의
$E(a)$, comoving distance, dust+$\Lambda$ 나이와 $\mu=1$ 기본 성장 계산은
독립 적분과 일치했다. 예를 들어 $z=1$ 거리는
`6818.454139268602 Mpc`로 독립값과 상대오차
$4.4\times10^{-16}$, $H_0t_0$는 analytic limit와
$1.2\times10^{-12}$ 이내, $a=0.5$ 성장함수는 독립 적분과 상대오차
$1.5\times10^{-7}$였다. 이것은 주어진 입력에 대한 수치 구현의 일관성만
보인다.

**[정리]** non-reduced Planck mass $M_P^2=1/G$를 쓰면 de Sitter horizon
entropy와 Friedmann 식에서

$$
S_{\rm dS}=\frac{\pi M_P^2}{H^2},
\qquad
\rho_\Lambda=\frac{3\Omega_\Lambda}{8}\frac{M_P^4}{S_{\rm dS}}
\tag{3}
$$

가 따른다. 식 (3)은 같은 $H$ scale의 항등식이다. phase-area ansatz를
추가하면 조건부 한-scale readout을 만들 수 있지만, 이를 서로 독립인 두
절대척도 예측으로 셀 수는 없다.

**[산출]** 관측 잔차 $r$와 covariance $C$가 사전에 동결되고 $C$가
대칭 양의 정부호일 때 $r^TC^{-1}r$는 유효한 이차형식이다. 현재 내장
DESI 행렬은 독립 Cholesky 검사를 통과했지만 parser 자체는 SPD를 보장하지
않으므로 이 결론은 해당 fixture와 명시적 SPD 전제에만 적용한다.

## 5. 활성 이론에서 제외한 경로

이 절은 완전 반례 또는 결정적 출처 결손이 있는 부모 주장의 삭제 범위를
기록한다. 아래 경로는 보존된 정리의 전제로 재사용할 수 없다.

| 제외 경로 | 결정적 근거 | 남는 범위 |
|---|---|---|
| 밀도분율 `ce_prediction` | 고정점에는 물리 밀도 사상이 없고 두 live background가 서로 다름 | branch를 밝힌 supplied density만 |
| cumulative $S(a)$ 성장 | log-grid에 uniform-grid Simpson을 적용해 $a\simeq0.1$에서 약 +33.9% | 별도 구현인 기본 residual 성장만 |
| 현재 Hubble closure | 복사 항 누락; `omega_b h2`를 100배 바꿔도 acoustic angle 불변 | 수정 전 toy 연구 아이디어만 |
| 무입력 우주상수 절대척도 | 같은 $H_0$--entropy--density scale의 재표현이며 여러 외부 선택 사용 | 식 (3)의 항등식만 |
| projected primordial $A_s$의 예측 지위 | 관측 target을 본 다섯 후보와 projector 선택; action normalization 부재 | 후보별 산술과 `[경험식]` 지위만 |
| exit-code 기반 과학 PASS | DESI 결과는 `REJECT`, validation은 `CAUTION`이어도 process exit 0 | report-only 실행 성공만 |

Hubble 경로의 반례는 수치적으로도 크다. $a=10^{-6}$에서 코드의 Ricci
readout은 `11.9664`였지만 복사를 포함한 정확값은 `0.0111918`이었다.
또한 `omega_b h2=0.001`과 `0.1`이 완전히 같은 acoustic angle을 냈다.
따라서 출력된 Hubble shift는 오차막대를 조정해 살릴 수 있는 근사가 아니라
현재 계산 경로에서 제외해야 하는 값이다.

원시 스펙트럼 경로는 raw 후보가 Planck snapshot에서 약 $197.8\sigma$
벗어나는 반면, 관측값에 가장 가까운 effective projection은 약
$0.17\sigma$였다. 후자는 다섯 후보를 본 뒤 선택된 target-aware readout이므로
독립 예측 성공으로 승격하지 않는다.

## 6. 관측 자료와 likelihood 판정

이 절은 같은 모형·같은 release·같은 covariance에서만 비교한 결과를
정리한다.

저장소 scorecard의 DESI DR2+CMB $\Omega_\Lambda$ 기준값은 공식 DR2 flat
$\Lambda$CDM 결과와 일치하지 않았다. 공식값
$\Omega_m=0.3027\pm0.0036$에서 평탄성을 적용하면
$\Omega_\Lambda=0.6973\pm0.0036$이며, 저장소 CE snapshot의 잔차는
$-0.78\sigma$가 아니라 $-2.28\sigma$가 된다
([DESI DR2 Results II](https://arxiv.org/html/2503.14738)). 또한
`Planck_ACT_SPT_combined` tuple은 하나의 공개 posterior로 식별되지 않는
hybrid라 관측 기준선에서 제외했다. Planck가 직접 보고한 양은
$\omega_b=\Omega_bh^2$와 $H_0$이며, 이를 $\Omega_b$ 오차로 바꾸려면 joint
chain covariance가 필요하다
([Planck 2018 VI](https://arxiv.org/abs/1807.06209)).

내장 DESI DR2 13점과 full covariance의 값 자체는 공식 public likelihood와
일치했다. 외부 $r_d=147.09\,\mathrm{Mpc}$를 고정한 모형은
$\chi^2=37.100260857$ / 13 dof, $p=3.996\times10^{-4}$로 기각됐다.
같은 자료에서 scale 하나를 사후 fit하면 $\chi^2=12.608346862$ / 12 dof,
$p=0.3981$이지만 이는 CE의 사전 예측이 아니다
([DESI DR2 data portal](https://data.desi.lbl.gov/doc/papers/dr2/),
[pinned public BAO likelihood](https://github.com/CobayaSampler/bao_data/tree/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2)).

이 DESI 자료는 preregistration 전에 이미 열람되어 exploratory다. future
slot은 `unassigned`, 평가는 `NOT_READY`이며, 2026년 DESI Results IV도 freeze
전에 공개된 같은 DR2 자료의 확장 likelihood라 독립 holdout이 아니다
([DESI DR2 Results IV](https://arxiv.org/abs/2607.27410)). 현재 checkout에는
CE 우주론을 확인하거나 기각할 독립 confirmatory holdout이 없다.

## 7. 코드·테스트가 실제로 보장하는 것

이 절은 녹색 테스트와 물리 검증을 분리한다. 선택한 12개 테스트 모듈의
116개 test 함수에는 AST 기준 무-assert test가 없었고, 확장 우주론 묶음은
`91 passed`, core 묶음은 `58 passed`, 문서 validator는 `47/47`이었다.
그러나 같은 validator는 active `[예측]`과 CE-specific physical closure가
각각 0이라고 명시한다.

전체 저장소 회귀는 `49 failed, 2500 passed, 14 skipped, 41 errors`로
종료됐다. 주된 실패는 감사 범위 밖 ScienceDB·AGI artifact 결손이지만,
현재 checkout을 clean release로 부를 수 없다는 사실은 남는다. 우주론
경로에서는 Hubble·holographic 실행물의 직접 pytest 부재, dimensionless
registry의 실제 우주론 식 coverage 공백, covariance domain/SPD 검사 누락,
`REJECT`·`CAUTION`과 exit 0의 분리가 주요 구현 부채다.

이번 감사에서 제품 코드와 정본 문서는 바꾸지 않았다. 따라서 소스에 남아
있는 강한 상태 문자열은 이 보고서가 승인한 주장이 아니며, 정정 전에는
출판·release 근거로 사용할 수 없다.

## 8. 재개 가능한 연구 경로

이 절은 현재 결과를 억지로 살리지 않고 검증 가능한 다음 문제로 바꾼다.

1. 즉시 완결 가능한 경로는 정확한 고정점 코어로 범위를 축소하는 것이다.
   exact $D$, 가지 선택과 수치 원장을 하나로 고정하고 물리 밀도 사상은
   주장하지 않는다.
2. 우주론 관측 경로는 아직 보지 않은 future BAO release, dataset hash,
   covariance, 후보 집합과 kill rule을 먼저 동결해야 한다. holdout에서는
   연속 scale을 fit하지 않는다.
3. DM/DE는 covariant action, background stress tensor와 perturbation을
   제시하는 새 모형에서 다시 시작해야 한다.
4. primordial spectrum은 projector 대신 Mukhanov--Sasaki 방정식,
   vacuum, reheating과 amplitude normalization을 고정해야 한다.
5. $H_0$는 복사·바리온 loading과 실제 recombination history를 포함한
   Einstein--Boltzmann likelihood로 다시 계산해야 한다.
6. phase-area 경로를 계속 연구한다면 정수·부호·계수를 하나의 새 공리로
   사전 고정하고, 같은 $H_0$ scale의 재표현이 아닌 독립 교차예측을 내야
   한다.

우선순위는 정확한 코어, 독립 holdout, action-level dark sector와
primordial/$H_0$ 계산, 마지막으로 entropy ansatz다. 각 경로는 사전 고정한
kill test를 통과하기 전 `[미완성]` 또는 `[경험식]`을 유지한다.

## 9. 한계

이 감사는 commit `5414336ae2ff20197efe3bf8a92ec5183ad079aa` 위의 현재 dirty
worktree를 대상으로 했다. 우주론 주장에 도달하지 않는 AGI, brain,
fusion, Guard와 삭제 상태 RBE 구현은 수학 전수 범위에서 제외했지만 전체
pytest 상태는 별도로 기록했다. 관측 비교는 2026-08-15에 접근 가능한 공식
release와 공개 likelihood를 사용했으며, 공개 joint chain이 없는 비율과
covariance는 `UNVERIFIED`로 남겼다. 독립 future holdout이 없으므로 이
보고서는 관측 확인 보고서가 아니라 수학·출처·구현의 폐쇄 감사다.

## 10. 재현성과 산출물

핵심 계산은 다음 명령으로 재현한다.

```powershell
python _workspace\ce\cosmology-theory-repository-audit-20260815\artifacts\verify_cosmology_math.py

python -m pytest tests\test_bootstrap_solver.py tests\test_core_model_selection.py tests\test_cosmology_ratio_audit.py tests\test_ce_residual_forward_model.py tests\test_recombination_drag_adapter.py tests\test_primordial_spectrum_readout_gate.py tests\test_dimensionless.py tests\test_holdout_preregistration.py -q -p no:cacheprovider

python examples\physics\ce_residual_forward_model.py --bao-dataset desi-dr2-all

python experiments\preregistration\validate_holdout_manifest.py experiments\preregistration\cosmology_future_holdout_v2.json
```

세부 근거는 `10-sources.md`의 관측 provenance, `11-math.md`의 독립 유도와
반례, `12-routes.md`의 대안 경로, `20-audit.md`의 32개 최소 주장 원장,
`31-validation.md`의 실행 결과에 분리해 기록했다. 코드 의존 경로와 전체
테스트 매핑은 `artifacts/code-inventory.md`, 명령 원문은
`artifacts/validation-command-ledger.md`에 있다.

## 11. 참고문헌

- Planck Collaboration, [Planck 2018 results VI: Cosmological parameters](https://arxiv.org/abs/1807.06209), A&A 641 A6, accessed 2026-08-15.
- DESI Collaboration, [DESI DR2 Results II](https://arxiv.org/html/2503.14738), PRD 112 083515, accessed 2026-08-15.
- DESI Collaboration, [DR2 publications and cosmology chains](https://data.desi.lbl.gov/doc/papers/dr2/), accessed 2026-08-15.
- CobayaSampler, [version-pinned DESI DR2 Gaussian BAO likelihood](https://github.com/CobayaSampler/bao_data/tree/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2), accessed 2026-08-15.
- ACT Collaboration, [ACT DR6 Power Spectra, Likelihoods and LambdaCDM Parameters](https://arxiv.org/abs/2503.14452), accessed 2026-08-15.
- SPT-3G Collaboration, [SPT-3G D1 spectra and cosmology, corrected v2](https://arxiv.org/abs/2506.20707v2), accessed 2026-08-15.
- BICEP/Keck Collaboration, [BK18 primordial gravitational-wave constraint](https://arxiv.org/abs/2110.00483), PRL 127 151301, accessed 2026-08-15.

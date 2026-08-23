<!-- PROVENANCE: 병행 세션(Codex, 2026-08-23)이 stage 파일로 직접 집필한 판본. 단일 작성자 규약 위반으로 artifacts에 이동 보존 (20-audit §6). 정본은 run 루트의 동명 파일. -->
# BA-V3-1 최종 보고 — 실제 뇌 자료 적합 이전의 계약 차단과 안정성 반례

Status: COMPLETE

## 초록

이 run은 생물학적 원시연산으로 구성한 시냅스 동역학이 여러 뇌 관측 비율과 창발 통계를 함께 재현할 수 있다는 BA-V3-1 계약을 실제 자료 적합 전에 감사했다. 1차 문헌의 표본·분모·시간창과 계약의 관측량을 대조하고, 제안된 wake/sleep map의 큰 가중치 극한과 질량수지를 독립적으로 검산했다. 그 결과 R1′, R3b′, R4′, R5′의 수치 게이트는 현재 형태로 출처가 잠기지 않았고, 생물 기준식·CE 추가항·측정식도 분리되어 있지 않았다. 또한 양의 수면 감쇠율 $\lambda(w)>0$만으로는 주기 정상상태가 보장되지 않으며, 계약이 허용하는 지속적 각성 증가 아래 가중치가 발산하는 해석적 반례가 성립한다. 따라서 현 계약은 `BLOCKED`이며 구현·피팅·L1/L2 주장은 열지 않았다. 살아남는 산출은 측정 정의가 고정된 좁은 관측 비교, 발산 반례, 그리고 실제 공개 자료에서 측정모형부터 고정해야 한다는 후속 계약 조건이다.

## 1. 문제와 판정 범위

BA-V3-1은 spike와 delay, Dale 제약, STDP, three-factor plasticity, lateral inhibition, homeostasis, 두 시간척도 구조, 수면 스케일링을 조합해 여섯 관측 비율과 두 창발 통계를 동시에 설명하려 했다. 그러나 여러 생물 기전을 한 상태식에 넣는 것과 실제 뇌 자료가 그 식을 지지하는 것은 다른 주장이다. 실제 자료 비교가 성립하려면 적어도 생물학적 기준 동역학 $F_{\rm bio}$, CE가 새로 제안하는 항 $\Delta F_{\rm CE}$, 관측 장치와 전처리를 포함한 측정식 $\mathcal H_d$가 분리되어야 한다.

본 감사의 판정은 다음 세 층을 구별한다. 표준 생물학 문헌에서 직접 정의된 비교는 [경험식/관측 비교], 제안된 동역학은 [가설], 수학적으로 닫힌 반례는 [정리]로 취급한다. 시뮬레이터가 아직 실행되지 않았으므로 어떤 생물학적 메커니즘도 [산출]이나 [확인]으로 승격하지 않는다.

## 2. 정의와 표기

후속 실제 자료 계약의 최소 분해는

$$
d x_t
=F_{\rm bio}(x_t,u_t;\theta)\,dt
+\Delta F_{\rm CE}(x_t,h_t;\phi)\,dt
+G(x_t)\,dW_t,
$$

$$
y_{d,k}=\mathcal H_d(x_{t_k};\psi_d)+\epsilon_{d,k}
$$

이다. 여기서 $x_t$는 잠재 생물 상태, $u_t$는 자극·행동·수면 조건, $h_t$는 CE 가설이 요구하는 추가 상태, $y_{d,k}$는 데이터셋 $d$의 실제 측정값이다. $\mathcal H_d$에는 표본 선택, rank/threshold, segmentation, censoring, animal–dendrite–synapse nesting과 오차모형이 들어가야 한다. 현 계약은 이 분해를 제공하지 않았으므로 기준식과 추가항의 잔차 개선을 비교할 수 없다.

계약의 수면 항은

$$
\lambda(w)=\frac{\lambda_0}{1+(w/\kappa)^2},\qquad
w^{\rm sleep}=w^{\rm wake}[1-\lambda(w^{\rm wake})]
$$

이다. $w$와 $\kappa$는 같은 강도 단위이고 $\lambda_0$는 한 수면 update당 무차원이어야 한다. 반면 STDP의 $\tau_\pm$, 제거 시간 $\tau_{\rm el}$, 발달 시간 $T_m$, turnover rate $\rho_\infty$는 물리 시간과 연결되어야 한다. 현 계약에는 tick–ms–수면 주기–일–월 변환표가 없으므로 이 시간척도들을 하나의 목적함수에서 정량 결합할 수 없다.

## 3. 조건부 공리와 정리

### 공리 A1 — 허용된 단일 시냅스 부분모형

하루 각성 동안 한 시냅스가 일정량 $a>0$만큼 증가하고, 수면 동안 계약의 $\lambda(w)$로 감소한다고 하자. 이는 현 계약이 금지하지 않는 지속적 양의 각성 drive의 최소 부분모형이다.

### 정리 T1 — $\lambda(w)>0$는 주기 정상상태의 충분조건이 아니다

하루 map은

$$
T(w)=(w+a)\left[1-\frac{\lambda_0}{1+((w+a)/\kappa)^2}\right]
$$

이다. 큰 $w$에서

$$
T(w)-w
=a-\frac{\lambda_0\kappa^2}{w+a}+O(w^{-3}).
$$

따라서 충분히 큰 $w$에 대해 $T(w)-w>a/2>0$이다. 그 영역에서 궤적은 매 주기 양의 증가량을 가지므로 tightness, invariant distribution, cyclostationary state가 일반적으로 보장되지 않는다. 즉 $\lambda(w)>0$라는 부호 조건은 정상성의 충분조건이 아니다. □

수치 witness는 $a=0.1$, $\lambda_0=0.2$, $\kappa=1$, $w_0=100$에서 1,000주기 뒤 $w=198.60810635598017$을 얻었고 최소 주기 증가량은 $0.09800219738329474>a/2$였다. 이 검사는 정리의 증명이 아니라 구현된 식이 해석식과 일치하는지 확인한 재현 보조물이다.

### 산출 D1 — 야간 질량수지

총 강도 $W=\sum_i w_i$와 야간 손실 $L=\sum_i\lambda(w_i)w_i$에 대해 총 하강분율은 $R_{3a}=L/W$이다. 상위 집합의 강도 점유율을 $s$, 그 집합의 하강분율을 $\ell_{\rm top}$이라 하면 나머지 집합은

$$
\ell_{\rm bottom}=\frac{R_{3a}-s\ell_{\rm top}}{1-s}
$$

를 만족해야 한다. 따라서 전체 약 18% 감소와 상위 집합의 작은 변화는 논리적으로 모순은 아니지만, $s$와 집단별 불확도가 없으면 하위 집합에 필요한 부담을 계산할 수 없다. scratch 검사는 아홉 조합에서 이 항등식을 오차 없이 재현했다.

## 4. 실제 관측과 계약의 불일치

Grutzendler 등의 성체 피질 spine 약 96% 월간 안정성은 확인되지만, 그 여집합 4%를 곧바로 removal-only 또는 전체 turnover로 놓을 수는 없다. 분자는 기준시점 spine 중 소실된 수, 분모는 기준시점 spine 수로 고정하고 gain·loss·net change를 따로 보고해야 한다. 따라서 R1′은 현재 수치 게이트로 사용할 수 없다.

Holtmaat 등의 persistent spine은 수명 8일 이상이라는 N-series 정의와 연령별 분율을 제공한다. 이 비교는 살아남지만 cohort 구성, 관찰 시작점, censoring, 표본 가중을 원 방법절과 함께 잠가야 한다. 계약의 다른 turnover 지표와 같은 분모로 간주해서는 안 된다.

de Vivo 등의 SBEM 연구는 수면군의 axon–spine interface 면적이 자발 각성군보다 18.9%, 강제 각성군보다 17.5% 낮았다고 보고한다. 이는 실제 형태학적 관측 비교이지 직접적인 시냅스 강도 측정은 아니다. “상위 20%”는 ASI 순위로 정한 분석 하위집단이며, 본문 결과는 그 집단에 대해 각각 +0.7%, +2.0%인 점추정치를 제시한다. 따라서 계약의 정확한 0–5% hard gate나 “상위 20% 불변” 법칙으로 승격할 수 없다. 다만 공식 공개 `synapse_data.csv`가 있어 측정모형을 고정하는 후속 분석의 가장 직접적인 시작점이다.

Yang 등의 결과는 운동학습 뒤 8시간 동안 특정 고빈도 가지에서 수면군 9.3±0.7%, 수면박탈군 4.9±0.7%의 spine 형성을 비교한다. 이를 일반적인 “수면이 형성을 증가시킨다”는 단일 scalar로 축약하면 가지 종류, 훈련, 박탈 조건, 시간창을 잃는다. R6′은 이 좁은 contrast로 다시 정의되어야 한다.

R4′의 “100일당 5% 미만 안정성”과 R5′의 1.5 비율은 대조한 1차 출처에서 같은 통계량으로 확인되지 않았다. 두 항목은 현 계약에서 삭제한다. 결과적으로 여섯 게이트의 동시 적합이라는 부모 주장은 관측 대상 자체가 고정되지 않았으므로 활성 주장으로 남을 수 없다.

## 5. 식별성과 모델 선택 한계

“자유 파라미터 8개보다 조건 10개가 많다”는 셈은 식별성 증명이 아니다. 일부 조건은 부등식이나 구조적으로 자동 통과하는 지표이고, $w_0$, $\tau_e$, homeostatic gain과 적용 위치, scaling locus 같은 설계 상수가 출력에 큰 영향을 준다. 또한 $\eta g(t)$는 곱으로 들어가 절대 학습척도가 분리되지 않을 수 있고, 좁은 $w$ 범위에서는 $\lambda_0$와 $\kappa$가 강하게 상관될 수 있다.

후속 계약은 사전에 고정한 관측벡터 $q(\theta)$에 대해

$$
J_{ab}=\frac{\partial q_a}{\partial\log\theta_b}
$$

의 rank·singular values·condition number와 profile likelihood를 계산해야 한다. 기준식, CE 추가항, matched control은 animal-level development/confirmation 분할에서 같은 likelihood 또는 predictive score로 비교해야 한다. 같은 holdout을 식 제안과 확인에 함께 쓰면 확인 증거가 아니다.

## 6. 주장 처분과 후속 경로

현 BA-V3-1 계약의 `BIO_EVIDENCE_L1(+L2)` 성공 주장, R4′·R5′ 게이트, $\lambda(w)>0$가 정상성을 보장한다는 주장, E1/E2가 강제되지 않은 창발 통계라는 주장은 활성 집합에서 제거한다. R2′, R3a′, R6′은 각각 메타데이터가 완전한 N-series 관측, SBEM/ASI 측정모형 비교, 훈련·박탈·8시간을 보존한 좁은 contrast로만 후속 계약에 들어갈 수 있다. R1′과 R3b′은 분자·분모·불확도·전처리 규칙이 1차 자료 수준에서 고정되기 전에는 점수화하지 않는다.

동역학 수리 후보 RT-S(각성 증가 포화), RT-G(큰 $w$에서도 사라지지 않는 수면 손실), RT-H(명시적 전역 항상성)는 모두 [가설]로 남는다. 현 자료는 어느 하나도 선택하지 않는다. 먼저 RT-0, 즉 실제 자료의 측정식과 분할을 잠근 뒤, 개발 자료에서 한 번에 구조식 하나만 추가하고 독립 confirmation에서 반증해야 한다.

재개 조건은 다음과 같다. 공식 데이터 파일의 checksum과 provenance를 기록하고, 열별 의미·결측·중첩 단위를 고정하며, $F_{\rm bio}$와 $\Delta F_{\rm CE}$를 분리하고, animal-level split·잔차 규칙·falsifier·matched controls·model-selection rule·revision trigger를 데이터 접촉 전에 계약한다. 이 조건을 만족하는 후속 run의 주장 상한은 “생물학적으로 동기화된 후보 동역학 족의 source-locked 관측 보정 및 측정모형 비교”이다.

## 7. 재현성

run 경로는 `_workspace/ce/brainruntime-bio-constrained-l1-20260822`이다. 출처 판정은 `10-sources.md`, 수학 감사와 상세 단위 검사는 `11-math.md`, 구조적으로 다른 수리 경로는 `12-routes.md`, 독립 지위 감사는 `20-audit.md`에 있다. 실행 보조물은 `artifacts/math-verify/periodic_drift_counterexample.py`와 `mass_balance_witness.py`이며, 저장된 식 이외의 모델 파라미터를 적합하지 않는다. 감사 게이트가 막혔으므로 `30-implementation.md`와 `31-validation.md`는 의도적으로 생략되었고 실제 데이터에 대한 피팅도 수행하지 않았다. `BLOCKED` 음성 결과를 최종 보고서로 닫는 코어 회귀검사는 Rust 단위 테스트 1건으로 통과했지만, 갱신된 로컬 실행 파일은 Windows Application Control에 차단되어 설치된 CLI의 `check final` 재실행은 보류했다. 정책은 우회하지 않았으며 CLI 재개에는 해당 실행 정책의 정상 복구가 필요하다.

## 8. 참조

- Grutzendler et al., *Nature* 420, 812–816 (2002), DOI `10.1038/nature01276`, 접근 2026-08-23.
- Holtmaat et al., *Neuron* 45, 279–291 (2005), DOI `10.1016/j.neuron.2005.01.003`, 접근 2026-08-23.
- de Vivo et al., *Science* 355, 507–510 (2017), DOI `10.1126/science.aah5982`; UW–Madison 공개 데이터셋 페이지, 접근 2026-08-23.
- Yang et al., *Science* 344, 1173–1178 (2014), DOI `10.1126/science.1249098`, 접근 2026-08-23.


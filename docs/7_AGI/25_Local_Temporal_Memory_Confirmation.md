# Local temporal-memory confirmation

이 문서는 local temporal-memory가 지정 horizon에서 baseline보다 구별되는지 확인한 좁은 confirmation 기록이다. 독자는 time-series prediction·holdout·counterfactual baseline의 기본을 아는 독자를 전제로 하며, 기억 비유는 구현 state와 실험 metric을 넘어서는 기제 주장이 아니다.

증명 명제와 판정식을 먼저 읽고 누수 방지, AML310 탐색, untouched AML32 확인, 독립 계산과 AGI 해석 경계를 차례로 확인한다. dataset provenance·split·seed·horizon·threshold가 정의역이며, 통과가 일반 memory·지능·생물학 결과로 승격되지 않는다.


> **지위 변경 (2026-08-12) — 생물학적 해석 강등.**
> 아래 원문이 기록한 사전등록 계산 게이트 PASS(AML32 $h=1$ 7/7, $h=6$ 7/7)는 계산적 사실로 유지된다. 그러나 이 결과를 신경 세포 내부의 시간 기억(neural local temporal memory)으로 읽는 생물학적 해석은 **[삭제된 해석]**으로 강등한다. 근거는 활동 비의존 형광 대조군에서 성립한 완전 반례다.
>
> 활동 비의존 GFP-only 계통 AML18의 11개 recording에 본 문서와 동일한 confirmatory 게이트($h=1$, $h=6$, 동일 모델 family·null 절차, min_pass $=\lceil 11\cdot 5/7\rceil=8$)를 점수 산출 전에 잠근 사전등록(`artifacts/agi/local_memory_aml18_preregistration.json`, 기대 방향 명시: GFP FAIL이면 본 문서 해석 생존, GFP PASS면 강등)에 따라 적용한 결과, $h=1$ 11/11 PASS(median $\Delta$ 0.024–0.079), $h=6$ 11/11 PASS($\Delta$ 0.211–0.383)로, AML32 GCaMP($h=1$ $\Delta$ 0.013–0.043, $h=6$ 0.115–0.213)보다 신호가 더 컸다. 활동 비의존 채널에서 lag 예측정보가 더 크게 나타나므로, 관측된 예측정보는 indicator kinetics와 전처리(Ratio2·I_smooth 저역통과)의 시간적 자기상관으로 설명 가능하며 세포 내부 기억기작을 지지하지 않는다.
>
> 산출물: `artifacts/agi/local_memory_aml18_h1_confirmatory.json`, `artifacts/agi/local_memory_aml18_h6_confirmatory.json`. 데이터: OSF dpr3h `AML18_moving.tar.gz`(sha256 `588d7666f4e8afebad1ab9b8483244a6de0303251d862425522c2b8dd78bbd82`), `artifacts/agi/cloudcell_data_manifest.json`에 등재. 계산 게이트 자체(누출 방지, null 절차, AML32 7/7 재현)는 유효하며, "전처리된 칼슘/형광 시계열에 lag 예측정보가 있다"는 계산적 사실은 산출로 유지된다.
>
> 아래 본문은 당시(2026-07~08) 시점 기술로 보존한다(**stale**). §7 표의 indicator-kinetics 행만 이 반례를 반영해 갱신했다.
>
> **최종 확정 (2026-08-12, 사전등록 본 실행).** 위 반례 이후 남아 있던 가능성, 즉 관측된 "GCaMP $\Delta$ < GFP $\Delta$"가 headroom 차이에 가려진 생물 신호일 가능성은 headroom 항등식 $\Delta=(1-R^2_C)\Delta'$로 분리했다. 이 항등식은 원래 관측된 strain 간 $\Delta$ 차이가 분모 $(1-R^2_C)$ 효과만으로 설명됨을 보이므로, target-level $\delta'$의 median-of-ratios를 교정 통계량으로 하는 killing test를 실행 전에 사전등록(`artifacts/agi/local_memory_gfp_matched_preregistration.json`; AML18 recording ID 목록의 전사 오류 1건은 실행 전에 교정했고 데이터 선택은 불변)하고 본 실행했다. 주검정 $h=6$에서 AML32 중앙값 0.3564($n=7$) 대 AML18 0.3522($n=11$), 차 $+0.0042$, exact one-sided Mann–Whitney $p=0.5351$로 사전등록 KILL 조건($p>0.05$이고 차 $\le 0.02$)을 충족했고, 부검정 $h=1$도 0.9686 대 0.9681, $p=0.3295$로 같은 방향이다. 따라서 생물학적 해석은 양 horizon 모두 KILL로 확정되며, 존속하는 주장은 "전처리+indicator 합성 과정의 비-Markov성(lag-2 조건부 예측정보)"뿐이다. $h=1$에서 양 strain 모두 $\delta'\approx 0.97$인 사실은 결정론적 스무딩 필터의 서명과 부합한다. 검정력 한계로, 이 설계는 잔차분산 약 2%p 미만의 효과를 배제하지 못한다(사전 명시). 산출물: `artifacts/agi/local_memory_gfp_matched_result.json`. 설계 분해(게이트 PASS ⟺ 관측 과정의 AR(1)-Markov성 위반)는 `_workspace/ce/agi-clarus-field-20260812/artifacts/bio_gate_redesign_note.md`에 있다.

> 최종 상태: `preregistered computational gate PASS` — 계산적 사실로 유지; 생물학적 해석은 위 블록에 따라 강등
>
> AML310 exploratory: `h=1 4/4`, `h=6 4/4`
>
> untouched AML32 confirmatory: `h=1 7/7`, `h=6 7/7`

## 1. 실제로 증명한 명제

이 명제는 local state·time window·prediction output과 지정 baseline의 차이를 제한된 fixture에서 비교한다. 증명의 정의역·초기 조건·metric 분모를 벗어나면 결론은 적용되지 않고, memory라는 이름이 인과 기제를 뜻하지 않는다.

이 결과가 지지하는 명제는 다음처럼 좁다.

> 움직이는 C. elegans의 이 calcium-activity 데이터에서, 한 뉴런의
> $t-1,t-2$ 측정값은 그 뉴런의 현재 측정값 $x_i(t)$만으로 만든
> 비선형 기준선을 조건으로 한 뒤에도 $t+h$ 측정값에 대한 held-out
> 예측정보를 가진다. 이 결과는 $h=1,6$과 별도의 AML32 일곱 동물에서
> 재현된다.

고정한 예측식은

$$
\widehat x_i(t+h)
=\beta_0+g\!\left(x_i(t)\right)
 \beta_1x_i(t-1)+\beta_2x_i(t-2),
$$

$$
g(x)=[x,x^2,x^3,\tanh x]
$$

이다. current-only 기준선은 $\beta_1=\beta_2=0$인 같은 ridge
family다. 따라서 단순 선형 current 기준선이 약해서 생기는 이득만을
세지 않았다.

## 2. 증명 판정식

판정식은 sample unit·horizon·error metric·threshold·불확실성을 고정해 pass/fail을 정한다. 추정량은 dataset split·seed에 조건부이며, 식의 통과는 scientific truth가 아니라 등록된 claim의 기계·통계 판정이다.

기록 $r$, 뉴런 $i$, horizon $h$에 대해

$$
d_{r,i,h}
=R^2_{r,i,h}(\mathrm{local})
-R^2_{r,i,h}(\mathrm{current\ nonlinear})
$$

로 두고, 동물별 효과를

$$
\Delta_{r,h}=\operatorname{median}_i d_{r,i,h}
$$

로 정의했다. $t-1,t-2$ 열을 train/validation/test 각 블록 안에서
함께 원형 이동하고 **매번 모델을 다시 학습**한 19개 null을
$\Delta^{(b)}_{r,h}$라 하면

$$
p_{r,h}
=\frac{1+\sum_{b=1}^{19}
\mathbf 1[\Delta^{(b)}_{r,h}\ge\Delta_{r,h}]}{20}.
$$

기록 하나의 사전등록 통과 술어는

$$
G_{r,h}=
\mathbf 1\!\left[
\begin{array}{l}
n_{\rm target}\ge20,\\
\Delta_{r,h}>0.01,\\
\Pr_i(d_{r,i,h}>0)\ge0.8,\\
p_{r,h}\le0.05
\end{array}
\right].
$$

확인 패널의 전체 술어는

$$
G_{\rm panel}
=
\mathbf 1\!\left[\sum_{r=1}^{7}G_{r,1}\ge5\right]
\land
\mathbf 1\!\left[\sum_{r=1}^{7}G_{r,6}\ge5\right].
$$

규칙은 AML32 activity를 평가하기 전에
`local_memory_aml32_preregistration.json`에 고정했다.

## 3. 누수 및 약한 대조군 방지

이 절은 future information, entity overlap, weak baseline이 만든 거짓 개선을 차단한다. provenance·split·baseline·ablation이 깨지면 결과는 expected failure 또는 rollback이며, 사후 tuning으로 복구하지 않는다.

- 시간순 60/20/20 분할과 5-sample embargo를 썼다.
- ridge는 validation에서만 고르고 test는 변환·학습·선택에 쓰지 않았다.
- eligible target은 train 구간의 결측률과 분산으로만 정했다.
- acquisition gap을 가로지르는 lag/target window는 제외했다.
- null은 과거 열의 자기상관과 두 lag 사이 관계를 보존하고, 각 null
  feature에서 모델과 ridge를 다시 맞췄다.
- test block을 변조해도 fitted-model SHA-256이 변하지 않는 회귀시험을
  통과했다.
- 합성 AR(2)는 통과하고, 진짜 AR(1) current-state process는 실패했다.
- 복제 단위는 뉴런이 아니라 독립 기록/동물이다.

## 4. 탐색 패널: AML310

AML310은 model·dataset·seed를 탐색하는 개발 panel이며 selection bias가 있을 수 있다. 이 패널의 수치는 hypothesis 생성에만 쓰고, confirmation 승격은 untouched split과 독립 계산을 요구한다.

| horizon | 기록별 median $\Delta R^2$ | 기록 통과 |
|---|---:|---:|
| $h=1$ | 0.0293, 0.0226, 0.0604, 0.0379 | 4/4 |
| $h=6$ | 0.2298, 0.1952, 0.3085, 0.1879 | 4/4 |

모든 기록의 positive-target fraction은 $0.901$ 이상이고 null rank
$p=0.05$였다. 이 결과를 본 뒤에도 AML32 기준은 바꾸지 않았다.

## 5. untouched 확인 패널: AML32

AML32는 탐색·tuning에서 분리한 confirmation panel로, 동일 판정식과 baseline을 적용한다. panel provenance·seed·horizon이 바뀌면 새 확인으로 등록해야 하며, 통과하지 않으면 탐색 결과를 지지하지 않는다.

### $h=1$

$h=1$은 지정한 한 time step horizon의 prediction fixture다. 다른 horizon과 분모·난이도가 다르므로 metric을 합치지 않고 seed·baseline·threshold에서 따로 판정한다.

| recording | targets | current $R^2$ | local $R^2$ | $\Delta R^2$ | positive | null $p$ | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20170610_105634 | 107 | 0.9759 | 0.9994 | 0.0230 | 0.991 | 0.05 | PASS |
| 20170613_134800 | 118 | 0.9771 | 0.9994 | 0.0223 | 1.000 | 0.05 | PASS |
| 20170424_105620 | 110 | 0.9677 | 0.9990 | 0.0309 | 1.000 | 0.05 | PASS |
| 20180709_100433 | 135 | 0.9865 | 0.9996 | 0.0131 | 1.000 | 0.05 | PASS |
| 20200309_151024 | 121 | 0.9553 | 0.9986 | 0.0427 | 1.000 | 0.05 | PASS |
| 20200309_153839 | 131 | 0.9539 | 0.9981 | 0.0425 | 0.992 | 0.05 | PASS |
| 20200309_162140 | 134 | 0.9613 | 0.9988 | 0.0376 | 1.000 | 0.05 | PASS |

사전등록 요구치는 5/7이고 관측값은 **7/7**이다.

### $h=6$

$h=6$은 더 긴 time horizon의 독립 fixture로, error accumulation과 OOD drift가 다를 수 있다. $h=1$ 통과는 이 조건의 성공을 보장하지 않으며, failure는 memory 가설의 적용 범위를 좁힌다.

| recording | targets | current $R^2$ | local $R^2$ | $\Delta R^2$ | positive | null $p$ | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20170610_105634 | 107 | 0.4875 | 0.6844 | 0.1949 | 0.991 | 0.05 | PASS |
| 20170613_134800 | 118 | 0.4778 | 0.6882 | 0.2098 | 1.000 | 0.05 | PASS |
| 20170424_105620 | 110 | 0.3673 | 0.5881 | 0.2135 | 0.991 | 0.05 | PASS |
| 20180709_100433 | 135 | 0.6862 | 0.8007 | 0.1153 | 1.000 | 0.05 | PASS |
| 20200309_151024 | 121 | 0.2405 | 0.4951 | 0.2388 | 0.950 | 0.05 | PASS |
| 20200309_153839 | 131 | 0.1745 | 0.4220 | 0.2561 | 0.969 | 0.05 | PASS |
| 20200309_162140 | 134 | 0.2735 | 0.5383 | 0.2430 | 0.993 | 0.05 | PASS |

사전등록 요구치는 5/7이고 관측값은 다시 **7/7**이다.

## 6. 독립 계산 검증

독립 계산은 builder와 분리된 implementation·artifact·seed에서 판정식을 재계산하는 재현성 gate다. 수치 parity는 등록된 fixture의 기계 evidence이며, dataset bias·과학적 기제·일반 AGI 성능을 해결하지 않는다.

verifier는 결과 파일의 `gate_passed`를 신뢰하지 않고 다음을 다시
계산한다.

1. 사전등록 implementation SHA-256 일치
2. AML32 archive SHA-256 일치
3. 정확히 사전등록한 일곱 recording인지 확인
4. 각 기록의 네 조건을 원시 수치에서 재계산
5. horizon별 pass count와 5/7 조건 재계산
6. $h=1,6$ 동시 통과 재계산

결과는

```text
proof_passed = true
errors = []
h=1: 7/7, required 5
h=6: 7/7, required 5
```

이다. threshold 변조와 implementation hash 변조를 거부하는 시험도
통과했다.

## 7. 무엇이 증명됐고 무엇은 아직 아닌가

이 절은 조건부 형식 결론, 확인 panel의 경험 evidence, 남은 구조 해석을 분리한다. 반례·OOD·새 split이 나타나면 승격을 취소하거나 범위를 좁히며, 그 공백을 그럴듯한 서사로 메우지 않는다.

| 명제 | 판정 |
|---|---|
| 고정된 코드와 artifact가 사전등록 술어를 만족 | **Exact computational PASS** |
| AML32 measured trace에 current를 넘는 aligned local-history 예측정보 존재 | **Confirmatory support, 7/7 at both horizons** |
| 이 정보가 calcium indicator/전처리 평활화가 아닌 세포 내부 기억기작 | **시험됨 — 반례 성립** (2026-08-12, AML18 GFP-only 11 recording에서 동일 게이트가 $h=1$·$h=6$ 모두 11/11 PASS, $\Delta$가 AML32보다 큼; 문서 머리 지위 변경 블록 참조) |
| 뉴런 간 population cloud가 local history 위에 추가 정보 제공 | 반증, 0/4 |
| anonymous activity에서 directed effective graph가 local보다 우수 | 반증, 0/4 |
| diffusion이 linear/persistence보다 우수 | 반증, 최대 1/4 |
| 뉴런 자체가 category-theoretic monad/CloudCell | 미증명이며 현재 자료로 식별 불가 |
| 이 결과가 AGI architecture를 직접 입증 | 미증명 |

특히 $p=0.05$는 19개 고정 phase-null이 주는 최소 해상도다. 이는
“모든 null보다 관측 정렬이 컸다”는 뜻이지, 생물학적 모집단에서 정확한
확률이 0.05라는 뜻은 아니다. 또한 calcium indicator kinetics나 기존
signal processing도 local history를 만들 수 있다. 따라서 정당한 결론은

$$
\boxed{
\text{measured neuron activity is predictively stateful over time}
}
$$

까지다. 현재 결과로

$$
\text{neuron}=\text{coded monadic CloudCell}
$$

을 쓰는 것은 증거 범위를 넘는다.

## 8. AGI 쪽에서 남는 의미

AGI 의미는 local-memory 결과가 설계 후보에 주는 제한된 input을 설명한다. 현재 증거는 feature·task·dataset 범위에 한정되며, 통합 agent 효능·의식·생물학 대응은 별도 baseline·ablation·OOD gate가 필요하다.

AGI 설계에 가져갈 수 있는 것은 존재론적 동일시가 아니라 설계 제약이다.

$$
\text{unit state}_{t+1}
=F(\text{unit state}_{t:t-2},\ \text{input}_t)
$$

처럼 각 unit에 짧은 local state를 두는 가정은 실제 데이터와 합치한다.
반대로 이번 자료는 dense population cloud, learned directed graph,
nonlinear diffusion을 local state 위에 반드시 얹어야 한다는 근거를
주지 않는다. 따라서 현재 우선순위는 **local recurrent state를 기본으로
두고, cross-unit 구조는 별도 데이터에서 증명될 때만 추가하는 것**이다.

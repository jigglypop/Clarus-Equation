---
question: Q-0008
attempt: 4
ladder_step: 6
claim: "사전등록 실행 기록 — 카드 F-02 사다리 6단 수치시험 K1·K2·K5(+K4 일관성)를 `verify/Q-0008/F-02/check_modes.py --mode her|mix|iid|defect`(seed 20260902, δ=0.005, 상수·창 무수정)로 실행한 결과 her_slope 0.5576±0.0187, her_ratio_128 34.24±1.68, mix_X_32 0.819±0.088, iid_slope −0.480±0.009가 모두 사전등록 창 안이고(가장 가까운 창 경계까지 각각 3.9σ·2.9σ·1.9σ·10.9σ), K4 defect 0.14307·−0.8982도 창 안(일관)이다. kill 발동 없음."
assumptions:
  - "실행 규약은 카드 scope '스크립트 상수' 항 그대로: δ=0.005, MIN_DET=0.05, seed 20260902(her 모드의 iid 비교열은 seed+1=20260903), n∈{8,16,32,64,128}, 256 trials/size, mix n=32·1024 trials, defect 격자 {4,8,16,32,64}·섭동 0.35·det>0.2 첫 표본. 스크립트 상수 PREREGISTERED·WINDOWS 8쌍은 카드 predicts·kill·consistency_checks와 일치(assemble_result.py가 기계 대조, constants_match_card=true)"
  - "표준오차는 trial bootstrap(B=2000, bootstrap seed 20260902)이다. 같은 seed·같은 추출 순서를 재현해 trial별 잔차를 얻었고, 재현 RMS와 스크립트 출력의 차는 0.0(<1e−12)이다. 사전등록 통계 자체는 bootstrap과 무관하며 판정은 스크립트가 쓴 값으로 한다"
  - "defect 모드는 사전등록 seed의 Δ 표본 하나에 대한 결정론 값이므로 표준오차가 없다. 카드 rev.2에 따라 kill이 아닌 일관성 검사다"
  - "공유 파일 verify/Q-0008/F-02/result.json에는 같은 시각에 병행 실행된 attempt-05의 qspine 블록(K3, 7단)이 병합돼 있다. 이 attempt는 그 블록을 생성·판정하지 않으며, 네 모드 블록은 attempt-04 로그와 값이 일치함을 확인했다"
  - "결과를 본 뒤 카드·스크립트·창·tol·seed를 바꾸지 않았다(sha256: check_modes.py fdb0e4e5…, F-02.formula.md 17ac890a…, result.json integrity 항)"
symbols:
  n: positive integer
  x: real
verify:
  # ---- K1 her_slope: 관측 0.5576106551570001, 창 [0.43, 0.63], 사전등록 0.5302±0.10
  - type: inequality
    lhs: "0.5576106551570001"
    rhs: "0.43"
    relation: ">="
  - type: inequality
    lhs: "0.5576106551570001"
    rhs: "0.63"
    relation: "<="
  - type: numeric
    expr: "0.5576106551570001 - 0.5302"
    tol: 0.10
  # ---- K1 her_ratio_128: 관측 34.24199061956648, 창 [26.0, 39.1], 사전등록 32.554±6.5
  - type: inequality
    lhs: "34.24199061956648"
    rhs: "26.0"
    relation: ">="
  - type: inequality
    lhs: "34.24199061956648"
    rhs: "39.1"
    relation: "<="
  - type: numeric
    expr: "34.24199061956648 - 32.554"
    tol: 6.5
  # ---- K2 mix_X_32: 관측 0.8193510400305117, 창 [0.49, 0.99], 사전등록 0.7406±0.25
  - type: inequality
    lhs: "0.8193510400305117"
    rhs: "0.49"
    relation: ">="
  - type: inequality
    lhs: "0.8193510400305117"
    rhs: "0.99"
    relation: "<="
  - type: numeric
    expr: "0.8193510400305117 - 0.7406"
    tol: 0.25
  # ---- K5 iid_slope: 관측 −0.4799658222826968, 창 [−0.58, −0.38], 사전등록 −0.4783±0.10
  - type: inequality
    lhs: "-0.4799658222826968"
    rhs: "-0.58"
    relation: ">="
  - type: inequality
    lhs: "-0.4799658222826968"
    rhs: "-0.38"
    relation: "<="
  - type: numeric
    expr: "-0.4799658222826968 + 0.4783"
    tol: 0.10
  # ---- K4 일관성(kill 아님) defect_ratio_64_over_8: 관측 0.14306936148581373, 창 [0.124, 0.158], 사전등록 0.140625±0.017
  - type: inequality
    lhs: "0.14306936148581373"
    rhs: "0.124"
    relation: ">="
  - type: inequality
    lhs: "0.14306936148581373"
    rhs: "0.158"
    relation: "<="
  - type: numeric
    expr: "0.14306936148581373 - 0.140625"
    tol: 0.017
  # ---- K4 일관성 defect_slope: 관측 −0.8981828134067237, 창 [−0.96, −0.86], 사전등록 −0.9069±0.05
  - type: inequality
    lhs: "-0.8981828134067237"
    rhs: "-0.96"
    relation: ">="
  - type: inequality
    lhs: "-0.8981828134067237"
    rhs: "-0.86"
    relation: "<="
  - type: numeric
    expr: "-0.8981828134067237 + 0.9069"
    tol: 0.05
  # ---- 산술 정합: 스크립트가 쓴 원시 RMS·ε로부터 통계를 재구성
  # [18] K2 X = (RMS_mix² − RMS_iid² − RMS_her²)/(RMS_iid·RMS_her), n=32 원시 RMS 대입
  - type: numeric
    expr: "((0.00011446399332413492**2 - 1.4034246409305986e-05**2 - 0.00010799629205976159**2)/(1.4034246409305986e-05*0.00010799629205976159)) - 0.8193510400305117"
    tol: 1.0e-9
  # [19] K1 ratio(128) = RMS_her(128)/RMS_iid(128), 원시 RMS 대입
  - type: numeric
    expr: "0.00023101582247149615/6.746565205221732e-06 - 34.24199061956648"
    tol: 1.0e-9
  # [20] K4 ratio = ε(64)/ε(8), 원시 ε 대입
  - type: numeric
    expr: "0.004474939214273945/0.03127810991675995 - 0.14306936148581373"
    tol: 1.0e-12
  # [21] 카드가 공개한 r48=ε(8)/ε(4)=0.5867(사전등록 훼손 사유)이 같은 seed에서 재현됨
  - type: numeric
    expr: "0.03127810991675995/0.05331084917780331 - 0.5867"
    tol: 1.0e-4
  # [22] K1 사전등록 ratio 숫자의 출처: √(E D_C(128)/127) = √(134587/127)
  - type: numeric
    expr: "sqrt(134587/127) - 32.554"
    tol: 1.0e-3
  # [23] K4 정확 항등식 예측값 (63/64²)/(7/64) = 9/64
  - type: identity
    lhs: "((64-1)/64**2)/((8-1)/8**2)"
    rhs: "9/64"
  # [24] defect 희석 법칙의 재배열 (n−1)/n² = 1/n − 1/n² (기호층)
  - type: identity
    lhs: "(n-1)/n**2"
    rhs: "1/n - 1/n**2"
---

# Q-0008 attempt-04 — 사다리 6단 수치시험 K1·K2·K5 (+K4 일관성) 사전등록 실행 기록

**이 문서는 유도가 아니라 사전등록 실행 기록이다.** 카드 `derivations/Q-0008/F-02.formula.md`(rev.2)의
6단 claim에 적힌 명령을 한 글자도 바꾸지 않고 실행했고, 결과를 본 뒤 카드·스크립트·창·seed·tol을 손대지
않았다. 기계 검사: 프론트매터 verify 블록(부등식 12·수치 11·항등식 2)과
`verify/Q-0008/attempt-04/result.json`(모드별 통계·창·통과 여부·실행 시각·소요·환경·해시),
`se_bootstrap.json`(표준오차), `log_<mode>.txt`(원본 stdout), `F-02_result_snapshot.json`(공유 result.json 사본).

## (S1) 규약 대조

$$ \text{script}\{\texttt{SEED},\delta,\texttt{MIN\_DET},\texttt{SIZES},\texttt{TRIALS},\texttt{MIX\_N},\texttt{MIX\_TRIALS},\texttt{DEFECT\_GRID},0.35,0.2,\texttt{PREREGISTERED}_{6},\texttt{WINDOWS}_{6}\}=\text{card} $$  (S1) `assemble_result.py`가 스크립트 상수를 import해 카드 전사값과 비교 — `constants_match_card: true`. seed 20260902(her의 iid 열은 20260903), δ=0.005, n∈{8,16,32,64,128}, 256 trials/size, mix n=32·1024 trials, defect 격자 {4,…,64}

## (S2) 실행

$$ \texttt{check\_modes.py --mode her}\ (99\,\mathrm s)\ \to\ \texttt{mix}\ (76\,\mathrm s)\ \to\ \texttt{iid}\ (48\,\mathrm s)\ \to\ \texttt{defect}\ (<1\,\mathrm s),\qquad 2026\text{-}09\text{-}02\ 14{:}13{:}09\text{–}14{:}16{:}53\ \mathrm{UTC} $$  (S2) 순차 실행(공유 result.json 병합 쓰기 경합 방지), 네 모드 모두 rc=0, MIN\_DET 기각 0회. 환경: Python 3.11.9, numpy 2.4.6, Windows 11 (10.0.26200), AMD64. 표준오차 재현 실행 209 s 추가 — 총 7.2 분

## (S3) 통계 정의 (스크립트와 동일)

$$ \mathrm{RMS}(n)=\sqrt{\tfrac1T\sum_{t=1}^{T}\epsilon_t(n)^2},\qquad \gamma=\text{slope of }\ln\mathrm{RMS}\text{ on }\ln n\ (\text{최소제곱, 5점}),\qquad X(32)=\frac{\mathrm{RMS}_{\rm mix}^2-\mathrm{RMS}_{\rm iid}^2-\mathrm{RMS}_{\rm her}^2}{\mathrm{RMS}_{\rm iid}\,\mathrm{RMS}_{\rm her}} $$  (S3) 12.4 정규화 simplicity 잔차 $\epsilon_t$의 trial RMS; verify[18][19][20]이 원시 RMS·ε에서 통계를 재구성한다

## (S4) 결과 — kill K1·K2·K5

| 통계 | 관측 | bootstrap SE | 95% CI | 창 | 사전등록 | 창 경계까지 σ (가까운 쪽) | 사전등록 대비 |
|---|---|---|---|---|---|---|---|
| K1 `her_slope` | **0.5576** | 0.0187 | [0.522, 0.595] | [0.43, 0.63] | 0.5302±0.10 | 3.87σ (위 0.63) | +1.47σ |
| K1 `her_ratio_128` | **34.24** | 1.68 | [31.06, 37.65] | [26.0, 39.1] | 32.554±6.5 | 2.88σ (위 39.1) | +1.00σ |
| K2 `mix_X_32` | **0.8194** | 0.0880 | [0.648, 0.997] | [0.49, 0.99] | 0.7406±0.25 | 1.94σ (위 0.99) | +0.89σ |
| K5 `iid_slope` | **−0.4800** | 0.0092 | [−0.498, −0.462] | [−0.58, −0.38] | −0.4783±0.10 | 10.9σ (양쪽) | −0.18σ |

$$ \gamma_{\rm her}=0.5576\in[0.43,0.63],\quad \frac{\mathrm{RMS}_{\rm her}(128)}{\mathrm{RMS}_{\rm iid}(128)}=34.24\in[26.0,39.1],\quad X(32)=0.819\in[0.49,0.99],\quad \gamma_{\rm iid}=-0.480\in[-0.58,-0.38] $$  (S4) 네 kill 통계 모두 창 안 — `kills_fired: []`. 세 통계는 사전등록값보다 위쪽(+1.5σ·+1.0σ·+0.9σ)에 있고 K5는 정확 격자 기울기 −0.4783과 0.2σ 안에서 일치한다

원시값: $\mathrm{RMS}_{\rm her}(n)=[4.987,7.664,11.849,17.038,23.102]\times10^{-5}$, 국소 기울기 $0.620\to0.628\to0.524\to0.439$(카드의 정확 국소 기울기 0.558→0.513과 같은 하강 추세이나 유한 trial 잡음이 크다), $\mathrm{RMS}_{\rm her}/\mathrm{RMS}_{\rm iid}=[1.974,3.831,8.427,17.704,34.242]$(정확값 1.988/3.998/8.048/16.195/32.554). i.i.d. $\mathrm{RMS}/(\sqrt{n-1}/n)=[7.85,7.97,8.07,7.65,7.97]\times10^{-5}$ — 형태 법칙의 앞인자가 격자 전체에서 5% 안에서 상수(기술 통계, 판정 아님). mix $n=32$: $\mathrm{RMS}_{\rm iid}=1.403\times10^{-5}$, $\mathrm{RMS}_{\rm her}=1.080\times10^{-4}$, $\mathrm{RMS}_{\rm mix}=1.145\times10^{-4}$.

## (S5) K4 일관성 검사 (kill 아님)

$$ \epsilon(n)=[5.331,3.128,1.690,0.878,0.447]\times10^{-2}\ (n=4,8,16,32,64),\qquad \frac{\epsilon(64)}{\epsilon(8)}=0.14307\in[0.124,0.158],\qquad \frac{d\ln\epsilon}{d\ln n}=-0.8982\in[-0.96,-0.86] $$  (S5) 결정론(Δ 표본 하나, SE 없음). 둘 다 창 안 → '일관'. 사전등록값 0.140625·−0.9069와의 차는 +0.0024·+0.0087(카드 uncertainty 0.017·0.05 안). 공개됐던 $r_{48}=\epsilon(8)/\epsilon(4)=0.5867$이 같은 seed에서 재현됨(verify[21]) — 카드가 예고한 조건부값 0.1425±0.0008·−0.8995±0.0035와 각각 0.7σ·0.4σ 안. 이 검사는 카드 rev.2가 정한 대로 기각 근거도 채택 근거도 아니다

## (S6) 정직성 메모

- 창 경계에 가장 가까운 통계는 K2 $X(32)$로 위 경계 0.99까지 1.94σ(95% CI 상단 0.997이 창을 살짝 넘는다). 창 안이지만 bootstrap 잡음 대비 여유가 2σ 미만이라, 독립 seed 재실행에서 위로 벗어날 확률이 무시할 수준은 아니다(대략 2.6%). 카드가 예고한 '창 위면 label 공분산 밖의 상관' 해석은 이 attempt에서 발동하지 않았다.
- K1 두 통계는 사전등록값보다 +1.5σ·+1.0σ 위이지만 카드 uncertainty(0.10·6.5) 안이며, 각각 위 창 경계까지 3.9σ·2.9σ.
- K5는 정확 격자 기울기와 0.2σ 안에서 일치 — 3단의 등방성 전제(traceless 평균 바닥 없음)가 이 격자에서 통제된다.
- 사전등록 통계는 스크립트가 쓴 값 그대로다. 표준오차는 같은 seed를 재현한 별도 스크립트(`check_se.py`)의 bootstrap이며, 재현 RMS와 스크립트 출력의 차는 정확히 0.0.
- 공유 `verify/Q-0008/F-02/result.json`에는 병행 실행된 attempt-05의 `qspine` 블록이 함께 들어 있다. 이 attempt의 판정 대상은 her·mix·iid·defect 넷뿐이며 K3(7단)는 여기서 평가하지 않는다.
- 스크립트·카드·창·seed·tol·trial 수는 실행 전후로 바꾸지 않았다(해시는 result.json `integrity`).

## 사다리 위치

6단(수치시험 K1·K2·K5)의 사전등록 실행이 끝났고 네 kill 통계가 모두 창 안이다. K4는 일관. 단 닫힘 여부는
judge가 정한다. 7단(K3 qspine)은 attempt-05의 몫이며 이 문서는 그 결과를 인용하지 않는다.

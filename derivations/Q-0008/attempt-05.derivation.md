---
question: Q-0008
attempt: 5
ladder_step: 7
claim: "카드 F-02 사다리 7단 예측시험 K3(Q-spine 블록)를 사전등록 그대로 실행한 결과, 두 통계가 모두 창 안이다:
  $n$-지수 $\\hat\\gamma_Q=0.49986\\pm0.0114$ (창 $[0.42,0.59]$, 사전등록 $0.5047$, 편차 $-0.42\\sigma$)와
  진폭비 $\\widehat{R}=\\mathrm{RMS}_Q(8)/\\mathrm{RMS}_{\\rm iid}(36)=6.6714\\pm0.193$ (창 $[6.01,7.65]$, 사전등록 $6.832$, 편차 $-0.83\\sigma$).
  K3는 발동하지 않았고, 판별 통계인 진폭비는 대안 셋(평균장 8.245·Cayley 9.064·chain 류 23.11)을 각각 $8.2\\sigma$·$12.4\\sigma$·$85\\sigma$로 배제한다."
assumptions:
  - "실행 규약은 카드 F-02 K3 문구와 `verify/Q-0008/F-02/check_modes.py --mode qspine`의 상수 그대로: seed 20260902(Q-spine·label 스트림), i.i.d. $n=36$ 스트림 seed 20260903, $b\\in\\{2,\\dots,8\\}$, 512 trials/b, $\\delta=0.005$, MIN_DET$=0.05$; 실행 전 상수 대조 10항목 일치, 실행 후 어떤 상수·창·숫자도 바꾸지 않았다"
  - "통계 정의는 스크립트 정의를 따른다: $\\mathrm{RMS}_Q(b)=\\sqrt{\\mathrm{mean}_{\\rm trials}\\,\\epsilon^2}$, 기울기는 $\\ln\\mathrm{RMS}_Q(b)$의 $\\ln E[n_b]$($E[n_b]=b(b+1)/2$ 정확값, 관측 평균 $n$이 아님)에 대한 최소제곱 기울기, 진폭비는 $b=8$의 RMS를 같은 trial 수의 i.i.d. $n=36$ RMS로 나눈 값"
  - "표준오차는 사전등록 항목이 아니라 이 attempt가 덧붙인 보고량이다: trial 단위 부트스트랩($B=2000$, seed 20260902; 깊이별·i.i.d. 블록별 복원추출)과 델타법($\\mathrm{Var}\\ln\\mathrm{RMS}=\\mathrm{Var}(\\epsilon^2)/(4T\\,\\overline{\\epsilon^2}{}^2)$) 둘을 보고하며 둘은 2% 이내로 같다. 창 통과 판정은 SE와 무관하게 점추정값의 창 포함 여부만으로 한다(카드 규약)"
  - "표준오차용 재현 스크립트 `verify/Q-0008/attempt-05/check_qspine.py`는 공식 스크립트와 같은 난수 호출 순서를 쓰며, 공식 실행의 RMS 7개·i.i.d. RMS·두 통계를 비트 단위로 재현했다(최대 차 0.0) — 별도 실행이 아니라 같은 실행의 trial 단위 기록이다"
  - "깊이별 관측/카드표 비(부수 진단)는 사전등록 kill 창이 아니며 판정에 쓰지 않는다"
  - "6단(K1·K2·K5, K4 일관성)은 이 attempt의 범위 밖이다(attempt 4가 병렬 실행). `verify/Q-0008/F-02/result.json`의 병합 사본에 그 모드 블록이 함께 들어 있으나 이 attempt는 qspine 블록만 실행·보고한다"
symbols:
  b: positive integer
  k: integer
verify:
  # [0] 사전등록 격자: E[n_b] = sum_{d=0}^{b-1}(b-d) = b(b+1)/2 (카드 verify[12], 4단)
  - type: identity
    lhs: "Sum(b-k,(k,0,b-1))"
    rhs: "b*(b+1)/2"
  # [1] 사전등록 진폭비 숫자의 출처: sqrt(E[D/n_8^2]) * 36/sqrt(35) = 6.8324 (카드 note 표 1.26071)
  - type: numeric
    expr: "sqrt(1.26071)*36/sqrt(35) - 6.8324"
    tol: 1.0e-4
  # [2] 진폭비 창 [6.01, 7.65] = 6.832 ± 0.82 (반올림 0.002 이내)
  - type: numeric
    expr: "abs(6.832 - 0.82 - 6.01) + abs(6.832 + 0.82 - 7.65)"
    tol: 5.0e-3
  # [3] 기울기 창 [0.42, 0.59] = 0.5047 ± 0.085 (반올림 0.0003 이내)
  - type: numeric
    expr: "abs(0.5047 - 0.085 - 0.42) + abs(0.5047 + 0.085 - 0.59)"
    tol: 1.0e-3
  # [4] 기록된 RMS 7개(result.json)로 최소제곱 기울기를 닫힌식으로 재계산 = 0.4998573796 (S4)
  - type: numeric
    expr: "(((log(3)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(2.4445995224839616e-05)+(log(6)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(3.7538025652719802e-05)+(log(10)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(4.6815328766737140e-05)+(log(15)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(5.7590550732849545e-05)+(log(21)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(6.8282495463181531e-05)+(log(28)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(7.6025621089825042e-05)+(log(36)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)*log(8.6758014992257244e-05))/((log(3)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(6)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(10)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(15)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(21)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(28)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2+(log(36)-(log(3)+log(6)+log(10)+log(15)+log(21)+log(28)+log(36))/7)**2)) - 0.4998573796"
    tol: 1.0e-9
  # [5] 기록된 RMS_Q(8)/RMS_iid(36) = 6.6713918570 (S4)
  - type: numeric
    expr: "8.6758014992257244e-05/1.3004484948818125e-05 - 6.6713918570"
    tol: 1.0e-9
  # [6] 기울기 점추정이 창 안 (S5)
  - type: inequality
    lhs: "0.42"
    rhs: "0.4998573796"
    relation: "<="
  - type: inequality
    lhs: "0.4998573796"
    rhs: "0.59"
    relation: "<="
  # [8] 진폭비 점추정이 창 안 (S5)
  - type: inequality
    lhs: "6.01"
    rhs: "6.6713918570"
    relation: "<="
  - type: inequality
    lhs: "6.6713918570"
    rhs: "7.65"
    relation: "<="
  # [10] 창 대비 σ 거리 (부트스트랩 SE 0.011401636·0.193028948): 기울기 하한 7.004σ, 진폭비 하한 3.426σ (S5)
  - type: numeric
    expr: "abs((0.4998573796 - 0.42)/0.011401636 - 7.004028) + abs((6.6713918570 - 6.01)/0.193028948 - 3.426387)"
    tol: 1.0e-4
  # [11] 4단 경로별 상계 sqrt(D/n^2) <= b 를 관측 진폭이 만족: sqrt(D/n^2)_obs = R·sqrt(35)/36 = 1.0963 <= 8 (S6)
  - type: inequality
    lhs: "6.6713918570*sqrt(35)/36"
    rhs: "8"
    relation: "<="
  # [12] 4단 상계가 주는 n-지수 상한 1/2: b-지수 1 ⇒ ln b / ln E[n_b] → 1/2 (chain 류 n^1 배제의 근거)
  - type: limit
    expr: "log(b)/log(b*(b+1)/2)"
    var: b
    point: oo
    expected: "1/2"
---

# Q-0008 attempt-05 — 사다리 7단 예측시험 K3 (Q-spine 블록): 사전등록 실행 기록

사전등록 실행 기록. 카드 `derivations/Q-0008/F-02.formula.md`의 K3를 `verify/Q-0008/F-02/check_modes.py --mode qspine`으로
문구 그대로 돌렸고(seed 20260902, $b\in\{2,\dots,8\}$, 512 trials/b, $\delta=0.005$; 실행 62 s, 기각 재추출 0회),
산출은 `verify/Q-0008/attempt-05/log_qspine.txt`(stdout)·`result_official_F-02.json`(공식 `result.json` 사본)·
`result.json`(같은 난수열의 trial 단위 재현 + 부트스트랩 SE, `check_qspine.py`)이다. 실행 후 상수·창·숫자는 고치지 않았다.

## (S1) 사전등록 통계와 창

$$ \mathrm{RMS}_Q(b):=\Bigl(\tfrac1T\sum_{t=1}^{T}\epsilon_t(b)^2\Bigr)^{1/2},\quad T=512,\qquad E[n_b]=\sum_{d=0}^{b-1}(b-d)=\frac{b(b+1)}2\in\{3,6,10,15,21,28,36\} $$  (S1.1) 통계 정의 — 12.4 정규화 simplicity 잔차의 trial RMS, 격자는 정확 $E[n_b]$ (verify[0])

$$ \hat\gamma_Q:=\underset{\rm OLS}{\rm slope}\bigl(\ln\mathrm{RMS}_Q(b)\ \text{on}\ \ln E[n_b]\bigr),\qquad \widehat R:=\frac{\mathrm{RMS}_Q(8)}{\mathrm{RMS}_{\rm iid}(36)} $$  (S1.2) 두 판정 통계 (`slope_vs_En`, `ratio_b8_over_iid36`)

$$ \hat\gamma_Q\in[0.42,0.59]=0.5047\pm0.085,\qquad \widehat R\in[6.01,7.65]=6.832\pm0.82,\qquad 6.832=\sqrt{1.26071}\cdot\frac{36}{\sqrt{35}} $$  (S1.3) 창의 출처 — 카드 K3 (verify[1][2][3]); 둘 중 하나라도 밖이면 K3 발동

## (S2) 상수 대조와 실행

$$ \text{SEED}=20260902,\ \text{iid seed}=20260903,\ \delta=0.005,\ b\in\{2..8\},\ T=512,\ n_{\rm iid}=36,\ \text{MIN\_DET}=0.05,\ \text{창·사전등록값 4개} $$  (S2.1) 스크립트 상수 10항목이 카드와 일치 (`check_qspine.py::assert_constants`, 실행 전 확인)

$$ \text{`check\_modes.py --mode qspine`}\ \to\ \texttt{verdict}=\{\text{slope: survive},\ \text{ratio: survive}\},\quad \text{rejections}=0,\quad 62\ \mathrm s $$  (S2.2) 공식 실행 (stdout `log_qspine.txt`; 14:15:45–14:16:47 UTC)

## (S3) 깊이별 결과

$$ \begin{array}{c|ccccccc} b & 2&3&4&5&6&7&8\\ \hline E[n_b] & 3&6&10&15&21&28&36\\ \bar n_{\rm obs} & 3.002&5.980&9.469&14.756&20.990&27.945&34.969\\ \mathrm{RMS}_Q/\delta^2 & 0.978&1.502&1.873&2.304&2.731&3.041&3.470\\ \mathrm{RMS}_Q(b)/\mathrm{RMS}_{\rm iid}(36) & 1.880&2.887&3.600&4.429&5.251&5.846&6.671\\ \text{카드표}\ \sqrt{E[D/n_b^2]}\,36/\sqrt{35} & 1.941&2.806&3.630&4.441&5.238&6.037&6.832\\ \text{관측/카드표} & 0.969&1.029&0.992&0.997&1.002&0.968&0.976 \end{array} $$  (S3.1) `result.json` `per_depth_accompanying` — $\mathrm{RMS}_{\rm iid}(36)/\delta^2=0.5202$; 마지막 두 줄은 부수 진단이지 사전등록 창이 아니다(깊이별 델타법 SE는 0.055…0.196으로, 7점 모두 $1.3\sigma$ 이내)

## (S4) 두 통계의 점추정

$$ \hat\gamma_Q=\frac{\sum_b(x_b-\bar x)\ln\mathrm{RMS}_Q(b)}{\sum_b(x_b-\bar x)^2}\Big|_{x_b=\ln E[n_b]}=0.4998574 $$  (S4.1) (S3.1)의 RMS 7개로 닫힌식 재계산 (verify[4], 잔차 $10^{-11}$) — 공식 `stats.qspine_slope_vs_En`과 비트 일치

$$ \widehat R=\frac{8.675801\times10^{-5}}{1.300448\times10^{-5}}=6.6713919 $$  (S4.2) (verify[5]) — 공식 `stats.qspine_ratio_b8_over_iid36`과 비트 일치

$$ \hat\gamma_b:=\underset{\rm OLS}{\rm slope}\bigl(\ln\mathrm{RMS}_Q(b)\ \text{on}\ \ln b\bigr)=0.8983 $$  (S4.3) 부수 관측 — 카드 note의 tree-only $b$-기울기 0.906과 비교(창 없음)

## (S5) 표준오차와 창 대비 위치

$$ \mathrm{SE}_{\rm boot}(\hat\gamma_Q)=0.01140,\quad \mathrm{SE}_{\delta}(\hat\gamma_Q)=0.01156,\quad \mathrm{CI}_{95}^{\rm boot}=[0.4776,\,0.5229] $$  (S5.1) 부트스트랩($B=2000$, seed 20260902)과 델타법이 1.4% 이내로 일치

$$ \mathrm{SE}_{\rm boot}(\widehat R)=0.1930,\quad \mathrm{SE}_{\delta}(\widehat R)=0.1957,\quad \mathrm{CI}_{95}^{\rm boot}=[6.289,\,7.052] $$  (S5.2) 같은 방법

$$ \hat\gamma_Q=0.4999\pm0.0114:\ \ \frac{0.4999-0.42}{0.0114}=7.00\sigma\ \text{(하한 위)},\ \ \frac{0.59-0.4999}{0.0114}=7.91\sigma\ \text{(상한 아래)},\ \ \frac{0.4999-0.5047}{0.0114}=-0.42\sigma $$  (S5.3) 창 [0.42,0.59] 안, 사전등록값과 $0.42\sigma$ (verify[6][7][10])

$$ \widehat R=6.671\pm0.193:\ \ \frac{6.671-6.01}{0.193}=3.43\sigma\ \text{(하한 위)},\ \ \frac{7.65-6.671}{0.193}=5.07\sigma\ \text{(상한 아래)},\ \ \frac{6.671-6.832}{0.193}=-0.83\sigma $$  (S5.4) 창 [6.01,7.65] 안, 사전등록값과 $0.83\sigma$ (verify[8][9][10])

## (S6) 사전등록 예측과 대안의 대조

$$ \begin{array}{l|cc|cc} \text{가설} & \gamma & \text{관측과의 거리} & R & \text{관측과의 거리}\\ \hline \text{F-02 K3 (사전등록)} & 0.5047\pm0.085 & -0.42\sigma & 6.832\pm0.82 & -0.83\sigma\\ \text{chain 류 (spine 지배)} & 1.0 & 43.9\sigma & 23.11 & 85.2\sigma\\ \text{평균장}\ \sqrt{E D}/E n & 0.533 & 2.91\sigma & 8.245 & 8.15\sigma\\ \text{균등 Cayley}(36) & - & - & 9.064 & 12.4\sigma \end{array} $$  (S6.1) 거리는 관측 부트스트랩 SE 단위 — 기울기 창은 평균장 0.533을 포함하므로(카드 문구) 기울기는 평균장을 $2.9\sigma$로만 가르고, 규약 판별은 진폭비가 $8.2\sigma$로 담당한다; chain 류는 두 통계 모두에서 배제

$$ \sqrt{D/n^2}\Big|_{\rm obs}=\widehat R\cdot\frac{\sqrt{35}}{36}=1.096\le b=8,\qquad \lim_{b\to\infty}\frac{\ln b}{\ln\,b(b+1)/2}=\frac12 $$  (S6.2) 4단 경로별 상계 $D/n^2\le b^2$를 관측이 만족하고, $b$-지수 1이 $n$-지수 $1/2$로 옮겨지는 격자 관계 (verify[11][12]) — chain 류 $n^1$은 정리로 배제되며 기울기 창은 그 수치 확인

## (S7) 판정과 지위

$$ \text{K3: 발동 없음}\quad(\hat\gamma_Q\in[0.42,0.59]\ \wedge\ \widehat R\in[6.01,7.65]) $$  (S7.1) 카드 '죽는 조건' 문구 그대로 — 7단 예측시험은 사전등록 창을 통과

$$ \text{7단 통과의 뜻: 가정된 }P_{\rm micro}(\text{가우스 등방 }\kappa\otimes I_{16},\ O(\delta^4)\text{ 절단})\cdot w_{\rm her}\ne0\text{ 공리}\cdot\text{고정 }\epsilon_{\rm res}\text{ 규약 아래의 조건부 진술} $$  (S7.2) 카드 7단 claim의 조건절을 그대로 옮김 — 사다리 완주(3·4·5·6단 닫힘)는 이 attempt의 결과가 아니며 judge가 정한다

$$ \text{하지 않은 것: 6단(K1·K2·K5, K4 일관성) 실행·보고 — attempt 4 몫;\ 관측 평균 }n\text{에 대한 재적합·격자 변경·창 재중심화 — 하지 않음} $$  (S7.3) 병합된 `F-02/result.json` 사본에 attempt 4가 쓴 her/mix/iid/defect 블록이 함께 있으나 이 attempt는 qspine 블록만 실행했다

## 검증 요약

| 항목 | 값 | 창 | 결과 |
|---|---|---|---|
| `qspine_slope_vs_En` | $0.49986\pm0.0114$ | $[0.42,0.59]$ | 안 (하한 $7.0\sigma$, 상한 $7.9\sigma$) |
| `qspine_ratio_b8_over_iid36` | $6.6714\pm0.193$ | $[6.01,7.65]$ | 안 (하한 $3.4\sigma$, 상한 $5.1\sigma$) |
| 공식 실행 재현 | RMS 7개·iid RMS·두 통계 | 비트 일치 | 최대 차 0.0 |
| 기각 재추출 | 0회 | MIN_DET 0.05 | 선언대로 |

산출물: `verify/Q-0008/attempt-05/result.json`, `result_official_F-02.json`, `log_qspine.txt`, `log_check_qspine_se.txt`, `check_qspine.py`.

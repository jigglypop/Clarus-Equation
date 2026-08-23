# 12-routes — BA-TR30 P0-1 우회 경로

Status: COMPLETE

실행 사유: 11-math P0-1 — 동결 $\tau_{\rm class}=\max(10^{-8},8\eta)$ + mean-L2 LOO 통계가 $d^\*=3,\eta>0$ 주 fold의 79–80%를 false abstain시켜 §4.3 "전 fold 필수" endpoint와 구조적으로 모순.

목표량(1문장): $D=\{1,2,3\}\times H=\{0,10^{-3},10^{-2}\}$ 전 주 fold에서 기권 없이 $\hat d=d^\*$ 식별과 게이트 내 예측을 발행하되, 차수-4 witness 기권(여유 $\gg1$)·shuffle 기각·독립 후보 스트림 제약을 그대로 유지한다.

정본 강제 부분: LOO 정의·hat 항등식·witness 기권·§4.4 독립성·§4.5 대조군·TR28/29 no-go. 자유 부분: 기권 통계의 형태와 상수, 파시모니 비교식, $N$.

주의 — §5 교정 레지스터상 §4.2 상수·통계는 calibration 개봉 후 "교정 불가" 범주다. 아래 후보는 전부 **계약 개정(또는 후속 run) 수준의 수리안**이며, 채택 판단은 오케스트레이터·auditor 소관. 본 레인은 수치만 공급한다.

| 후보 | 경로 | dof | target-aware | 핵심 수치 (`artifacts/math-verify/*.json`) | 죽이는 반증 시험 |
|---|---|---:|---|---|---|
| R-A | $\tau_{\rm class}$ 상수 재교정: $8\eta\to c\,\eta$, $c\approx60$ ($s_3/\eta$ p99=56에서 역산) | 1 | 예 | false abstain $\approx1\%$/fold 잔존(max 81); witness 여유 $856\eta/60\eta=14.3\times$; shuffle 최소 $s=1.26>0.6$ 기각 유지 | witness 기권 <100% 또는 shuffle 기각 <100%가 한 fold라도 발생하면 폐기. 96 잡음 fold에서 잔존 1%로 STOP 확률 여전히 높음 |
| R-B | 기권 통계 교체: studentized PRESS $s'_d=\frac1N\sum_i\lVert e_i^{\rm loo}\rVert\sqrt{1-h_{ii}}$ (성분 표준편차가 정확히 $\eta$가 되어 레버리지 꼬리 제거), $\tau=\max(10^{-8},8\eta)$ 유지 | 1 (통계 선택) | 예 | false abstain 0/200 전 9셀; witness 기권 200/200, 여유 $234\eta=29\tau$; shuffle 최소 $s'=0.94\gg0.008$ | 새 통계에서 witness 여유가 $10\times$ 미만으로 떨어지거나 shuffle 통과 사례 발생 시 폐기 |
| R-C | 파시모니 floor: $\hat d=\min\{d: s_d\le(1{+}\rho)\min_{d'}s_{d'}+10^{-8}\}$ | 1 | 예 | $\eta=0,d^\*{=}1$ 오선택 4/200 → 0/200; 잡음 셀 선택 불변 | floor가 $\eta>0$에서 과소 차수 선택을 유발하면($\hat d<d^\*$ 1건이라도) 폐기 — 관측상 0건 |
| R-D | $N$ 증가 $14\to24$ (TR28의 24행 관행): $p/N$ 하락으로 레버리지·게이트 꼬리 동시 완화 | 1 | 아니오 (구조적) | 미시뮬레이션 — 방향만 유도: $\bar h=10/24=0.42$로 감소, $1/(1-h)$ 꼬리 축소 | $N=24$ 재시뮬레이션에서 false abstain·게이트 초과가 fold당 $10^{-3}$ 미만으로 안 떨어지면 폐기 |

결합 R-B+R-C (측정 완료, `routes_log.json`): 전 9셀 false abstain 0, 비기권 식별 99–100%, 식별+게이트 0.99–1.0/셀. 잔존 결함: fold당 $\sim0.5$–$1\%$ 오선택·게이트 꼬리(P1-3) → development 144 주 fold 전량 통과 확률 $\approx e^{-0.88}\approx0.41$. 즉 **R-B+R-C만으로는 "전 fold 필수" 규율 아래 STOP 확률이 여전히 $\sim59\%$** — 완전 수리는 R-D(또는 게이트 상수 여유 재산정) 병행이 필요하다.

순위: R-B+R-C (낮은 dof, 상수 8η·게이트 보존, witness/shuffle 교차 예측 전부 통과) > R-D > R-A. R-A는 잔존 실패율이 가장 크고 상수가 관측 분포에서 직접 역산된 target-aware 단독 수리라 최하위.

look-elsewhere: 후보 4개 전부 P0 관측 후 구성 — 채택 시 fresh calibration seed(117001)에서 사전 고정 후 development를 열어야 하며, 본 레인 수치는 설계 근거이지 통과 증거가 아니다.

경로 부재 아님: 죽는 경로 확인 — "동결 §4.2 문면 그대로 실행" 경로는 반례로 사망(11-math P0-1). 재조정 금지 조건(§4.7-3) 때문에 run 내 수리는 calibration 단계 계약 해석에 달렸고, 그 판단은 본 레인 소관이 아니다.

## 부록 — R-D 폐기 조건 시험 결과 (오케스트레이터 요청, 후속 검산)

R-D 행의 "미시뮬레이션"은 본 부록으로 대체된다. $N=24$ + R-B(studentized PRESS) + R-C(floor) 결합, master seed 910024, 셀당 6000 draws (`artifacts/math-verify/rd_confirm_6000_log.json`): 전 9셀에서 기권·오식별·게이트 초과 각 0/6000 (총 실패율 95% 상한 $5{\times}10^{-4}<10^{-3}$), witness 기권 6000/6000 여유 최소 41.9배, shuffle 기각 3000/3000. R-D 폐기 조건(fold당 실패율 $10^{-3}$ 미만 미달)은 **발동하지 않았다**. 본문 "R-B+R-C만으로는 STOP 확률 $\sim59\%$" 추정은 $N=14$ 기준이며, $N=24$ 결합에서는 144 fold 전량 통과 보수 하한 $\approx0.931$로 개선된다. 채택·계약 개정 판단은 오케스트레이터·auditor 소관이다.

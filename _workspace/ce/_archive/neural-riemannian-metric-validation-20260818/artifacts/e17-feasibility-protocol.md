# E17 유효동역학 계량 탐색 프로토콜

Status: FROZEN AFTER SCHEMA INSPECTION AND BEFORE NUMERIC OUTCOME ANALYSIS

## 범위

이 분석은 공개 E17 Figure 2의 saline/DCZ 단일 시행 시계열에서 유효동역학 계량의 기술적 가능성을 검사한다. 직접 시냅스 연결 $W^s$의 학습 전후 변화가 없으므로 `NRM-H1A`를 검정하지 않는다. DCZ는 NDNF interneuron을 활성화하는 gain/inhibition 조작이므로 결과는 `NRM-H4`와 effective-connectivity 대리인 `NRM-H1B`의 탐색적 자료로만 쓴다.

## 표본과 chart

입력은 Figure 2의 11개 `DCO*_dff.mat` 파일에 있는 `cont_data.Sal.branch`와 `cont_data.DCZ.branch`다. 파일명 접두사 `DCO1`, `DCO2`, `DCO4`를 동물 식별자로 사용한다. 세션, 시행, frame과 ROI는 반복 측정이며 독립 표본으로 세지 않는다.

각 세션과 조건에서 공개 배열 순서의 앞 60%를 calibration, 뒤 40%를 봉인한 test로 사용한다. 공개 파일에는 이 배열이 실제 취득 시간 순서임을 확인할 timestamp provenance가 없으므로 결과를 미래 시행 예측이라고 부르지 않고 held-out 시행 예측이라고 부른다. 두 조건의 calibration frame을 합친 평균과 표준편차로 공통 대각 chart를 정한다. calibration 표준편차가 $10^{-8}$ 이하이거나 유한값이 아닌 ROI는 두 조건 모두에서 제거한다. test 값은 chart 선택이나 모형 선택에 쓰지 않는다.

## 동역학과 계량

각 조건의 calibration 전이를 사용해 절편을 포함한 OLS 모형을 맞춘다.

$$
z_{t+1}=Jz_t+b+\varepsilon_t,
\qquad
\operatorname{Cov}(\varepsilon_t)=Q.
$$

$Q$ 추정치에는 수치 안정성을 위해 $10^{-6}\operatorname{tr}(Q)/r$를 대각에 더한다. 이것은 과정 잡음 추정량의 ridge이며 metric ridge가 아니다. 고정 horizon은 $H=5$ frames고 metric ridge는 $\lambda=0$으로 고정한다. 시간 불변 모형의 과정 잡음 도달가능성 공분산과 계량은 다음과 같다.

$$
C_H=\sum_{k=0}^{H-1}J^kQ(J^k)^\top,
\qquad
g_H=C_H^{-1}.
$$

공통 chart를 이 분석 전체에서 고정한다. $Q$의 isotropic ridge는 비직교 chart 변화에 대해 공변적이지 않으므로 이 결과는 일반 좌표 불변성을 주장하지 않는다.

유효 표본 전이가 $10(r+1)$개 미만이거나 $C_H$가 양의 정부호가 아니면 해당 세션을 실패로 기록하고 대체 초매개변수를 탐색하지 않는다.

## 봉인한 test 종말점

각 test trial의 서로 겹칠 수 있는 모든 $t\to t+H$ 쌍에서 $H$-step 평균과 $C_H$를 사용해 Gaussian negative log predictive density(NLPD)를 계산한다. 이 쌍들을 독립 표본으로 세지 않고 먼저 trial과 session 안에서 요약한다. 세션별로 다음 점수를 기록한다.

- own-condition full covariance NLPD
- 다른 조건 모형의 NLPD
- own-condition 평균과 $\operatorname{diag}(C_H)$를 쓰는 diagonal baseline
- own-condition 평균과 $\operatorname{tr}(C_H)I/r$를 쓰는 isotropic baseline
- persistence 평균과 calibration에서 추정한 isotropic residual variance를 쓰는 baseline

또한 saline과 DCZ $C_H$ 사이 affine-invariant SPD distance, predicted log-determinant change, test residual covariance의 observed log-determinant change와 두 변화의 부호 일치를 기록한다. 점수는 먼저 세션 안에서 평균한 뒤 동물 안에서 평균한다.

## 판정 제한

동물이 3마리뿐이므로 $p$값을 confirmatory population evidence로 보고하지 않는다. own-condition과 full-covariance 점수가 일관되게 좋아도 effective-dynamics feasibility일 뿐이다. $g_H$는 $J,Q$의 결정적 요약이므로 충분한 direct state-space 모형보다 새로운 정보를 더한다는 `NRM-H2` 증거로 세지 않는다. Figure 3의 iGluSNFR, Figure 4의 종단 수상돌기, Figure 2의 조작 자료를 서로 이어 한 연쇄로 해석하지 않는다.

Figure 4 재구현은 공개 분석 코드가 지정한 같은 dendrite의 day-pair Pearson correlation을 계산하되, 공식 기대 수치와의 equality/tolerance 검사가 없으므로 재현 성공이라고 부르지 않는다. animal ID가 없는 dendrite-level 값은 기술통계로만 보고한다. Figure 3의 synapse 개수와 field 구조는 동일 시냅스 pre/post 자격 판정에만 사용한다.

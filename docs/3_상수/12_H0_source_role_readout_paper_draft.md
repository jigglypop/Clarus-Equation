# $H_0$ source-role 원고 재현성 상태 **[미완성]**

이 파일은 논문 결론이 아니라 원고 재개 조건을 기록한다. 조건부 수학은
[우주론 수식 문서](9_우주론_수식_의미와_후보.md), 데이터 의존성은
[readout 재현성 노트](../검증_원장/상수_H0_readout_law_audit.md)와
[TDCOSMO 재현성 노트](../검증_원장/상수_TDCOSMO_real_covariance_audit.md)에 분리한다.

독자는 선형회귀, 공분산 행렬, rank와 관측 likelihood의 기본 뜻을 안다고 가정한다. 먼저 원고가 아직 재현 불가한 이유를 확인하고, 이어 source-role 선형 toy model이 어떤 가정에서만 식별 가능한지 읽은 뒤, 마지막으로 실제 원고를 재개하기 위한 입력·검증·반증 조건을 확인하는 순서로 읽는다.

현재 checkout에는 $\texttt{examples/physics/h0_readout/h0_paper_package_gate.py}$
및 그 입력·산출물이 없다. 따라서 표·그림, 수치 likelihood와 package 재현을
완료했다고 기록하지 않는다.

## source-role의 닫힌 선형 toy model

현재 checkout에 원자료와 실행 패키지가 없으므로, 이 절은 실제 $H_0$ 관측 결론이 아니라 식별 가능성을 설명하는 조건부 선형 모형만 다룬다. 서로 다른 source role은 서로 다른 열방향으로 구별되어야 하며, 같은 모양의 열을 반복해도 새로운 정보가 생기지 않는다는 것이 아래 rank 조건의 적용 경계다.

**[공리]** source별 요약벡터를

$$
y=X\beta+\varepsilon,\qquad
\varepsilon\sim N(0,C),\quad C>0
$$

로 두고 $X$의 각 열을 사전에 정의한 source role로 해석한다.

**[정리]** $\beta$가 식별 가능한 것과 $X$가 full column rank인
것은 동치다. 이때 generalized least-squares 추정량은

$$
\widehat\beta
=(X^TC^{-1}X)^{-1}X^TC^{-1}y,
\qquad
\operatorname{Cov}(\widehat\beta)
=(X^TC^{-1}X)^{-1}.
$$

rank가 부족하면 서로 다른 role 조합이 같은 평균을 만든다. 같은
rank-deficient 행 패턴을 반복 측정하기만 해서는 분해가 고유해지지 않으며,
식별하려면 새로운 독립 열방향을 드러내는 설계행이 필요하다. 실제 원고에서는
$X,C,y$를 원자료와 함께 공급해야 이 조건부 모형을 사용할 수 있다.

## 원고 재개 조건

toy model의 GLS 공식은 $X,C,y$와 source 정의가 실제로 공개되고 고정될 때만 원고에 적용할 수 있다. 아래 항목은 결과를 꾸미기 위한 체크리스트가 아니라 재현 가능한 likelihood·ablation·holdout을 제공하지 못할 경우 source-role 주장을 반증 불가능한 상태로 남긴다는 최소 조건이다.

- 실제 source manifest와 공개 covariance/Fisher 원자료
- 고정된 source-role 정의와 사전 지정된 ablation
- 기준모형과 동일한 likelihood 및 nuisance-parameter 처리
- 표·그림 생성 코드, 실행 환경과 checksum
- 독립 관측 채널에서의 holdout 비교

위 항목이 갖춰질 때까지 source-role 관측 주장은 **[미완성]**이다.

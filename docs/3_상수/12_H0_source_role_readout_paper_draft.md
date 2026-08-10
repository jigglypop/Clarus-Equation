# \(H_0\) source-role 원고 재현성 상태 **[미완성]**

이 파일은 논문 결론이 아니라 원고 재개 조건을 기록한다. 조건부 수학은
[우주론 수식 문서](9_우주론_수식_의미와_후보.md), 데이터 의존성은
[readout 재현성 노트](10_H0_readout_law_audit.md)와
[TDCOSMO 재현성 노트](11_TDCOSMO_real_covariance_audit.md)에 분리한다.

현재 checkout에는 \(\texttt{examples/physics/h0_readout/h0_paper_package_gate.py}\)
및 그 입력·산출물이 없다. 따라서 표·그림, 수치 likelihood와 package 재현을
완료했다고 기록하지 않는다.

## source-role의 닫힌 선형 toy model

**[공리]** source별 요약벡터를

\[
y=X\beta+\varepsilon,\qquad
\varepsilon\sim N(0,C),\quad C>0
\]

로 두고 \(X\)의 각 열을 사전에 정의한 source role로 해석한다.

**[정리]** \(\beta\)가 식별 가능한 것과 \(X\)가 full column rank인
것은 동치다. 이때 generalized least-squares 추정량은

\[
\widehat\beta
=(X^TC^{-1}X)^{-1}X^TC^{-1}y,
\qquad
\operatorname{Cov}(\widehat\beta)
=(X^TC^{-1}X)^{-1}.
\]

rank가 부족하면 서로 다른 role 조합이 같은 평균을 만든다. 같은
rank-deficient 행 패턴을 반복 측정하기만 해서는 분해가 고유해지지 않으며,
식별하려면 새로운 독립 열방향을 드러내는 설계행이 필요하다. 실제 원고에서는
\(X,C,y\)를 원자료와 함께 공급해야 이 조건부 모형을 사용할 수 있다.

## 원고 재개 조건

- 실제 source manifest와 공개 covariance/Fisher 원자료
- 고정된 source-role 정의와 사전 지정된 ablation
- 기준모형과 동일한 likelihood 및 nuisance-parameter 처리
- 표·그림 생성 코드, 실행 환경과 checksum
- 독립 관측 채널에서의 holdout 비교

위 항목이 갖춰질 때까지 source-role 관측 주장은 **[미완성]**이다.

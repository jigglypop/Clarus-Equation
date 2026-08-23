# 30-implementation — BA-SRM1 실제 시냅스 부분공간 실행

Status: COMPLETE

## 구현 범위

이번 구현은 Allen-SynPhys r2.1 small SQLite에서 mouse V1의 흥분성(`ex`)과
억제성(`in`) 연결을 분리해 strict 4차원 측정모형만 실행했다. raw event가 없는
small 판본에서 conductance, release probability, directed delay, STDP eligibility,
homeostasis 또는 구조 변화를 보간하지 않았다.

입력 좌표는

$$
z=\left(
\log\frac{|r_1|}{r_{\rm ref,\chi}},
\log\frac{L_{\rm soma}}{1\,{\rm m}},
\log\frac{R_{\rm in,post}}{1\,\Omega},
\log\frac{\tau_{m,\rm post}}{1\,{\rm s}}
\right)
$$

이고, target은

$$
y=\left(
s_\chi a_2/r_{\rm ref,\chi},
s_\chi a_{6:8}/r_{\rm ref,\chi},
s_\chi a_{9:12}^{250\rm ms}/r_{\rm ref,\chi},
v_{5:8}
\right)
$$

이다. 모든 기준값, 평균, covariance, whitening, target scale, ridge 계수,
$R_\chi$, 그래프와 hyperparameter는 train에서만 만들었다.

## 데이터 접근 순서

1. `query_feature_rows()`가 target 네 열을 선택하지 않은 채
   `synapse → pair → experiment → slice → dynamics → post intrinsic → cell`을
   join했다.
2. slice ID의 SHA-256 첫 byte로 train/development/confirmation을 고정했다.
3. `query_targets()`는 train+development pair ID에 대해서만 target 네 열을
   읽었다.
4. rank, gauge, development 예측이 모두 통과한 stratum에 대해서만 별도의
   confirmation query를 실행하도록 fail-closed 분기를 구현했다.
5. 실제 실행에서는 두 stratum 모두 중지돼 confirmation query가 호출되지
   않았다.

join 뒤 pair ID와 synapse ID 중복이 하나라도 생기면 집계하지 않고 즉시
실패한다. NULL, 비유한값, 비양수 strict input은 epsilon이나 imputation 없이
제외한다.

## 수치 모형

train $z$ covariance를 $\Sigma$라 할 때

$$
\Sigma_s=\Sigma+10^{-6}\frac{\operatorname{tr}\Sigma}{4}I,
\qquad x=L^{-1}(z-\bar z),\quad LL^T=\Sigma_s
$$

로 whitening했다. 따라서 고정된 $x$ chart에서 $g_{\rm ref}=I$이고, 이는 원래
chart의 $\Sigma_s^{-1}$을 공변적으로 옮긴 표현이다. target도 train 평균과
표준편차로 표준화했다.

반응식 $\mathcal H_2$는 절편 1개, 선형항 4개, $i\le j$인 2차항 10개를 가진
15-basis 다출력 ridge다. slice-grouped 5-fold CV에서
$\alpha\in\{10^{-6},\ldots,10^2\}$를 선택했다. 야코비안은 basis의 해석
미분으로 계산하며

$$
g_{\rm resp}(x)=J(x)^TR_\chi^{-1}J(x)
$$

를 구성했다. $R_\chi$는 train residual의 target별 대각 분산이다.

그래프는 train $g_{\rm ref}$ 거리의 symmetric-union kNN으로 만들었다. edge
길이는 endpoint metric의 사다리꼴 평균을 썼고, Dijkstra 최단경로를 계산했다.
held-out query는 가장 가까운 train node $k$개에만 붙였다. query-query edge와
test 기반 adjacency 재구성은 없다. 실제 bandwidth는 각 train graph edge
길이 중앙값에 고정 multiplier를 곱했다.

## 비교 모형

동일한 development rows에서 다음 control을 비교했다.

- reference Euclidean graph;
- diagonal response metric graph;
- train-mean constant full response metric graph;
- direct quadratic $\mathcal H_2$;
- raw four-factor linear ridge;
- strength-only, distance-only, membrane-only ridge;
- train global mean과 source cell-type category mean.

strict complete-case 분석에서는 missingness-only가 global mean과 같아 별도
중복 점수를 만들지 않았다. small DB의 event row 수가 0이므로 protocol-order
shuffle은 `UNAVAILABLE_SMALL_DB_HAS_ZERO_EVENT_ROWS`로 기록했다.

## 사후 구현감사 수정

독립 구현감사는 최초 코드가 bootstrap draw 안에서 support의 5% 분위수를
사용해 계약의 support 최솟값보다 약하다는 P1을 찾았다. 결과를 통과시키는
방향이 아니라 더 엄격한

$$
T_b=\min_{z\in\mathcal S}
\frac{\lambda_{\min}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
{\lambda_{\max}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
$$

로 수정하고 같은 seed로 재실행했다. 자세한 provenance는
`revisions/01-bootstrap-minimum-correction.md`에 있다. 예측값과 STOP 판정은
변하지 않았고 confirmation은 계속 봉인됐다.

## 구현 파일과 최종 SHA-256

| 파일 | 역할 | SHA-256 |
|---|---|---|
| `schema_audit.py` | outcome-free schema/support 감사 | `8d8484715c8cd6198ec57405778dd1527502ff45ad18cdd2057efc429711459f` |
| `srm1_analysis.py` | 수치 primitive, graph, rank, gauge | `bd80e263951f7719894662f88b6e37bff5a4a0765cc0d57abd0fa51e36d92403` |
| `run_analysis.py` | fail-closed 실제자료 실행 | `4fb43d238bea667b94a767fa1364a311629a0f060271fe358ddd58c4c3fd606b` |
| `test_srm1_analysis.py` | 집중 수치 회귀검사 | `8c12c90bd873a9239ae3927ac5224c46d1ff66b42ab2b4688c7cf1053367573b` |
| `artifacts/analysis/results.json` | 최종 기계 결과 | `1b34d7cf39f2d7241b5ad69bb14040328639685f09b394b0728f0def57412d7c` |


# G9-R: 실제 fsaverage 피질 기하 검증

TemplateFlow의 좌반구 fsaverage 10k pial surface, sulcal depth, scalar curvature를
사용한다. 꼭짓점은 10,242개, 삼각형은 20,480개이며 다운로드 총량은
373,423 bytes다. 2.3 GB인 HCP S1200 묶음은 비용 제한 때문에 `SKIPPED_COST`로
남겼다.

이 단계는 단일 집단평균 표면의 연관성 검증이다. 발생, 유전, 개인차, 기능,
인과성이나 잠재공간의 꼬임을 직접 검증하지 않는다. 중심 좌표의 방위각을 여덟
sector로 나눠 짝수 sector를 validation, 홀수 sector를 한 번만 여는 locked test로
사용한다.

## V1: 파서 실패

`GZipBase64Binary`를 gzip wrapper로만 해석해 zlib stream에서 실패했다. 이는
과학 가설 실패가 아니라 입력 파서 실패다. V2에서 zlib 우선, gzip fallback 및
decoded element-count 검사를 등록했다.

## V2: one-ring 국소기하 강한 가설 실패

기준은 전역 위치와 제공된 scalar curvature다. 후보에는 one-ring edge 길이,
좌표 Laplacian 크기와 방사 성분, 이웃 covariance 고유값 비율을 추가했다.

- 기준 RMSE: 0.363450
- 후보 RMSE: 0.357108
- 개선: 1.745% — 등록 기준 5% 미달
- 기준 부호 정확도: 76.729%
- 후보 부호 정확도: 76.250% — 0.479%p 하락

따라서 좌표 의존 one-ring 특징이 scalar curvature를 넘어 강한 추가 설명력을
가진다는 주장은 반증됐다.

## V3: 위치-only 기준에 대한 약한 가설

V3는 기준을 전역 위치만으로 낮추고 curvature와 local geometry의 결합을
비교했다. validation에서 RMSE가 33.64% 줄고 부호 정확도가 14.59%p 올랐지만,
scalar-curvature tier 대비 국소 특징의 추가 RMSE 개선은 1.745%이고 부호 정확도는
0.479%p 하락했다. 그러므로 V3 통과는 주로 이미 측정된 curvature의 정보이며
국소 방향성이나 꼬임의 증거가 아니다.

## V4: Laplace–Beltrami 내재기하

V4는 one-ring 요약을 질량집중 positive-cotangent Laplace–Beltrami 연산자로
교체하고 scalar-curvature tier를 다시 강한 기준으로 사용한다. 삼각 메시의
질량행렬 `M`과 cotangent stiffness `W`로

\[
\Delta_M f(i)=\frac{1}{M_i}\sum_j w_{ij}(f_j-f_i)
\]

를 계산한다. 음의 cotangent edge는 사전등록대로 0으로 잘라 양의 확산 연산자를
만든다. 조밀한 10,242 × 10,242 행렬이나 전체 고유분해는 만들지 않고 edge flux로
연산한다.

추가 특징은 다음과 같다.

- 곡률장의 Laplace–Beltrami: `Δ_M c`
- carré-du-champ gradient energy:
  `Γ(c)=1/2[Δ_M(c²)-2cΔ_M c]`
- embedding의 mean-curvature magnitude: `0.5 ||Δ_M X||`
- 고정된 1, 4, 16 explicit heat step의 `exp(tΔ_M)c` 근사

열확산 step size는 `0.45 / max_i(degree_i/M_i)`로 고정했다. 전체 메시의 기하와
curvature는 transductive covariate로 사용할 수 있지만 holdout sector의 sulcal-depth
label은 특징 구성이나 학습에 사용하지 않는다.

사전등록 성공 기준은 scalar 기준 대비 RMSE 3% 이상 감소, 평균 부호 정확도
0.5%p 이상 증가, 네 sector 중 최소 세 sector의 RMSE 개선이다.

### Validation 결과

- 기준 RMSE: 0.363450
- 후보 RMSE: 0.328318
- RMSE 감소: 9.666%
- 부호 정확도: 76.729% → 78.110%, +1.381%p
- RMSE 개선 sector: 4/4
- 결과: `PASS`

### Locked test 결과

- 기준 RMSE: 0.405757
- 후보 RMSE: 0.374275
- RMSE 감소: 7.759%
- 부호 정확도: 75.846% → 77.175%, +1.328%p
- RMSE 개선 sector: 4/4
- 결과: `PASS`

단, test sector 7의 부호 정확도는 60.372%에서 59.598%로 하락했다. 평균 gate는
통과했지만 모든 공간 위치에서 개선됐다는 주장은 성립하지 않는다.

## 해석 한계

V4는 단순 one-ring 통계보다 Laplace–Beltrami 미분·확산이 실제 집단평균 피질에서
추가적인 연관 정보를 가진다는 증거다. 그러나 그 정보가 발생상의 꼬임, 장거리
연결성, 유전 또는 계산 기능에서 왔다는 것을 구별하지 못한다. 특히 heat feature는
이웃의 측정 curvature를 전달하므로, 결과는 우선 `intrinsic multiscale smoothing
helps sulcal-depth prediction`으로 해석해야 한다. 꼬임 가설에는 성장 시계열,
백질 방향장, 개인별 표면 및 독립 데이터셋에서의 외부 검증이 추가로 필요하다.

## V5–V8: 이방성 루프와 우반구 외부 복제

### V5: pial 주방향 이방성 — 실패

접평면에서 one-ring covariance의 최대 고유벡터를 부호 없는 방향장으로 정하고,
edge 정렬도 제곱으로 cotangent 전도도를 바꿨다. 등방성 V4를 기준으로 RMSE는
0.006% 악화했고 부호 정확도는 0.068%p만 증가했으며 RMSE 개선 sector는 1/4였다.
이 방향은 성장축보다 메시 삼각분할 방향을 반영했을 가능성이 크다. 우반구 잠금
시험은 열지 않았다.

### V6: 곡률 경계보존 확산 — sign gate 실패

V5 방향장을 폐기하고 다음 Perona–Malik형 edge conductivity를 사용했다.

\[
w^A_{ij}=w_{ij}\left[\epsilon+(1-\epsilon)
\exp\left(-\left(\frac{|c_i-c_j|}{a\,\mathrm{median}|\delta c|}\right)^2\right)\right]
\]

전도도 하한은 0.1, 고정 contrast scale은 0.5와 2.0이다. 좌반구에서 등방성
V4보다 RMSE가 2.330% 줄고 3/4 sector가 개선됐지만 부호 정확도는 0.0067%p
하락했다. 따라서 전체 V6 gate는 `FAIL`이다.

### V7: 깊이/sign 이중 head — 근소한 실패

V6 특징은 그대로 두고 연속 깊이 ridge와 `{-1,+1}` sign ridge를 분리했다.
RMSE 개선은 2.330%로 유지됐고 sign 이득은 0.1854%p였으나 사전등록 기준
0.2%p에 0.0146%p 미달했다. 임계값을 사후 변경하지 않고 `FAIL`로 보존했다.

### V8: 우반구 연속 깊이 외부 복제 — 통과

좌반구 결과를 본 뒤 주장을 연속 sulcal-depth RMSE로 명시적으로 좁혔다. 이
사후 선택 사실을 공개하고, 그때까지 열지 않은 TemplateFlow 우반구 10k를 새
확증 자료로 잠갔다. 부호 정확도는 출력하지만 gate로 사용하지 않았다.

- 등방성 LB 기준 RMSE: 0.353357
- 경계보존 LB 후보 RMSE: 0.346192
- RMSE 감소: 2.028%
- RMSE 개선 sector: 8/8
- 비게이팅 부호 정확도: 77.648% → 77.985%, +0.337%p
- 결과: `PASS`

이 결과는 경계보존 리만 확산의 **feature family**가 반대 반구에서도 연속 깊이
예측에 추가 정보를 준다는 복제다. 각 우반구 fold는 나머지 우반구 일곱 sector로
계수를 다시 학습했으므로 좌반구에서 학습한 계수 자체의 전이는 아니다. 또한
group-average 좌우 반구는 서로 다른 개인 표본이 아니므로 완전한 독립 코호트
복제로 간주해서는 안 된다.

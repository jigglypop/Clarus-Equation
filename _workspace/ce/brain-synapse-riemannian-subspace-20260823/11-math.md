# 11-math — BA-SRM1 strict 부분공간 수학 감사

Status: COMPLETE

P0 verdict after Revision 1: `PASS`

Claim mapping: `BA-SRM1-C2`, `BA-SRM1-C3`, `BA-SRM1-C4`, `BA-SRM1-C5`.

이 판정은 식의 조건부 정합성만 뜻한다. 실제 데이터의 rank·예측·생물 적합성은
아직 열리지 않았다.

## 1. 최초 초안에서 기각한 식

아래는 최초 초안에 있었지만 감사에서 **폐기되어 활성 식이 아닌 비교량**이다.

$$
\Delta g_{CE}=g-I\qquad\text{[REJECTED; NOT ACTIVE]}
$$

는 일반 affine rechart $z'=Az+b$에서 공변적이지 않았다. $g$는

$$
g'=A^{-T}gA^{-1}
$$

로 변하지만 모든 chart에 새 $I$를 넣으면 일반적으로

$$
g'-I\ne A^{-T}(g-I)A^{-1}.
$$

단위 재척도만으로 “CE 차이”가 생기는 반례이므로 이 정의를 삭제했다. 또한
동일 pulse summary를 factor와 target에 같이 넣는 self-prediction seam도 있었다.
둘 다 outcome 접촉 전에 수정했다.

## 2. strict chart의 무차원성

선택 좌표는

$$
z=\left(
\log\frac{|r_1|}{r_{\rm ref,\chi}},
\log\frac{L_{\rm soma}}{L_{\rm ref}},
\log\frac{R_{\rm in,post}}{R_{\rm ref}},
\log\frac{\tau_{m,\rm post}}{t_{\rm ref}}
\right).
$$

$r_1/r_{\rm ref}$는 V/V, $L/L_{\rm ref}$는 m/m, $R/R_{\rm ref}$는
$\Omega/\Omega$, $\tau/t_{\rm ref}$는 s/s다. 따라서 모든 log 인자는
무차원이다. target 앞 세 성분도 V/V이고 variability 성분은 source algorithm의
log-normalized 무차원 값이다.

0·음수·nonfinite에는 log를 적용할 수 없다. 임의 epsilon은 chart를 바꾸므로
금지하고 결측 receipt로 처리한다. E/I, PSP/PSC, species, clamp mode별 reference를
섞지 않는다.

## 3. response pullback의 정확한 지위

train-only quadratic response map을

$$
y=\mathcal H_2(z)+\epsilon,qquad
\epsilon\sim\mathcal N(0,R_\chi)
$$

로 두고 $J=\partial_z\mathcal H_2$라 하면

$$
g_{\rm resp}(z)=J(z)^TR_\chi^{-1}J(z).
$$

이는 $R$이 $z$와 무관한 Gaussian-location model에서의 Fisher이자 output-space
Mahalanobis metric의 passive pullback이다. 일반 likelihood Fisher로 넓혀 부르지
않는다. $R$을 diagonal로 두는 것은 source fact가 아니라 고정 model choice다.

임의의 $v$에 대해

$$
v^Tg_{\rm resp}v=(Jv)^TR_\chi^{-1}(Jv)\ge0
$$

이므로 PSD다. $R_\chi\succ0$일 때

$$
g_{\rm resp}\succ0
\Longleftrightarrow
\operatorname{rank}J=4.
$$

output이 네 개인 것은 필요조건일 뿐 충분조건이 아니다. target끼리 중복되거나
어떤 입력 방향에 무감하면 rank가 떨어진다. exact SPD는

$$
\sigma_{\min}\!\left(R_\chi^{-1/2}Jg_{\rm ref}^{-1/2}\right)>0
$$

로 확인하고, generalized eigenvalue ratio $10^{-4}$는 수학적 rank 정의와 분리한
practical stability gate다. 실패 시 ridge로 rank를 만들지 않는다.

여기서 $a_2$, $a_{6:8}$, $a_{9:12}^{250\rm ms}$와 $v_{5:8}$는 각각
scalar summary 하나이므로 $y\in\mathbb R^4$, $J\in\mathbb R^{4\times4}$다.
첨자 범위는 벡터 성분 수가 아니라 source가 중앙값/variability를 계산한 pulse
구간을 나타낸다.

## 4. 공변 기준 계량

원 source-locked chart의 train covariance에 한 번만 고정 shrinkage를 적용해

$$
\Sigma_s=\Sigma+10^{-6}\frac{\operatorname{tr}\Sigma}{4}I,
\qquad
g_{\rm ref}=\Sigma_s^{-1}
$$

로 둔다. rechart 뒤 covariance와 shrinkage를 다시 계산하지 않고

$$
g_{\rm ref}'=A^{-T}g_{\rm ref}A^{-1},\qquad
g_{\rm resp}'=A^{-T}g_{\rm resp}A^{-1}
$$

로 transport한다. 이렇게 해야 원 chart에서 쓴 isotropic shrinkage가 일반 affine
변환 아래 새로운 gauge artifact를 만들지 않는다.

비교 tensor는

$$
\Delta g_{\rm resp}=g_{\rm resp}-g_{\rm ref}
$$

이며 CE 동역학 항이 아니다. $Delta F_{CE}=0$이다. 실제 평가는 generalized
eigenvalue, $\operatorname{tr}(g_{\rm ref}^{-1}g_{\rm resp})$, determinant ratio,
line-element ratio처럼 congruence-invariant한 양으로 한다.

## 5. constant metric 반례와 matched control

$g=cI$이면

$$
d_g(z,z')^2=c\|z-z'\|^2
$$

이므로 RBF bandwidth를 $\ell/\sqrt c$로 바꾼 Euclidean kernel과 같다. 더 일반적인
constant SPD $g=L^TL$도 $z'=Lz$라는 선형 rechart의 Euclidean model이다.
따라서 M1이 isotropic M0만 이겨서는 state-dependent geometry 증거가 아니다.

필수 control은 reference Euclidean, diagonal response metric, train-average constant
full metric, direct quadratic response map이다. M1이 constant-full control을 이기지
못하면 “nonconstant response-aware geometry”를 STOP한다.

## 6. graph geodesic의 held-out 정의

train node만 $g_{\rm ref}$ distance로 symmetric-union $k$NN adjacency를 만들고,
각 edge $(a,b)$에

$$
\ell_{ab}=\sqrt{(z_a-z_b)^T
\frac{g_{\rm resp}(z_a)+g_{\rm resp}(z_b)}{2}
(z_a-z_b)}
$$

를 준다. Dijkstra shortest path가 discrete geodesic 근사다. 연결되지 않는 $k$는
무효다.

held-out $z_*$는 $g_{\rm ref}$ 기준 가장 가까운 train node $k$개에만 같은 방식으로
붙인다. test-test edge와 test를 이용한 adjacency 재구성을 금지한다. 이로써
transductive leakage 없이 $d_g(z_*,z_{train})$가 정의된다.

gauge test에서는 node labels와 adjacency를 그대로 transport한다. rechart에서
kNN을 다시 만들면 graph가 변하는 반례가 있으므로 금지한다. 고정 QR·diagonal·
shear·translation suite와 계약 tolerance로 line element, generalized spectrum,
prediction을 확인한다.

## 7. measurement-overlap 반례의 처리

동일 pulse response로 $z$와 $y$를 만들면 $\mathcal H(z)\approx y$가 구성적으로
성립해 $J^TR^{-1}J$와 ELPD가 부풀 수 있다. Allen의 latency·rise·decay는 여러
pulse position을 평균하므로 이 반례에 걸린다. 이를 strict chart에서 제거하고
`SHARED_SUMMARY_DIAGNOSTIC`으로 격리했다.

휴지막 amplitude는 source pipeline의 `previous_pulse_dt>8 s` 조건, target은
pulse 2/6--8/9--12 조건이다. small DB의 row-level event table은 비어 있어
독립 pulse ID를 재구성할 수 없으므로 `PIPELINE_SEPARATED / ROW_LEVEL_UNVERIFIED`
상한을 유지한다.

## 8. directed delay 경계

한 synapse의 latency를 scalar feature로 기록할 수는 있지만 일반적으로
$d_{ij}\ne d_{ji}$다. 따라서

$$
d_{ij}=d_R(i,j)=d_R(j,i)
$$

인 Riemannian propagation distance로 해석할 수 없다. directed order는 BA-CG1의
quasi-metric과 raw-delay/order null에서 따로 검사한다.

## 9. 최종 수학 판정

Revision 1은 잘못된 $g-I$, shared-summary leakage, 불명확한 rank 조건,
held-out query graph와 gauge adjacency 공백을 닫았다. strict 식은 조건부로
잘 정의된다. 이후 실제 데이터에서 다음 중 하나면 Riemannian 승격은 실패한다.

- $R$ 비양정/ill-conditioned;
- observed support에서 $J$ rank loss;
- bootstrap stability cutoff 실패;
- fixed gauge suite 실패;
- variable metric이 constant-full/direct controls를 이기지 못함.

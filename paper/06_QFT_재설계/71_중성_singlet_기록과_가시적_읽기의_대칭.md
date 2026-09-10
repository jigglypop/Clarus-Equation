# 71. 중성 singlet 기록과 가시적 읽기의 대칭

## 71.1 계산 전 등록: CE-UR4

**[후보 공리]** 70장의 CE-UR3에서 기록 adjoint의 질량 이동만 바꾼다.
약한 triplet을 가볍게 하는 계수 6 대신 singlet을 가볍게 하는 계수 2를 쓴다.
관측값에 연속 매개변수를 맞추는 수정이 아니라 별도 질량 공리의 분기다.
정준 전역 N=1 SU(5) 작용, 물질 내용, 공통 경계 결합과 표기법은 70장을 따른다.

$$
W_X=\sum_{r,s}K_{rs}\operatorname{Tr}(X_rX_s)
+2\lambda_R V\sum_r\operatorname{Tr}X_r^2
+2\lambda_R\sum_r\operatorname{Tr}(\Sigma X_r^2),
\qquad K=m_0(5I+\tfrac\pi2\sigma_x).
$$

이외의 작용 계수는 바꾸지 않는다. 특히 기록별 Yukawa나 관측별 보정을 넣지 않는다.
가벼운 좌표는 $X_r=S_rT_Y$이며 나머지 질량은 같은 Hessian으로 구한다.
아래 결과 절을 쓰기 전에 이 절의 SHA-256을 재현 JSON에 고정한다.

1. 전체 164 실수 scalar, 106 Weyl, 24 vector 질량을 직접 구성하고 표현별
   스펙트럼과 독립 대조한다. $m_0=0$, $X_1=xT_Y/2$, $X_2=ixT_Y/2$의
   $F=D=0$과 초대칭 multiplet 질량 상쇄를 검사한다. 자유 기록의 KG 정규화,
   에너지 및 CP 사상은 같은 $K$로 검사한다. 기본 부동소수 허용치는 $10^{-8}$이다.
2. 70장의 동결 PDG 입력과 같은 MS/DR 변환, 낮은 soft 질량 구간
   $[M_Z,100M_Z]$, $m_-\ge10^5M_Z$, $m_+/M_X\le10^{-3}$를 유지한다.
   무거운 질량의 상대 soft 변화는 $10^{-3}$ 이하이며 좌표별 잔여 정합 상계가
   $0.05$보다 작은지 먼저 확인한다. singlet의 게이지 지수가 0인 질량 가족을
   공통 결합과 정합 길이를 제거한 식 및 독립 선형계획으로 검사한다.
   가족이 배제되지 않아도 매개변수 동결·무피팅 예측·공동 RMSE 성공으로 판정하지 않는다.
3. 게이지 가족이 생존하면 무거운 chiral 장을 tree 수준에서 소거하여 singlet과
   가시적 Higgs 사이의 선도 superpotential 및 Kähler 항을 유도한다.
   Goldstone 영 모드를 역행렬에 넣지 않고 그 source가 0인지 확인한다.
   유효작용은 무거운 질량보다 작은 에너지·장·보조장 범위에만 적용한다.
4. 관측 후보 $R=\operatorname{diag}(0,1)$가 두 기록 복사본의 절대 표지를 뜻한다고
   가정한다. 기록 교환 $X_1\leftrightarrow X_2$에 불변인 장치 초기 상태,
   가시적 관측량과 전체 시간진화를 사용하며 별도의 비대칭 기록 참조를 공급하지 않는다.
   전체 작용의 교환 대칭을 검사하고, 두 표지의 가시적 출력이 동일한지 증명한다.
   동일하다면 이 장치 조건에서 절대 표지와 물리적 관측 기록의 동일시를 기각한다.
   비대칭 참조가 있는 유한 모형을 양성 대조로 써서 반례의 범위를 확인하되,
   그 대조 장치를 이 작용에서 유도한 실물 관측 장치로 주장하지 않는다.

**[판정 한계]** 공급한 초대칭·게이지·시공간 구조가 기초 공리로부터 유도되었다고
주장하지 않는다. soft 깨짐의 기원, electroweak 진공, 실제 관측 기록과 중력·우주론
예측은 별도 미완성 조건이다. scientific_success는 false, full_joint_rmse는 null로 둔다.
읽기 반례가 나오면 그 조건을 보존하고 관계적 표지 또는 참조 준비를 다음 분기로 남긴다.

## 71.2 검산 결과

**[조건부 판정]** 중성 기록의 정확 초대칭 한계는 유지된다. 이번 한 루프 정합
가족은 배제되지 않지만, 가족의 넓은 허용 범위를 동결된 예측으로 볼 수는 없다.
같은 작용에서 Higgs와의 결합을 얻어도 교환 대칭을 유지하는 장치에는 두 복사본의
절대 표지가 보이지 않는다. 따라서 이 장치 조건에서 $R$를 실제 읽기 기록으로
동일시하는 주장을 기각한다. CE-UR4의 전체 장론을 기각하거나 OBJ-02를 달성한 것은 아니다.

재현 파일은 [ce_singlet_record_observability.py](../../verify/ce_singlet_record_observability.py)와
[계산 JSON](../../verify/ce_singlet_record_observability.json)이다.
사전등록 해시는 858befdec3b167395038147f74681ee9d6ee8ef1ecd16b1a732d84dd404240cf이다.
기존 [70장](70_초대칭_기록의_보호와_전체_게이지_정합.md)의 작용 구성 코드를
별도 파일에서 호출하고 기록 질량 블록에만 $-4I$를 더한다. 이전 분기의 결과는 수정하지 않는다.
모든 질량 검산의 단위는 $V=g_5=\lambda_\Sigma=\lambda_R=\lambda_H=1$이다.

### 71.2.1 같은 작용의 질량과 영 질량 조건

70장의 $J_\Sigma X=P_0\{\Sigma,X\}$ 고유값을 사용하면

$$
\mathcal M_X=K+\lambda_R(J_\Sigma+2V),\qquad
m_\pm=m_0(5\pm\pi/2),\qquad M_X=5\sqrt2g_5V.
$$

| 기록 표현 | 복사본별 차원 | 부호를 보존한 holomorphic 질량 |
|---|---:|---|
| $(8,1)_0$ | 8 | $m_\pm+6g_5V$ |
| $(1,3)_0$ | 3 | $m_\pm-4g_5V$ |
| $(1,1)_0$ | 1 | $m_\pm$ |
| $(3,2)_{-5/6}+(\bar3,2)_{5/6}$ | 12 | $m_\pm+g_5V$ |

물리적 질량은 절댓값이다. 음의 holomorphic 질량을 음의 scalar 질량제곱으로
해석하지 않는다. $m_0=0$ 진공의 전체 질량제곱 중복도는 다음과 같다.

| $M^2/V^2$ | 실수 scalar | Weyl | vector |
|---:|---:|---:|---:|
| 0 | 24 | 18 | 12 |
| 1 | 50 | 25 | 0 |
| 16 | 12 | 6 | 0 |
| 25 | 34 | 17 | 0 |
| 36 | 32 | 16 | 0 |
| 50 | 12 | 24 | 12 |
| 합계 | 164 | 106 | 24 |

표의 영 scalar에는 gauge Goldstone 좌표가 포함된다. 3세대 물질의 45 chiral
좌표는 $H=\bar H=0$ 및 직접 기록 Yukawa가 없는 검사 배경에서 질량과 기록
배경 의존성이 0이므로 양변의 질량 0 multiplet로 생략한다. 이는 물질을
게이지 beta 함수에서 생략한다는 뜻이 아니다.

독립 표현별 질량식과 직접 행렬의 비영 $m_0=0.003$ 대조 오차는
$1.78\times10^{-15}$이다. $T_Y$는 모든 비깨진 게이지 생성자와 교환하므로
두 가벼운 chiral 기록은 SM singlet이고 그 게이지 지수는 0이다.

**[조건부 유도]** $m_0=0$에서

$$
X_1=\frac{x}{2}T_Y,\qquad X_2=\frac{ix}{2}T_Y
$$

는 정확히 $F=D=0$이다. 두 $X_r$는 각각 normal이고 $\Sigma_0$와 교환한다.
따라서 $D=0$이다. 기록 $F$는 singlet 영 질량 때문에 0이고,
$F_\Sigma$의 source는 $X_1^2+X_2^2=0$ 때문에 사라진다. 전체 정준 실수
방향의 길이 제곱은 $2\sum_r|dS_r/dx|^2=1$이다.
이 배경에서 비영 질량의 실수 scalar 1개와 vector 편광 3개를 세면 Weyl
자유도 2개와 같은 질량별 중복도를 가져 초대칭 regulator의 한 루프
supertrace가 상쇄된다. $x=0,0.07,0.3$에서 가중 스펙트럼의 최대 차이는
$7.82\times10^{-14}$, $F$ 오차는 $5.16\times10^{-15}$이다.
전체 실수 좌표의 세 독립 방향에서 potential gradient 유한차분과 Hessian을
대조한 오차는 $4.21\times10^{-13}$이다.

70장의 초대칭 superpotential RG 법칙으로 새 조합

$$
r=\mu_R-2\lambda_R\frac{m_\Sigma}{\lambda_\Sigma},\qquad
\beta_r=2\gamma_Xr
$$

를 얻는다. 계수 2를 넣은 기호 미분으로 검산했다. 같은 질량 독립 cubic
tensor는 $16\pi^2\gamma_\Sigma=-16/5$와 $16\pi^2\gamma_X=-29/5$를 준다.
이는 공급한 정확 초대칭 이론의 섭동적 보호이며 soft 질량 보호나
$\lambda_\Sigma=\lambda_R=\lambda_H=g_5$의 RG 불변성을 뜻하지 않는다.
일반 superpotential RG 법칙의 출처는
[Martin, SUSY primer §6.5](https://arxiv.org/abs/hep-ph/9709356)이다.

자유 singlet의 양의 주파수 부분공간에서 $K$로 시간진화하면
$p_R(t)=\sin^2(\pi t/2)$이다. KG 정규화와 자유 에너지 $5m_0$, 형식적
Kraus $P_aU$의 trace 보존·완전양성을 검사했다. 이는 관측 장치가 에너지
교환까지 수행한다는 증명이 아니다. 실제 $R$ 읽기의 가능성은 71.4에서 판정한다.

## 71.3 같은 동결 입력의 한 루프 게이지 검사

**[수치 가족 검사]** 자료, 공분산 미공급 상태, 비정규화 $g_Y$, MS/DR 변환과
soft 구간은 [동결 입력](../../verify/ce_gauge_matching_inputs.json) 및 70장과 같다.
중성 기록의 가벼운 질량 로그는 게이지 정합에 들어가지 않는다.

$$
k=(1,1,5/3),\quad b=(-3,1,11),\quad
w=(1,-12/7,3/7),\quad w\cdot k=w\cdot b=0.
$$

$$
x^{\rm MS}=ka+b\ell-\sum_jd_jh_j+\Delta
+\frac{(3,2,0)}{12\pi}+\epsilon,\qquad
a=\alpha_5^{-1},\quad
\ell=\frac{\ln(M_X/M_Z)}{2\pi},\quad
h_j=\frac{\ln(m_j/M_Z)}{2\pi}.
$$

낮은 각 superpartner 질량은 독립으로
$0\le h_j\le\ln100/(2\pi)$이며 bino의 지수는 0이다.
이는 상관된 soft 작용을 아직 구성하지 않은 확대 가족이다.
$b_5=7$이며 무거운 multiplet까지 더하면 $7k$가 되는 정합 척도
beta 항등식은 세 좌표 모두 성립한다.
MS/DR 변환은
[Martin–Vaughn 식 (2.4)](https://arxiv.org/abs/hep-ph/9308222)을 따른다.

$$
\begin{aligned}
d_8&=(3,0,0),& d_3&=(0,2,0),&
d_B&=(2,3,25/3),\\
d_\Sigma&=(3,2,0),&d_H&=(1,0,2/3),&
d_8+d_3+d_B&=5k .
\end{aligned}
$$

작은 $m_\pm/M_X$를 잔여항으로 보낸 고정 threshold는

$$
\Delta=\frac{1}{2\pi}\left[
(d_\Sigma+d_H)\ln\sqrt2+
2d_8\ln\frac{\sqrt{50}}6+
2d_3\ln\frac{\sqrt{50}}4+
2d_B\ln\sqrt{50}\right].
$$

물리적 vector·Goldstone·gaugino·scalar의 절대 beta 계수를 포함한 무거운
soft 변화의 계수 합은 $(24,27,59)$이다. $\eta=10^{-3}$과
$|\ln(1+e)|\le |e|/(1-|e|)$를 써서 soft 변화와 작은 기록 질량의 생략 오차를
각 좌표에서

$$
|\epsilon_i|
\le\frac16\left[(24,27,59)_i\frac{\eta}{1-\eta}
+10k_i\frac{3}{400}\right]
=\left(\frac{1319}{79920},\frac{151}{8880},
\frac{1471}{47952}\right)_i<0.05
$$

로 제한한다. 여기서 $\sqrt{50}<7.1$과
$0.0071/(1-0.0071)<3/400$, $\pi>3$을 사용했다.
이후 $\epsilon_i$를 독립 $\pm0.05$로 더 확대한다. 두 루프 보정을
이 상계에 포함했다고 주장하지 않는다.

공통 $a,\ell$를 제거한 상수는

$$
C=w\cdot\left[\Delta+\frac{(3,2,0)}{12\pi}\right]
=\frac1\pi\left(\frac{57}{14}\ln2-3\ln3-\frac1{28}\right).
$$

$A=\alpha_{\rm em}^{-1}$, $s=\sin^2\theta_W$에 대해
$x_2=As$, $x_Y=A(1-s)$이다. 같은 soft 종별 표에서

$$
\sum_j\min(-w\cdot d_j,0)=-\frac{29}{7},\qquad
\sum_j\max(-w\cdot d_j,0)=\frac{11}{2}.
$$

따라서 $x_3=(15s-3)A/7+C-\sum_jw\cdot d_jh_j+w\cdot\epsilon$를
동결 EW 좌표 구간의 양 끝에서 평가하면

$$
5.168267365916\le x_3\le12.619150126539,\qquad
0.079244639296\le\alpha_s\le0.193488441909 .
$$

동결 관측 좌표 구간 $0.1153\le\alpha_s\le0.1207$은 이 범위 안에 있다.
이는 공분산을 가진 신뢰구간이나 물리적 mass spectrum의 존재 증명이 아니다.
전체 세 좌표의 선형계획에서

$$
a\ge0,\qquad
\ell\ge\frac{\ln[10^8(m_+/m_-)]}{2\pi}
$$

까지 넣어 얻은 양 끝값은 제거식과 $1.43\times10^{-14}$ 이내로 일치한다.
LP 양 끝은 구간 상·하계의 검산용이며 관측값에 맞춘 매개변수 선택이 아니다.

**[판정]** 이 확대된 한 루프 가족에서는 이전 triplet 분기의 배제 근거가 사라진다.
동결된 soft 작용, electroweak 진공과 예측값을 얻지 않았으므로 게이지 통합의
경험적 성공으로 승격하지 않는다. 후보별 재피팅은 수행하지 않았다.
이미 공개된 관측과 이전 반례를 알고 연 탐색 분기이므로, 같은 자료에서의
미배제를 독립 holdout 예측 검증으로 세지 않는다. 새로운 구조 가정의 선택 비용도
미완성 공동 RMSE에서 누락한 채 성공 점수를 보고하지 않는다.

## 71.4 같은 작용의 portal과 관측 표지 반례

### 71.4.1 무거운 chiral 장의 tree 소거

**[조건부 유도]** 가벼운 singlet과 두 Higgs doublet에 대해

$$
Q=S_1^2+S_2^2,\qquad P=\bar H_{\rm weak}H_{\rm weak}
$$

로 둔다. 이는 holomorphic 식이며 $Q$에 복소켤레를 붙이지 않는다.
무거운 $\Sigma$ singlet의 holomorphic 질량은 $-\lambda_\Sigma V$이고
weak triplet은 $-5\lambda_\Sigma V$이다. $2\operatorname{Tr}T_Y^2=1$에서
source는

$$
J_s=\frac{\lambda_RQ+3\lambda_HP}{2\sqrt{15}},\qquad
J_a=\lambda_H\bar H_{\rm weak}\frac{\sigma_a}{2}H_{\rm weak},\qquad
\sum_aJ_a^2=\frac{\lambda_H^2P^2}{4}.
$$

무거운 정상 좌표 $Z$의 $W=\tfrac12Z^{\mathsf T}MZ+J^{\mathsf T}Z+\cdots$에서
$Z=-M^{-1}J+\cdots$를 대입하여 $-\tfrac12J^{\mathsf T}M^{-1}J$를 얻는다.
그 결과

$$
W_{\rm eff}
=\frac12S^{\mathsf T}KS+
\frac{\lambda_R^2}{120\lambda_\Sigma V}Q^2+
\frac{\lambda_R\lambda_H}{20\lambda_\Sigma V}QP+
\frac{\lambda_H^2}{10\lambda_\Sigma V}P^2+\cdots .
$$

특히 공통 경계 결합에서 portal은 $g_5QP/(20V)$이다.
각 chiral 장의 질량 차원 1로 세면 $Q,P$는 차원 2,
$W_{\rm eff}$의 모든 표시 항은 차원 3이다. Higgs 사차항의 계수에는 singlet과
triplet의 기여가 모두 포함된다.

정준 Kähler 항에 무거운 해를 대입한 같은 차수의 보정은

$$
\delta\mathcal K=
\frac{|J_s|^2}{|\lambda_\Sigma V|^2}
+\sum_a\frac{|J_a|^2}{25|\lambda_\Sigma V|^2}.
$$

이는 차원 2이며 정준 metric에 양의 Gram 행렬을 더한다.
깨진 gauge 방향의 light current와 Goldstone source는 이 배경에서 0이다.
Goldstone 영 모드를 역행렬에 넣지 않았다. 물질을 포함하는 전체 저에너지
작용을 이 표시 항들로 대체한 것은 아니다.

기호 Pauli 항등식 및 원래 82차원 cubic tensor의 복소 입력 6개로 얻은
소거식이 $7.05\times10^{-19}$ 이내에서 일치하고 다른 heavy source는 0이다.
tree chiral 소거는 저에너지·작은 mixing·작은 보조장 영역의 선도 근사이다.
일반 소거 규칙과 적용 조건은
[Brizi–Gómez-Reino–Scrucca 식 (2.3)–(2.5), (2.21)](https://arxiv.org/abs/0904.0370)을 따른다.
위 portal을 실제 검출률이나 증폭 장치로 계산한 것은 아니다.

### 71.4.2 가시적 절대 표지를 구별할 수 없는 조건

**[정리, 조건부]** 모든 기록 모드를 교환하는 $\mathcal P$를 잡고, 가시적
장에는 항등으로 작용하게 한다. 위 전체 작용은

$$
\mathcal P:X_1\leftrightarrow X_2,\qquad
\mathcal P_{\rm one}=\sigma_x
$$

에 불변이다. $K$의 혼합항도 불변이며 $m_0=0$일 때만 성립하는 대칭이 아니다.
질량·cubic·gauge representation·진공을 실제 좌표 순열로 검사한 오차는 0이다.
표준 물질 Yukawa에는 복사본 지표가 없어 이 대칭을 깨지 않는다.

전체 시간진화가 $[U,\mathcal P]=0$이고, 보조 기록 모드 및 장치의 초기 상태
$\eta$가 교환에 불변이라고 가정한다. 논리 기록의
$\rho_1=\sigma_x\rho_0\sigma_x$에 대해 가시적 출력은

$$
\begin{aligned}
\mathcal C(\rho_1)
&=\operatorname{Tr}_{\rm hidden}
U(\rho_1\otimes\eta)U^\dagger\\
&=\operatorname{Tr}_{\rm hidden}
\mathcal P U(\rho_0\otimes\eta)U^\dagger\mathcal P^\dagger\\
&=\mathcal C(\rho_0).
\end{aligned}
$$

마지막 등호는 hidden 부분추적의 unitary 불변성이다. 입자 수가 변하거나 다른
기록 모드로 전이해도 $\mathcal P$가 모든 hidden 기록에 작용하면 같은 증명이다.
이는 대칭을 보존하는 양자 Hamiltonian에 관한 정리이며, 모든 비섭동·중력
효과가 이 대칭을 보존한다고 별도 증명한 것은 아니다.

가시적 POVM을 뒤로 당긴 논리 효과 $E$는 $[E,\sigma_x]=0$이므로

$$
E=aI+b\sigma_x,\qquad 0\le a\pm b\le1 .
$$

따라서 $|0\rangle,|1\rangle$의 확률은 모두 $a$이고 trace distance는 0이다.
동일 사전확률의 최소 판별 오류는 $1/2$이다. 또한 $R=\operatorname{diag}(0,1)$에 대해

$$
\|E-R\|_{\rm op}\ge\max(|a|,|a-1|)\ge\frac12,
$$

이고 $E=I/2$가 하계를 달성한다. 접근 가능한 측정 전체에 대한 하계이므로
portal 계수를 조정해 이 반례를 피할 수 없다.
대칭과 참조 상태의 일반적 구분은
[Bartlett–Rudolph–Spekkens, quantum reference frames](https://arxiv.org/abs/quant-ph/0610030)를
참고할 수 있으며, 이 복사본 교환의 부분추적 증명과 적용 범위는 위에 명시했다.

**[범위]** 이 정리는 절대 복사본 표지를 불변인 가시적 장치로 읽는 경우에 적용한다.
모든 hidden 상태나 질량 고유상태의 구별 불가능성을 뜻하지 않는다.
초기 $|0\rangle$ 자체는 비대칭 자원이므로 불변 진공과 불변 조작만으로
무조건 준비할 수 없다. 이런 초기 상태를 경계 조건으로 허용하더라도,
별도 참조가 없는 가시적 출력은 교환된 두 준비를 구별하지 못한다.

### 71.4.3 관계적 참조의 양성 대조와 다음 분기

두 가지 유한 unitary는 위 정리의 범위를 독립 검산한다. 첫째,

$$
U=P_+\otimes I_E+P_-\otimes X_E,\qquad
P_\pm=(I\pm\sigma_x)/2
$$

는 교환 대칭을 지킨다. pointer를 $|0\rangle_E$로 준비하면 복사본
$|0\rangle,|1\rangle$은 같은 $I_E/2$를 주지만 질량 기저 $|+\rangle,|-\rangle$는
서로 직교하는 pointer 출력을 준다. 두 trace distance는 각각 0과 1이다.

둘째, 참조 $A$를 추가하여

$$
R_{\rm rel}=\frac{I-Z_SZ_A}{2},\qquad
U_{\rm rel}=(I-R_{\rm rel})\otimes I_E+R_{\rm rel}\otimes X_E
$$

로 두면 $[U_{\rm rel},X_SX_A]=0$이다.
참조 $|0\rangle_A$를 공급할 때 pointer는 두 $S$ 표지를 완전히 구별하지만,
참조를 불변 $I_A/2$로 바꾸면 두 출력은 같다. 각 trace distance 1과 0,
unitary 및 교환 오차 0을 직접 검사했다.

이 대조 unitary를 CE-UR4에서 유도했다고 주장하지 않는다. 참조 준비,
이 상호작용의 실현, 장치 에너지와 기록 지속성은 미완성이다.
**[다음 후보]** OBJ-02의 표지를 절대 복사본 $R$ 대신 관계적
$R_{\rm rel}$로 정의하는 별도 가정을 시험할 수 있다. 이때 같은 작용에서
참조와 pointer를 준비·결합·유지하는 과정을 먼저 구성해야 한다.

## 71.5 연구 목표와의 거리

| 항목 | 이번 판정 |
|---|---|
| 정의·작용·버전·반증 조건 사전 고정 | 71.1 해시 및 독립 파일에 기록 |
| 정확 초대칭 singlet 영 질량과 자유 정규화 | 조건부 유도와 행렬 검산 |
| 같은 자료의 전체 게이지 정합 | 확대된 한 루프 가족에서 배제되지 않음 |
| 같은 작용의 가시적 portal | tree 유효작용의 좁은 유도 |
| 불변 장치의 절대 복사본 읽기 | 조건부 기각, 오류율 하계 $1/2$ |
| 비동기 확정·관계적 실제 장치 | 미완성 |
| 양자·중력·우주론 공통 예측 | 미완성 |
| VAL-01 공동 RMSE | 계산 준비 미완성, 값 없음 |
| VAL-02 관측별·후보별 재피팅 | 수행하지 않음; soft 예측은 아직 미동결 |

게이지 구간의 겹침과 형식적 CP 사상은 자연에 대한 물리 증명이나 전체 목표의
달성이 아니다. scientific_success=false와 full_joint_rmse=null을 유지한다.

집중 재현 스크립트, AST, 원본 코드·입력 해시, 70장과 71장의 사전등록 해시,
수식 정규화와 관련 네 문서의 상대 링크 117개를 검사했다. 공용 실행 코드와
기존 테스트 계약은 바꾸지 않았으며 전체 회귀를 실행한 결과로 보고하지 않는다.

후속 [72장](72_기록_쌍의_산란과_관계적_관측량.md)은 같은 portal에서 Higgsino 쌍의
tree 산란을 계산한다. 유한 에너지 packet의 위상 정보가 확률에 남지만, 서로 다른
관계적 표지에 소멸하지 않는 두 상태가 있어 소멸만으로 전체 표지를 읽는 사상을 기각한다.

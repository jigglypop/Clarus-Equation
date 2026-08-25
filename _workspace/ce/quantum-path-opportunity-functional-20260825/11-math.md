# 11-math — 비선택 경로 기회비용의 타입별 유도

Status: COMPLETE

## 1. instrument와 비선택 상태

quantum instrument $\{\mathcal I_a\}$에 대해

$$
p_a=\operatorname{tr}\mathcal I_a(\rho),
\qquad
\widetilde\rho_a=\mathcal I_a(\rho),
\qquad
\rho_a=\widetilde\rho_a/p_a
\tag{1}
$$

로 둔다. 선택 outcome이 $o$이면 비선택 집합 $U$의 총확률과 조건부 상태는

$$
p_U=1-p_o,
\qquad
\rho_U=\frac1{p_U}\sum_{a\in U}\widetilde\rho_a
\quad(0<p_U<1)
\tag{2}
$$

다. 식 (2)는 조건부 상태의 정의이지 선택 branch에 추가되는 energy source가
아니다.

## 2. 정보 기회비용의 최소 정의

유한한 decoherent/coarse-grained outcome에서 각 비선택 outcome의 surprisal은

$$
I_a=-\ln p_a
\tag{3}
$$

다. $I_a$는 무차원이며 $p_a=0$인 개별 항은 $0\ln0:=0$의 극한으로만 가중합에
넣는다. 비선택 경로의 총 정보 기회비용 후보를

$$
\boxed{
C_I(o):=-\sum_{a\in U}p_a\ln p_a
}
\tag{4}
$$

로 정의한다. $q_a=p_a/p_U$이면

$$
C_I(o)=p_U[-\ln p_U+H(q)],
\qquad
H(q)=-\sum_{a\in U}q_a\ln q_a.
\tag{5}
$$

따라서 식 (4)는 “비선택 총질량”과 “그 내부 다양성”을 함께 기록한다.
$p_U\to0$이면 $C_I\to0$이므로 비선택 경로가 사라지는 극한도 연속이다.
반면 $-\ln p_U$만 쓰면 $p_U\to0$에서 발산하므로 aggregate event의 surprisal로는
쓸 수 있어도 기회비용 총량의 중심 정의로는 부적절하다. $H(q)$만 쓰면 두 outcome
문제에서 $U$가 singleton이 되어 항상 0이므로 역시 총량을 잃는다.

기존 carrier measure는 정보 가중 measure로

$$
\boxed{
\mu_C(B)=\int_{\Gamma_{\rm ns}}
w(\gamma)c(\gamma)\mathbf1_B(F(\gamma))
\nu_{\rm ns}(d\gamma),
\qquad c(\gamma)=-\ln p_\gamma
}
\tag{6}
$$

처럼 장식할 수 있다. $w,c$가 무차원이면 $\mu_C$는 원래 $\mu_F$와 같은
weighted-count 타입이며 energy density가 아니다.

연속 path에는 개별 $p_\gamma$가 일반적으로 0이므로 식 (6)을 그대로 쓰지
않는다. 유한 coarse-graining을 고정하거나 reference measure $r$에 대한
Radon--Nikodym derivative로

$$
D(q\|r)=\int\ln\!\left(\frac{dq}{dr}\right)dq
\tag{7}
$$

를 써야 한다. $q\not\ll r$이면 $D=\infty$이고, 값은 reference와 instrument,
coarse-graining에 의존한다.

## 3. 정보만으로 energy를 만들 수 없는 차원 반례

$p$, $C_I$, $H$와 $D$의 차원 벡터는 모두

$$
[p]=[C_I]=[H]=[D]=1.
$$

무차원량들의 대수적 조합만으로 에너지 차원 $M L^2T^{-2}$를 만들 수 없다.
따라서

$$
\text{probability/entropy alone}
\not\Rightarrow
\text{energy}
\tag{8}
$$

는 완전한 차원 반례다. 에너지로 환산하려면 적어도 $E_*$, $k_BT$ 또는
$\hbar/\tau_*$ 같은 독립 scale이 필요하다.

$$
E_C=E_*C_I,
\qquad
E_C=k_BT C_I,
\qquad
E_C=\frac{\hbar}{\tau_*}C_I.
\tag{9}
$$

식 (9)는 차원이 맞지만 $E_*$, $T$, $\tau_*$의 물리적 선택을 유도하지 않는다.
차원 정합은 energy ontology의 증명이 아니다.

## 4. 열역학적으로 energy 차원을 얻는 조건

dimensionless von Neumann entropy를

$$
S(\rho)=-\operatorname{tr}(\rho\ln\rho)
$$

라 하고, Hamiltonian $H$, bath temperature $T>0$와

$$
\gamma_T=\frac{e^{-H/(k_BT)}}{Z},
\qquad
Z=\operatorname{tr}e^{-H/(k_BT)}
\tag{10}
$$

를 지정한다. exponential 인자 $H/(k_BT)$는 무차원이다. 비평형 자유에너지는

$$
F_T(\rho)=\operatorname{tr}(\rho H)-k_BT S(\rho)
\tag{11}
$$

이고 직접 계산하면

$$
\boxed{
F_T(\rho)-F_T(\gamma_T)
=k_BT D(\rho\|\gamma_T)
}
\tag{12}
$$

다. 따라서 비선택 상태가 조건부로 실제 준비되고 thermal operation의 resource로
사용될 때

$$
E_{\rm opp}^{\rm cond}
:=k_BT D(\rho_U\|\gamma_T)
\tag{13}
$$

는 energy 차원의 nonequilibrium free-energy excess다. 전체 trial당 평균
가중치를 원하면 별도 선택으로

$$
\overline E_{\rm opp}=p_U E_{\rm opp}^{\rm cond}
\tag{14}
$$

를 쓸 수 있다. 식 (13)--(14)는 $H,T,\gamma_T$, 준비·회수 protocol과 허용
operation class에 의존한다. 비선택 가능성이 선택 branch에 저장한 에너지라는
뜻이 아니다.

$k_BT[-\ln p]$도 energy 차원은 갖지만, Landauer형 해석에는 물리 memory와
reset/measurement protocol이 필요하다. 단순히 outcome이 실현되지 않았다는
사실만으로 그 열이 발생하거나 저장되지는 않는다.

## 5. 반사실적 energy regret

각 outcome에 별도 energy value $E_a$를 지정하면

$$
C_E(o)=\sum_{a\in U}p_a(E_a-E_o)
\tag{15}
$$

를 계산할 수 있다. 식 (15)는 energy 차원이지만 음수가 될 수 있다. 경제학적
“최선의 포기 대안”을 흉내 내려면

$$
C_E^+(o)=\sum_{a\in U}p_a[E_a-E_o]_+
\quad\text{or}\quad
C_E^{\max}(o)=[\sup_{a\in U}E_a-E_o]_+
\tag{16}
$$

를 추가로 선언해야 한다. 그러나 물리에서 높은 energy가 더 높은 value라는 법칙은
없으므로 식 (16)의 양의 부분은 효용 선택이지 보존법칙이 아니다.

## 6. path integral과 effective action

환경 자유도 $\xi$를 closed-time path에서 적분하면

$$
e^{iS_{\rm IF}[q_+,q_-]/\hbar}
=\int\mathcal D\xi_+\mathcal D\xi_-
e^{i(S[q_+,\xi_+]-S[q_-,\xi_-])/\hbar}
\tag{17}
$$

인 influence action을 정의할 수 있다. 이것은 환경의 기억·dissipation·noise를
system의 유효동역학에 남기는 정식 경로다. 그러나 Lorentzian integral은
일반적으로 양의 probability sum이 아니고 $S_{\rm IF}$는 복소수일 수 있으므로,
그 자체를 양의 opportunity energy로 읽지 않는다.

Euclidean ratio에서

$$
\Gamma_{\rm ns}=-\hbar\ln(Z_{\rm ns}/Z_{\rm ref})
\tag{18}
$$

는 log 인자가 무차원일 때 action 차원을 갖는다. energy가 되려면
$\Gamma_{\rm ns}/\tau_*$ 또는 thermal identity
$F=-k_BT\ln Z$처럼 추가 시간·온도 구조가 필요하다. $Z_{\rm ref}$는 additive
normalization을 고정하며, regulator와 reference가 없으면 continuum determinant나
path entropy가 발산할 수 있다.

## 7. 중력원이 되기 위한 별도 bridge

정보 기회비용 $C$는 scalar 하나일 뿐 pressure, momentum flux와 anisotropic
stress를 정하지 않는다. 실제 중력 source 후보는 독립 공리로 covariant action을

$$
S_{\rm opp}[g,\chi;C]
=-\int d^4x\sqrt{-g}\,
V_{\rm opp}(C,\chi,\nabla\chi;\epsilon_*)
\tag{19}
$$

처럼 선언한 뒤

$$
\boxed{
T_{\mu\nu}^{\rm opp}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}
}
\tag{20}
$$

로 정의해야 한다. $\epsilon_*$는 독립 energy-density scale이다.

가장 단순한 조건부 예로 $V_{\rm opp}=\epsilon_*f(C)$이고 $C$가 metric과
무관한 상수이면

$$
T_{\mu\nu}^{\rm opp}=-\epsilon_*f(C)g_{\mu\nu},
\qquad w=-1.
\tag{21}
$$

를 얻는다. 이것은 dark-energy-like readout이지만 $\epsilon_*$와 $f$를 새로
가정한 결과다. $C=C(x)$를 외부 함수로 고정하면

$$
\nabla^\mu T_{\mu\nu}^{\rm opp}
=-\partial_\nu[\epsilon_*f(C)]
\tag{22}
$$

이므로 단독 보존되지 않는다. $C$ 또는 $\chi$의 운동방정식과 apparatus,
environment, reservoir stress를 포함한 full diffeomorphism-invariant action이
있어야 total conservation을 닫을 수 있다.

## 8. 두 outcome 수치 예

$p=(0.8,0.2)$, $o=0$, $U=\{1\}$이면

$$
H(p)=-0.8\ln0.8-0.2\ln0.2
=0.5004024235381879,
$$

$$
-\ln p_U=1.6094379124341003,
$$

$$
C_I(o)=-0.2\ln0.2
=0.3218875824868201.
\tag{23}
$$

$q=(1)$이므로 $H(q)=0$이다. full two-outcome delta
$\widehat q=(0,1)$와 uniform reference $r=(1/2,1/2)$를 비교하면
$D(\widehat q\|r)=\ln2=0.6931471805599453$이지만, reference를 바꾸면 값도
바뀐다. $E_0=0$, $E_1=\Delta$이면 식 (15)는 $0.2\Delta$다. $\Delta$ 또는
다른 energy scale을 주지 않으면 앞의 세 정보량은 모두 nat 단위의 무차원수다.

## 9. 수학 판정

살아남는 최소 정의는

$$
\boxed{
\text{nonselected history}
\xrightarrow{\ C_I\ }
\text{dimensionless opportunity measure }\mu_C
}
\tag{24}
$$

다. 지정된 thermodynamic setup에서만 식 (13)의 free-energy excess로 올라갈 수
있고, 중력원은 식 (19)--(20)의 독립 action bridge를 통과해야 한다.

“에너지 없는 에너지”를 **실제 energy가 아닌 정보적 shadow price**라는 뜻으로
쓰면 일관된다. 외부 scale 없이 실제 energy라고 하거나 비선택 경로가 자동으로
중력원이라고 하면 차원 반례와 conditional-state 반례 때문에 성립하지 않는다.

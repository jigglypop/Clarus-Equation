# B2 완성 모형: 에너지 가중 생존율과 바리온 밀도

## 1. 적용 범위

이 문서는 경로의 **개수 확률**을 우주 바리온 밀도와 동일시하지 않는다.
대신 공변 stress tensor로 정의한 **에너지 가중 측도**에 CE 고정점 조건을
부과하는 최소 폐쇄 모형을 제시한다. 이 모형 안에서

$$
x=e^{-(1-x)D_{\rm eff}}
\quad\Longrightarrow\quad
\Omega_b(a_0)=x
$$

가 정리로 성립한다. 여기서 $a_0=1$은 관측 기준 초곡면이다. 이 등식은
초기우주의 경로 개수나 바리온-광자 비를 곧바로 동일시한 결과가 아니다.

## 2. 공변 에너지 측도

공간적으로 평탄한 FLRW 배경의 관측 초곡면을 $\Sigma_0$라 하고, 미래방향
단위 법선을 $n^\mu$, 공변 시간 병진 벡터를 $\xi^\mu$라 둔다. 전체 작용의
metric variation으로 얻은 stress tensor가 다음처럼 분해된다고 하자.

$$
T_{\mu\nu}=T^{(b)}_{\mu\nu}+T^{(d)}_{\mu\nu},
\qquad
\nabla^\mu T_{\mu\nu}=0.
$$

$b$는 baryon-number를 운반하는 표준모형 sector이고 $d$는 그 여집합이다.
초곡면 에너지는

$$
E_i[\gamma;\Sigma_0]
=\int_{\Sigma_0}d\Sigma\,
T^{(i)}_{\mu\nu}[\gamma]n^\mu\xi^\nu,
\qquad i\in\{b,d\},
$$

로 정의한다. 허용 상태에서는 $E_b,E_d\ge0$이고
$E_{\rm tot}=E_b+E_d>0$라 가정한다. Euclidean 준비 측도

$$
d\mu(\gamma)=Z^{-1}e^{-S_E[\gamma]/\hbar}\,\mathcal D\gamma
$$

에 대해 정규화된 에너지 가중치는

$$
\mathbb P_E(A)=
\frac{\int_A E_{\rm tot}[\gamma]\,d\mu(\gamma)}
{\int E_{\rm tot}[\gamma],d\mu(\gamma)}
$$

이다. 따라서 단순 경로 확률 $\mu(A)$와 $\mathbb P_E(A)$는 일반적으로
다르다. 두 측도가 같으려면 최소한
$\operatorname{Cov}_\mu(\mathbf1_A,E_{\rm tot})=0$가 추가로 필요하다.

## 3. B2 폐쇄 공리

다음 네 조건을 B2 모형의 정의로 고정한다.

1. **고정 projector:** $\Pi_b$는 관측값을 본 뒤 정하는 집합이 아니라,
   baryon-number current $J_B^\mu$를 운반하는 sector의 projector다.
2. **에너지 readout:** CE의 생존변수는 경로 개수가 아니라
   $x=\langle E_b\rangle_\mu/\langle E_{\rm tot}\rangle_\mu$로 정의한다.
3. **독립 결함 입력:** $D_{\rm eff}$는 $\Omega_b$를 넣어 역산하지 않고
   차원·결합상수 branch에서 먼저 고정한다.
4. **관측면 지정:** 고정점은 $a_0=1$의 renormalized stress tensor에
   부과한다. 다른 epoch로 옮길 때에는 6절의 전달식을 반드시 사용한다.

조건 2는 연산자 언어로

$$
x=
\frac{\operatorname{Tr}\!\left(\rho_0 H_0^{1/2}\Pi_bH_0^{1/2}\right)}
{\operatorname{Tr}(\rho_0H_0)}
$$

이다. $[H_0,\Pi_b]=0$인 block-conserving limit에서는 분자가
$\operatorname{Tr}(\rho_0H_0\Pi_b)$로 단순화된다. 비가환인 경우에도 위의
대칭형 정의는 양수이고 기저 선택에 무관하다.

## 4. B2 정리와 증명

**정리.** 2절과 3절의 조건을 만족하고, $D_{\rm eff}>1$일 때 물리 가지
$0<x<1$에 CE 재귀식

$$
x=e^{-(1-x)D_{\rm eff}}
$$

을 부과하자. 그러면

$$
x=-\frac{W_0(-D_{\rm eff}e^{-D_{\rm eff}})}{D_{\rm eff}},
\qquad
\Omega_b(a_0)=x.
$$

**증명.** 첫 식에 $xe^{-D_{\rm eff}x}=e^{-D_{\rm eff}}$를 적용하면
$-D_{\rm eff}x=W_0(-D_{\rm eff}e^{-D_{\rm eff}})$를 얻는다. $D_{\rm eff}>1$
에서는 $x=1$인 경계 가지와 $0<x<1$인 저생존 가지가 있으며, $W_0$가
후자를 준다. 한편 균질 FLRW에서

$$
\langle E_i\rangle_\mu=V_0\rho_i(a_0),
\qquad
\rho_{\rm tot}(a_0)=\rho_c(a_0)
$$

이므로 공통 부피 $V_0$가 소거되어

$$
x=\frac{\rho_b(a_0)}{\rho_{\rm tot}(a_0)}
=\frac{\rho_b(a_0)}{\rho_c(a_0)}=\Omega_b(a_0).
$$

따라서 결론이 성립한다. $\square$

이 증명은 “경로 수의 $4.9\%$가 바리온이다”라는 명제를 사용하지 않는다.
핵심은 CE 재귀의 표본공간을 처음부터 공변 에너지 측도로 고정한 것이다.

## 5. 에너지 보존과 전하 장부를 갖는 동역학적 실현

위 폐쇄가 unitary dynamics와 양립함을 보이는 최소 구성은 두 개의 에너지
퇴화 sector에 대한 회전이다. $\mathcal H=\mathcal K\otimes\mathbb C^2$,
$H=H_{\mathcal K}\otimes I_2$로 두고, 내부 기저를
$\{|B\rangle,|\bar L\rangle\}$로 잡는다. 두 상태의 전하는 각각
$(B,L)=(1,0)$과 $(0,-1)$이므로 둘 다 $B-L=1$이다. 다음 unitary를 쓴다.

$$
U_x=
\begin{pmatrix}
\sqrt{1-x}&\sqrt{x}\\
-\sqrt{x}&\sqrt{1-x}
\end{pmatrix},
\qquad [U_x,H]=[U_x,Q_{B-L}]=0,
\qquad [U_x,Q_B]\ne0.
$$

초기 $|\bar L\rangle$ 상태에 $U_x$를 적용하면
$U_x|\bar L\rangle=\sqrt{x}|B\rangle+\sqrt{1-x}|\bar L\rangle$이므로
첫 sector의 에너지 가중치는 정확히 $x$다. 총에너지와 $B-L$은 보존되지만
$B$와 $L$은 각각 보존되지 않는다. 이는 바리온 sector의 가중치를 바꾸는
unitary가 $[U,Q_B]=0$일 수 없다는 전하 선택규칙을 명시적으로 만족한다.

실제 재가열/전기약 모형에서는 이 $2\times2$ 회전을 CP-비대칭 반응,
sphaleron과 transport를 포함한 더 큰 unitary 또는 그 CPTP 축약으로
교체한다. 반대로 $B$가 정확히 보존되는 분기에서는 동역학이 $x$를 생성할
수 없으므로 초기 상태에 이미 baryon weight $x$가 있어야 한다. 이 구성은
B2가 양의 측도·정규화·unitarity·전하 장부를 동시에 만족하는 비어 있지
않은 모형임을 보인다.

## 6. 다른 epoch로의 전달

$a_\star\ne a_0$에서 성분을 지정하면 서로 다른 상태방정식 때문에 분율은
그대로 유지되지 않는다. 상호작용원 $Q_i$를 포함한 식은

$$
\dot\rho_i+3H(1+w_i)\rho_i=Q_i,
\qquad \sum_iQ_i=0.
$$

따라서 전달행렬 $T(a_0,a_\star)$를 계산한 뒤

$$
\boldsymbol\Omega(a_0)=
\frac{T(a_0,a_\star)\boldsymbol\rho(a_\star)}
{\mathbf1^{\mathsf T}T(a_0,a_\star)\boldsymbol\rho(a_\star)}
$$

로 읽어야 한다. $\Omega_b(a_\star)=x$를 가정하고 곧바로
$\Omega_b(a_0)=x$라고 쓰는 레거시 절차는 이 문서의 B2 모형이 아니다.

바리온-광자 비도 정의식

$$
\eta_b(a_0)=
\frac{\Omega_b(a_0)\rho_c(a_0)}{m_bn_\gamma(a_0)}
$$

로 계산하며, $H_0$, $T_{\rm CMB}$, 평균 baryon mass를 입력 장부에
기록한다. $\eta_b$는 $x$만으로 추가 입력 없이 생기는 독립 예측이 아니다.

## 7. 수치와 검증 계약

$D_{\rm eff}$가 고정되면 $x$는 bracketed solver와 Lambert-$W$ 식으로
독립 검산한다. 관측 비교에는
[`OBSERVATIONAL_BASELINE_2026-08-06.md`](OBSERVATIONAL_BASELINE_2026-08-06.md)의
$\omega_b=\Omega_bh^2$와 $h$의 공동 covariance를 사용한다. 느슨한
$\Omega_b=0.0486\pm0.0010$을 사용해 만든 과거의 “$0.05\sigma$” 문구는
현재 검증값으로 재사용하지 않는다.

다음 조건은 B2를 반증하거나 적용범위 밖으로 만든다.

- $D_{\rm eff}$를 $\Omega_b$에서 역산했는데 독립 출력이라고 보고한 경우
- $\Pi_b$가 데이터 비교 후 선택된 경우
- renormalized $T_{\mu\nu}$가 보존되지 않거나 분모 에너지가 양수가 아닌 경우
- 다른 epoch의 분율을 전달식 없이 동일시한 경우
- 동일 데이터가 $D_{\rm eff}$ 고정과 최종 검증에 중복 사용된 경우

## 8. 완료 판정

B2는 이제 다음 의미에서 닫혀 있다.

| 항목 | 완성 내용 |
|---|---|
| 표본공간 | Euclidean 준비 측도와 에너지 가중 측도를 분리 |
| 물리량 | metric variation의 $T_{\mu\nu}$로 $E_b/E_{\rm tot}$ 정의 |
| 대수 | Lambert-$W$ 물리 가지와 안정성 조건 고정 |
| 동역학 | 에너지 보존 unitary 실현과 일반 전달식 제시 |
| 관측 | $a_0$, $\rho_c$, covariance와 입력 재사용 금지 명시 |

즉, 정리의 결론은 명시된 B2 공리계 안에서 정확하다. 자연이 이 projector와
에너지 재귀를 선택하는지는 독립 likelihood와 holdout으로 검증한다. 이는
수학적 폐쇄를 관측 사실로 가장하지 않으면서도, 누락된 메커니즘 대신 실제
계산 가능한 모형을 제공한다.

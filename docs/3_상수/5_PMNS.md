# 5층: PMNS -- 단일 질량행렬 benchmark

## 개관

중성미자 혼합은 세 확률 사이의 ``보정 예산''이 아니라 charged-lepton과 neutrino mass matrix의 상대 대각화다.

$$
U_{\rm PMNS}=U_{eL}^\dagger U_\nu.
$$

따라서 각 혼합각에 서로 다른 규칙을 붙이지 않고, 하나의 대칭적인 Majorana mass matrix를 먼저 만든 뒤 세 각·CP 불변량·질량차를 함께 계산한다. 아래는 기존 TBM 직관과 CE의 $\delta_N$을 보존하는 정합한 benchmark construction이다.

---

## 0차 기준선과 물리적 섭동

TBM 행렬을

$$
U_{\rm TBM}=\begin{pmatrix}
\sqrt{2/3}&1/\sqrt3&0\\
-1/\sqrt6&1/\sqrt3&1/\sqrt2\\
1/\sqrt6&-1/\sqrt3&1/\sqrt2
\end{pmatrix}
$$

로 둔다. 이는 $\theta_{13}=0$인 0차 texture이지, $d=3$만으로 유일하게 따라오는 정리가 아니다.

일반적인 질량행렬 섭동에서는 mass basis의 비대각 성분과 고유값 간격이 함께 들어간다. CP를 보존하는 real-symmetric 2--3 block을 적절히 rephase한 예에서는

$$
\tan2\eta=
\frac{2|\Delta M_{23}|}
{m_3^{(0)}-m_2^{(0)}+\Delta M_{33}-\Delta M_{22}}.
$$

따라서 ``SU(3) 생성원이 8개이므로 확률이 $\delta/8$''이라는 결론은 섭동론에서 나오지 않는다. 아래에서는 $\delta_N/8$을 관측량별 보정이 아니라 **하나의 rotation angle을 정하는 texture ansatz**로 사용한다.

---

## TM1 benchmark

charged-lepton basis를 $U_{eL}=I$로 고르고

$$
U_\nu=U_{\rm TBM}R_{23}(\eta,\varphi),
$$

$$
R_{23}(\eta,\varphi)=
\begin{pmatrix}
1&0&0\\
0&c_\eta&s_\eta e^{-i\varphi}\\
0&-s_\eta e^{i\varphi}&c_\eta
\end{pmatrix}
$$

로 정의한다. 구성상 $U_\nu U_\nu^\dagger=I$다. CE neutral-projector seed를

$$
\boxed{\sin^2\eta=\frac{3\delta_N}{8}},
\qquad \delta_N=0.17791300
$$

로 한 번만 사용하면

$$
\eta=0.26125904\ {\rm rad}
$$

이다. 이 단일 rotation에서

$$
\boxed{s_{13}^2=\frac{\sin^2\eta}{3}=\frac{\delta_N}{8}=0.02223912}
$$

와 TM1 sum rule

$$
\boxed{s_{12}^2=1-\frac{2}{3(1-s_{13}^2)}=0.31817003}
$$

가 동시에 나온다. 기존 $s_{12}^2=(1-3s_{13}^2)/3$은 PMNS unitarity가 요구하는 식이 아니므로 폐기한다.

대기각은 같은 행렬에서

$$
s_{23}^2=
\frac{\frac12c_\eta^2+\frac13s_\eta^2
+\frac{2}{\sqrt6}s_\eta c_\eta\cos\varphi}
{1-s_{13}^2}
$$

로 나온다. 따라서 octant와 leptonic CP는 하나의 연속 위상 $\varphi$에 함께 의존한다. $i$의 존재나 ``남은 $7/8$ 예산''만으로 $s_{23}^2$ 또는 $\delta_{CP}$를 고정하지 않는다.

---

## 실제 Majorana mass matrix

질량 고유값 $m_i\ge0$와 Majorana phase matrix

$$P_M=\operatorname{diag}(1,e^{i\alpha_{21}/2},e^{i\alpha_{31}/2})$$

를 두고 $U=U_\nu P_M$라 하자. flavour basis의 대칭 질량행렬을

$$
\boxed{M_\nu=U^*\operatorname{diag}(m_1,m_2,m_3)U^\dagger}
$$

로 정의하면

$$
U^TM_\nu U=\operatorname{diag}(m_1,m_2,m_3)
$$

가 정확히 성립한다. 즉 세 혼합각, Dirac phase, 두 Majorana phase와 질량차는 모두 한 행렬의 출력이다. 이는 정합한 존재 construction이며, $m_i$, $\varphi$, $\alpha_{21}$, $\alpha_{31}$을 정하는 UV flavour symmetry는 별도 gate다.

Dirac CP의 convention-independent 검사는

$$
J_\ell=\operatorname{Im}
(U_{e1}U_{\mu2}U_{e2}^*U_{\mu1}^*)
$$

로 한다. $\delta_{CP}^{\rm PMNS}=3\pi/2$를 사용하려면 $\varphi$와 표준 PDG phase 사이의 변환을 이 불변량으로 계산해야 하며, CKM phase의 켤레라는 이유만으로 넣지 않는다.

---

## 공동 검증 계약

benchmark parameter vector와 observable vector를

$$
\mathbf p=(m_{\rm lightest},\Delta m_{21}^2,\Delta m_{3\ell}^2,
\varphi,\alpha_{21},\alpha_{31}),
$$

$$
\mathbf O=(s_{12}^2,s_{13}^2,s_{23}^2,\delta_{CP},
\Delta m_{21}^2,\Delta m_{3\ell}^2)
$$

로 둔다. 한 global-fit release·mass ordering·공분산을 고정해

$$
\chi^2_{\rm PMNS}=(\mathbf O_{\rm th}-\mathbf O_{\rm fit})^T
C^{-1}(\mathbf O_{\rm th}-\mathbf O_{\rm fit})
$$

를 한 번 계산한다. 각 angle의 서로 다른 snapshot과 오차를 골라 개별
$\sigma_{\rm obs}$를 더하지 않는다. $m_{\rm lightest}$와 Majorana phases는 oscillation이 직접 측정하지 않으므로 별도 cosmology·$0\nu\beta\beta$ likelihood에 둔다.

---

## 현재 출력

| 항목 | benchmark 결과 | 지위 |
|---|---:|---|
| $s_{13}^2$ | 0.02223912 | $\sin^2\eta=3\delta_N/8$ texture의 출력 |
| $s_{12}^2$ | 0.31817003 | 같은 unitary TM1 matrix의 필수 sum rule |
| $s_{23}^2$ | $\varphi$ 의존 | phase/UV texture 없이 숫자 고정 금지 |
| $\delta_{CP}$ | $\varphi$ 의존 | CKM phase와 독립 |
| $M_\nu$ | $U^*D_mU^\dagger$ | exact symmetric benchmark construction |

이 재구성은 기존의 좋은 $s_{13}^2$ seed를 보존하면서, eigenvalue gap 누락·임의 generator 분배·거짓 unitarity budget을 제거한다. 실제 모형으로 닫히려면 $\Delta M_{23}$와 mass spectrum을 만드는 gauge-invariant Weinberg operator 또는 UV seesaw를 제시해야 한다.

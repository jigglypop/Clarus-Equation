# 비선택 양자경로의 정보적 기회비용

Status: COMPLETE

## 초록

본 연구는 선택되지 않은 양자경로를 “기회비용”으로 세는 최소 함수를 정의했다.
유한한 측정 outcome에서 비선택 weighted surprisal
$C_I(o)=-\sum_{a\ne o}p_a\ln p_a$는 무차원 정보량으로 일관된다. 이 값은
Hamiltonian과 bath가 정해진 경우에만 $k_BT D(\rho_U\|\gamma_T)$라는
비평형 자유에너지로 조건부 승격된다. 실제 중력원은 정보량이 아니라 별도
공변 유효작용의 metric variation으로 정의해야 한다. 따라서 “에너지 없는
에너지”는 정보적 shadow price라는 뜻으로는 성립하지만, 암흑에너지나
암흑물질이라는 동일성은 아직 성립하지 않는다.

## 1. 문제와 핵심 결과

측정 outcome 하나가 선택되면 다른 outcome은 실제 기록에서 제외된다. 경제학적
비유로 그 제외를 기회비용이라 부를 수 있지만, 물리학에서는 비용의 타입부터
정해야 한다. 확률은 무차원이고 에너지는 차원이 있으므로 두 양은 자동으로 같을
수 없다.

본 연구가 채택한 가장 작은 정보적 비용은

$$
\boxed{
C_I(o)=-\sum_{a\ne o}p_a\ln p_a
}
\tag{1}
$$

다. 이 식은 비선택 경로의 확률질량과 내부 다양성을 함께 센다. 실제 energy로
읽는 단계는 식 (1)의 정의가 아니라 추가 열역학 또는 유효장이론이다.

## 2. instrument와 비선택 상태

quantum instrument $\{\mathcal I_a\}$의 outcome 확률과 조건부 상태를

$$
p_a=\operatorname{tr}\mathcal I_a(\rho),
\qquad
\rho_a=\frac{\mathcal I_a(\rho)}{p_a}
$$

로 둔다. 선택 outcome을 $o$, 비선택 집합을 $U$라 하면

$$
p_U=1-p_o,
\qquad
\rho_U=\frac1{p_U}\sum_{a\in U}\mathcal I_a(\rho)
$$

다. 이 식은 비선택 조건부 상태의 정의다. 선택 branch의 stress tensor에
$\rho_U$를 더하라는 규칙은 아니다.

## 3. 정보 기회비용의 유도

$q_a=p_a/p_U$를 비선택 집합 안의 조건부 분포라 하자. 식 (1)에
$p_a=p_Uq_a$를 대입하면

$$
\begin{aligned}
C_I(o)
&=-\sum_{a\in U}p_Uq_a\ln(p_Uq_a)\\
&=-p_U\ln p_U\sum_{a\in U}q_a
-p_U\sum_{a\in U}q_a\ln q_a\\
&=p_U[-\ln p_U+H(q)].
\end{aligned}
\tag{2}
$$

식 (2)에서 첫 항은 제외된 총확률질량을, 둘째 항은 비선택 대안의 다양성을
센다. 유한 outcome에서 $p_U\to0$이면 $C_I\to0$이다. 반면
$-\ln p_U$만 비용으로 택하면 대안이 사라질수록 발산하고, $H(q)$만 택하면
비선택 대안이 하나인 두-outcome 문제에서 항상 0이다. 따라서 둘을 합친
weighted surprisal이 더 적합하다.

기존 0D-carrier 모형에는

$$
\mu_C(B)=\int w(\gamma)[-\ln p_\gamma]
\mathbf1_B(F(\gamma))\nu_{\rm ns}(d\gamma)
\tag{3}
$$

로 붙일 수 있다. 식 (3)은 carrier를 정보 기회비용으로 장식할 뿐, energy
density를 만들지 않는다. 연속 path에서는 개별 경로 확률이 0일 수 있으므로
유한 coarse-graining 또는 사전 고정 reference에 대한 relative entropy가
필요하다.

## 4. 언제 energy 차원을 얻는가

정보량만으로 energy 차원을 만들 수 없다. 독립 scale $E_*$를 넣은
$E_*C_I$, thermal scale을 넣은 $k_BT C_I$, 시간 scale을 넣은
$\hbar C_I/\tau_*$는 차원상 energy지만, scale 선택은 새 물리 입력이다.

Hamiltonian $H$, bath temperature $T>0$와 Gibbs state

$$
\gamma_T=\frac{e^{-H/(k_BT)}}{Z}
$$

를 고정하면 더 물리적인 식을 얻는다. dimensionless von Neumann entropy를
$S(\rho)=-\operatorname{tr}(\rho\ln\rho)$라 할 때

$$
F_T(\rho)=\operatorname{tr}(\rho H)-k_BT S(\rho)
$$

이고

$$
\boxed{
F_T(\rho_U)-F_T(\gamma_T)
=k_BT D(\rho_U\|\gamma_T)
}
\tag{4}
$$

가 성립한다. 식 (4)는 비선택 상태가 실제로 준비되고 지정된 thermal operation
안에서 자원으로 쓰일 때의 조건부 자유에너지 excess다. trial당 평균량을 원하면
$p_U$를 추가로 곱할 수 있다. Landauer 원리는 물리 memory의 비가역적 reset과
열비용을 연결하지, 모든 미실현 outcome이 energy를 저장한다고 말하지 않는다.

## 5. 영향함수와 중력원

비선택·환경 경로가 실제 동역학에 남는 형식으로는 Feynman--Vernon influence
functional이 더 직접적이다.

$$
e^{iS_{\rm IF}[q_+,q_-]/\hbar}
=\int\mathcal D\xi_+\mathcal D\xi_-
e^{i(S_+-S_-)/\hbar}.
\tag{5}
$$

이 적분은 기억, 감쇠와 잡음을 유효동역학에 남긴다. 그러나 $S_{\rm IF}$는
일반적으로 복소·비국소이고 action 차원이다. 양의 local energy density가 아니다.

중력 source를 말하려면 독립 energy-density scale $\epsilon_*$와 공변 action을

$$
S_{\rm opp}=-\int d^4x\sqrt{-g}\,
V_{\rm opp}(C,chi,\nabla\chi;\epsilon_*)
\tag{6}
$$

처럼 먼저 채택한 뒤

$$
T_{\mu\nu}^{\rm opp}
=-\frac2{\sqrt{-g}}
\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}
\tag{7}
$$

로 정의해야 한다. 예를 들어 $V_{\rm opp}=\epsilon_*f(C)$이고 $C$가 상수이면
$T_{\mu\nu}=-V_{m opp}g_{\mu\nu}$라서 $w=-1$인 readout을 얻는다. 그러나 이
결과는 $\epsilon_*$와 potential을 새로 가정했기 때문에 나오는 조건부 EFT다.
$C(x)$가 외부에서 변하면 이 stress는 단독 보존되지 않으므로 apparatus,
environment와 reservoir를 포함한 full action이 필요하다.

## 6. 명시적 두-outcome 계산

$p=(0.8,0.2)$이고 첫 outcome을 선택하면

$$
H(p)=0.5004024235381879,
$$

$$
-\ln p_U=1.6094379124341003,
$$

$$
C_I=-0.2\ln0.2=0.3218875824868201.
$$

이 셋은 모두 nat 단위의 무차원 정보량이다. 비선택 outcome이 하나이므로
$H(q)=0$이다. $E_0=0$, $E_1=\Delta$를 별도로 주면 expected energy regret는
$0.2\Delta$지만, $\Delta$가 없으면 energy 비용은 정의되지 않는다.

## 7. 관측 지위와 미완성 과제

이 연구는 관측 우주론 자료를 사용하지 않았다. 식 (1)은 정보 readout이고,
식 (4)는 지정된 thermodynamic setup의 조건부 결과며, 식 (6)은 아직 미시적으로
유도하지 않은 물리 사상이다. 암흑물질·암흑에너지 동일성, $\epsilon_*$의 값,
압력과 섭동, abundance와 Einstein--Boltzmann 관측량은 미완성이다.

다음 연구에서는 측정이 순간 사건인지 유한 시간의 interaction region인지 먼저
구분해야 한다. 완료된 discrete record는 outcome 공간의 점으로 볼 수 있지만,
apparatus coupling은 일반적으로 시간과 공간에 걸쳐 있다. 따라서 측정의
“0차원성”은 시공간 support와 별도로 정의해야 한다.

## 8. 재현성

수치·차원 검산은 다음 명령으로 재현한다.

```powershell
.codex\hooks\python.cmd python _workspace\ce\quantum-path-opportunity-functional-20260825\artifacts\verify_opportunity_cost.py
```

검산기는 14개 등식·부등식과 차원 관계를 허용오차 $10^{-12}$에서 확인한다.

## 9. 참고문헌

1. R. Landauer, “Irreversibility and Heat Generation in the Computing Process,”
   *IBM Journal of Research and Development* 5 (1961),
   [DOI 10.1147/rd.53.0183](https://doi.org/10.1147/rd.53.0183).
2. T. Sagawa and M. Ueda, “Minimal Energy Cost for Thermodynamic Information
   Processing,” *Physical Review Letters* 102 (2009),
   [DOI 10.1103/PhysRevLett.102.250602](https://doi.org/10.1103/PhysRevLett.102.250602).
3. F. G. S. L. Brandão et al., “Resource Theory of Quantum States Out of Thermal
   Equilibrium,” *Physical Review Letters* 111 (2013),
   [DOI 10.1103/PhysRevLett.111.250404](https://doi.org/10.1103/PhysRevLett.111.250404).
4. R. P. Feynman and F. L. Vernon Jr., “The Theory of a General Quantum System
   Interacting with a Linear Dissipative System,” *Annals of Physics* 24 (1963),
   [DOI 10.1016/0003-4916(63)90068-X](https://doi.org/10.1016/0003-4916(63)90068-X).
5. E. Calzetta and B. L. Hu, “Closed-Time-Path Functional Formalism in Curved
   Spacetime,” *Physical Review D* 35 (1987),
   [DOI 10.1103/PhysRevD.35.495](https://doi.org/10.1103/PhysRevD.35.495).

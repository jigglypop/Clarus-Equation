# 측정은 한순간인가: 0차원 기록과 유한시간 벽의 분리

Status: COMPLETE

## 초록

이 연구는 “0차원을 측정 행위의 차원으로 볼 수 있는가”와 “측정이 언제나 벽인가”를 측정 결과, 상호작용 지지, 시간 과정이라는 서로 다른 공간으로 나누어 검사했다. 유한 이산 결과 하나는 결과공간에서 0차원 원자로 정의할 수 있지만, 그것을 만드는 측정 상호작용은 일반적으로 유한시간의 worldline 또는 worldtube를 차지한다. 완전 직교 포인터 사영에 대한 dephasing map은 정확한 벽 모형을 주며, 유한 누적강도에서는 부분벽이고 무한강도 또는 이상적 Zeno 극한에서만 hard wall이 된다. 예제에서는 $\eta=0.776869839851570$과 잔여 coherence $0.111565080074215$를 얻었다. 벽 형성과 함께 정의한 기회비용은 무차원 정보량이며 에너지나 암흑부문 stress가 아니다. 따라서 0차원 기록에서 persistent fold-memory 및 암흑물질·암흑에너지로 가는 두 물리 사상은 미완성으로 남는다.

## 1. 문제를 네 층으로 분해한다

“측정의 차원”이라는 한 표현에는 실제로 네 질문이 섞여 있다.

1. 결과공간 $\mathcal O$의 위상 차원은 얼마인가?
2. 검출기와 계가 결합하는 시공간 영역 $\mathcal R_M\subset M$의 차원은 얼마인가?
3. protocol이 점유하는 시간구간 $[t_0,t_1]$은 유한한가?
4. 선택 전후의 history를 가르는 operational cut은 어떤 경계인가?

유한 이산 집합 $\mathcal O=\{r_1,\ldots,r_n\}$과 singleton $\{r\}$은 위상적으로 0차원이다. 이 의미에서는 완료된 측정 기록을 **0차원 record atom**이라고 부를 수 있다. 그러나 유한시간 동안 작동하는 점검출기는 1차원 worldline segment, 2차원 공간 표면 검출기는 시간까지 포함해 3차원 worldtube, 유한 부피 장치는 보통 4차원 시공간 영역을 점유한다. 따라서 record의 0차원성과 물리적 상호작용의 0차원성은 같은 명제가 아니다.

## 2. 유한시간 측정의 최소 모형

계 $S$, apparatus $A$, 초기 apparatus 상태 $\sigma_A$를 두고 $t_0<t_1$에서

$$
U_M(t_1,t_0)=\mathcal T\exp\left[-\frac{i}{\hbar}
\int_{t_0}^{t_1}H_{SA}(t)dt\right]
$$

로 결합시킨다. 완전 직교 pointer projector $\{\Pi_r\}$가

$$
\Pi_r\Pi_s=\delta_{rs}\Pi_r,
\qquad
\sum_r\Pi_r=I_A
$$

를 만족할 때 결과별 instrument는

$$
\mathcal I_r^{[t_0,t_1]}(\rho)
=\operatorname{tr}_A\left[(I\otimes\Pi_r)U_M
(\rho\otimes\sigma_A)U_M^\dagger(I\otimes\Pi_r)\right]
$$

이다. 각 map은 completely positive이고 trace non-increasing이며, 합 $\sum_r\mathcal I_r$은 trace preserving이다. 일반 POVM effect $E_r$를 같은 sandwich에 그대로 넣으면 안 된다. $E_0=E_1=I/2$이면 $\sum_rE_r=I$이지만 $\sum_rE_r^2=I/2$라서 전체 trace가 보존되지 않는다. 일반 경우에는

$$
\mathcal I_r(\rho)=\sum_\alpha M_{r\alpha}\rho M_{r\alpha}^\dagger,
\qquad
\sum_{r,\alpha}M_{r\alpha}^\dagger M_{r\alpha}=I
$$

인 Kraus instrument가 필요하다.

이 식에서 측정은 $[t_0,t_1]$ 동안 진행되는 과정이다. endpoint $r$만 점 같은 기록이다. 또한 전체 $S+A$는 unitary하게 얽힌 상태로 남을 수 있으므로, 여기서 말하는 벽은 전 우주의 파동함수에 대한 객관적 붕괴가 아니라 선택한 record algebra 또는 접근 가능한 reduced state에 상대적인 벽이다.

## 3. 벽은 이분법이 아니라 강도다

직교 record partition $\{P_r\}$에 대해

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r
$$

를 둔다. $P_rP_s=\delta_{rs}P_r$이므로

$$
\mathcal D_P^2=\mathcal D_P,
$$

이고 $P_r$ 자체가 Kraus operators이므로 $\mathcal D_P$는 CPTP다. 이제

$$
\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P,
\qquad 0\le\eta\le1
$$

를 정의한다. Kraus list $\sqrt{1-\eta}I$, $\sqrt\eta P_r$가 completeness를 만족하므로 이 family도 CPTP다. rank-one pointer basis에서는

$$
(\Phi_\eta\rho)_{rr}=\rho_{rr},
\qquad
(\Phi_\eta\rho)_{rs}=(1-\eta)\rho_{rs}\quad(r\ne s).
$$

따라서 $\eta=0$은 벽 없음, $0<\eta<1$은 coherence가 통과하는 부분벽, $\eta=1$은 선택한 block 사이 coherence를 완전히 지우는 hard wall이다. 약한 측정과 unsharp measurement는 부분벽의 직접 사례이고, identity instrument는 벽이 전혀 없는 대조군이다. 그러므로 “모든 측정은 언제나 hard wall”이라는 보편 명제는 제거해야 한다.

## 4. 벽이 시간에 걸쳐 만들어지는 정확한 해

벽 형성률을 $\gamma(t)\ge0$라 하고 phenomenological generator를

$$
\dot\rho_t=\gamma(t)(\mathcal D_P-I)\rho_t
$$

로 둔다. 누적강도

$$
\Gamma(t)=\int_{t_0}^{t}\gamma(s)ds
$$

는 무차원이다. $\mathcal D_P$가 projector superoperator이므로 operator space를 $\operatorname{im}\mathcal D_P$와 $\ker\mathcal D_P$로 나누면 지수함수가 정확히

$$
e^{\Gamma(\mathcal D_P-I)}
=\mathcal D_P+e^{-\Gamma}(I-\mathcal D_P)
$$

가 된다. 따라서

$$
\rho_t=\mathcal D_P\rho_{t_0}
+e^{-\Gamma(t)}(I-\mathcal D_P)\rho_{t_0}
=\Phi_{\eta(t)}(\rho_{t_0}),
$$

$$
\boxed{\eta(t)=1-e^{-\Gamma(t)}}
$$

이다. 이 식은 사용자의 질문에 대한 직접적인 답을 준다. 측정이 한순간일 필요는 없고, 벽의 형성은 연속적일 수 있다. 더 강하게, 유한 $\Gamma$에서는 항상 $\eta<1$이므로 완전한 벽은 유한시간·유한 rate 모형의 일반 결과가 아니라 $\Gamma\to\infty$의 이상극한이다. 반복 사영의 Zeno limit도 적절한 Hamiltonian/domain 조건과 $N\to\infty$를 요구하는 별도 이상극한이다.

## 5. 계산 예제

$$
\rho_0=\frac12
\begin{pmatrix}1&1\\1&1\end{pmatrix},
\qquad
P_0=|0\rangle\langle0|,
\quad P_1=|1\rangle\langle1|
$$

이면

$$
\rho_t=
\begin{pmatrix}
1/2&e^{-\Gamma(t)}/2\\
e^{-\Gamma(t)}/2&1/2
\end{pmatrix}.
$$

$\gamma_0=2$, $t-t_0=0.75$에서는 $\Gamma=1.5$이고

$$
\eta=1-e^{-1.5}=0.776869839851570,
\qquad
\rho_{01}=e^{-1.5}/2=0.111565080074215.
$$

즉 대각 확률은 그대로이지만 coherence의 약 $22.31\%$가 아직 남는다. 이 측정은 상당히 강하지만 hard wall은 아니다.

## 6. 선택되지 않은 경로의 기회비용을 시간 과정에 붙이기

결과 $o$가 완료된 뒤의 endpoint 기회비용은 선행 연구에서

$$
C_I(o,t_1)=-\sum_{a\ne o}p_a(t_1)\ln p_a(t_1)
$$

로 정의했다. 그러나 최종 결과 $o$를 측정 도중에 미리 사용하면 retrospective/postselected 계산이 된다. 실시간 예측 가능한 ensemble quantity는

$$
\overline C_I(t)
=\sum_o p_o(t)C_I(o;t)
=\sum_a p_a(t)[1-p_a(t)][-\ln p_a(t)]
$$

이다. 벽이 실제로 형성되는 구간에만 가중하려면

$$
\boxed{
C_{\rm wall}=\int_{t_0}^{t_1}
\dot\eta(t)\,\overline C_I(t)dt
}
$$

를 정의할 수 있다. $\dot\eta$의 차원은 $T^{-1}$이고 $dt$는 $T$이므로 $C_{\rm wall}$은 무차원이다. $(p_0,p_1)=(0.8,0.2)$가 일정하고 위 예제의 $\eta$를 쓰면

$$
\overline C_I=0.293213034199730,
\qquad
C_{\rm wall}=0.227788362921137.
$$

이 수치는 측정벽이 형성되는 동안 배제 가능성이 얼마나 누적되었는지를 나타내는 정보 회계량이다. $k_BT$, $\hbar/\tau$, Hamiltonian gap 또는 covariant action density가 따로 주어지지 않았으므로 에너지가 아니다.

## 7. persistent 0차원 접힘장과 연결되는 정확한 위치

현재 닫힌 구조와 CE 가설의 경계는 다음 합성으로 표현할 수 있다.

$$
(S,A,\rho,\sigma_A)
\xrightarrow{\ U_M[t_0,t_1]\ }
\{\mathcal I_r\}_{r\in\mathcal O}
\xrightarrow{\ \text{retention map}\ }
\mu_{F,t}
\xrightarrow{\ K^F_{\ell,R}\ }
\chi.
$$

첫 화살표는 유한시간 quantum instrument다. 둘째 화살표는 완료되거나 배제된 기록이 persistent spatial-0D carrier로 남는다는 **미완성 물리 사상**이다. 셋째 화살표는 선행 run의 하나의 Volterra 환경장 모형이다. 예를 들어 예측 가능한 wall-deposition 후보는

$$
d\mu_{\rm wall}(t,B)
=\dot\eta(t)
\sum_a p_a(t)[1-p_a(t)][-\ln p_a(t)]
\mathbf1_B(F_t(a))dt
$$

로 쓸 수 있다. 이 식은 양의 무차원 measure를 만들지만, 표준 양자역학에서 유도된 법칙은 아니다. $F_t$, retention/decay rule, instrument dependence, covariance와 no-double-counting을 정해야 한다. 또한 이 deposit이 새 carrier를 만드는지 기존 carrier를 활성화하는지도 서로 다른 모형이므로 분리해야 한다.

retention map을 채택한 뒤에만 선행 환경장 식

$$
\chi(t,\mathbf x)=b(t,\mathbf x)+A\int_{t_i}^{t}ds
\int_{\Sigma_s}K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma(\chi(s,\mathbf y))\mu_F(d^3y)
$$

와 연결할 수 있다. 여기서도 0차원인 것은 각 공간 절편의 carrier support이며, 지속 carrier의 시공간 support는 worldline이다.

## 8. 암흑부문으로 가는 데 남은 다리

정보 measure $\mu_{\rm wall}$ 또는 환경장 $\chi$를 Einstein 방정식에 직접 넣을 수 없다. 최소한 covariant action

$$
S_{\rm opp}[g,\chi]
=-\int d^4x\sqrt{-g}\,
V_{\rm opp}(\chi,C_{\rm wall};\epsilon_*)
$$

와

$$
T_{\mu\nu}^{\rm opp}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}
$$

를 포함한 total conservation law가 필요하다. 에너지밀도 척도 $\epsilon_*$는 정보량에서 나오지 않는 독립 입력이다. $V_{\rm opp}$가 거의 상수라는 가정을 더하면 조건부로 $w\simeq-1$을 만들 수 있지만, 그것은 암흑에너지의 유도가 아니라 EFT 공리다. 암흑물질에는 다시 clustering, sound speed, Jeans scale, lensing 및 $a^{-3}$ scaling을 만족하는 별도 동역학이 필요하다.

따라서 현재 판정은 다음과 같다. “측정 기록의 0차원성”과 “측정벽의 유한시간 형성”은 수학적으로 함께 둘 수 있다. “선택되지 않은 경로의 기회비용”도 그 형성 과정에 무차원 가중치로 붙일 수 있다. 그러나 그것이 실제로 persistent fold, 에너지, 암흑물질 또는 암흑에너지라는 동일시는 아직 공리 및 관측 과제다.

## 9. 관측 비교와 한계

이번 연구는 관측자료를 적합하지 않았다. 기존 measurement literature는 instrument, weak measurement, continuous records, Zeno limit, bounded spacetime coupling을 지지하지만, record를 새 물리 차원이나 우주론적 stress로 해석하지 않는다. non-Markovian 환경에서는 recoherence 때문에 $\eta(t)$가 단조롭지 않을 수 있고, time-dependent noncommuting pointer basis에는 단일 $\mathcal D_P$ witness가 충분하지 않다. objective-collapse 해석과 unitary-only 해석 사이의 존재론도 이 operational 계산으로 결정되지 않는다.

## 10. 재현성

수학·반례·무차원 감사는 `11-math.md`, `12-routes.md`, `artifacts/dimensionless-audit.md`에 있고, 계산은 다음 명령으로 재현한다.

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\measurement-wall-dimensionality-20260825\artifacts\verify_measurement_wall.py'
```

## 참고문헌

- E. B. Davies and J. T. Lewis, “An operational approach to quantum probability,” *Commun. Math. Phys.* 17 (1970), [doi:10.1007/BF01647093](https://doi.org/10.1007/BF01647093), accessed 2026-08-25.
- E. B. Davies, “On the repeated measurement of continuous observables in quantum mechanics,” *J. Funct. Anal.* 6 (1970), [doi:10.1016/0022-1236(70)90064-9](https://doi.org/10.1016/0022-1236(70)90064-9), accessed 2026-08-25.
- B. Misra and E. C. G. Sudarshan, “The Zeno’s paradox in quantum theory,” *J. Math. Phys.* 18 (1977), [doi:10.1063/1.523304](https://doi.org/10.1063/1.523304), accessed 2026-08-25.
- Y. Aharonov, D. Z. Albert, and L. Vaidman, “How the result of a measurement ... can turn out to be 100,” *Phys. Rev. Lett.* 60 (1988), [doi:10.1103/PhysRevLett.60.1351](https://doi.org/10.1103/PhysRevLett.60.1351), accessed 2026-08-25.
- M. Ozawa, “Universally valid reformulation of the Heisenberg uncertainty principle on noise and disturbance in measurement,” *Phys. Rev. A* 67 (2003), [doi:10.1103/PhysRevA.67.042105](https://doi.org/10.1103/PhysRevA.67.042105), accessed 2026-08-25.
- V. P. Belavkin, “Quantum continual measurements and a posteriori collapse on CCR,” *Commun. Math. Phys.* 146 (1992), [doi:10.1007/BF02097018](https://doi.org/10.1007/BF02097018), accessed 2026-08-25.
- A. Barchielli and M. Gregoratti, “Quantum continuous measurements: The stochastic Schrödinger equations and the spectrum of the output,” *Quantum Meas. Quantum Metrol.* 1 (2013), [doi:10.2478/qmetro-2013-0005](https://doi.org/10.2478/qmetro-2013-0005), accessed 2026-08-25.
- C. J. Fewster and R. Verch, “Quantum Fields and Local Measurements,” *Commun. Math. Phys.* 378 (2020), [doi:10.1007/s00220-020-03800-6](https://doi.org/10.1007/s00220-020-03800-6), accessed 2026-08-25.

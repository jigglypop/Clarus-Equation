# 출처 레인

Status: COMPLETE

접근일: 2026-08-21

## 채택 출처와 허용되는 주장

| 주제 | 출처 | 이 run에서 허용되는 좁은 주장 | 금지되는 확대 해석 |
|---|---|---|---|
| Pullback metric | MIT 18.965, *Geometry of Manifolds*, https://www.mit.edu/~anser/files/18_965.pdf | $g=f^*G=J_f^\top GJ_f$이며 $f$가 immersion일 때 pullback은 Riemann metric이다. | rank-deficient $J$에 ridge를 더한 것을 원래의 Riemann metric이라고 부르지 않는다. |
| Pullback coordinate form | Mannheim, *Riemannian Geometry—Metrics and Connections*, https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/schmidt/FSS2024/Riemannian_Geometry/Web/RGch3.html | 좌표에서 pullback tensor와 full-column-rank 조건을 확인한다. | 이 표준 정리가 특정 뇌 회로의 생물학적 타당성을 증명하지 않는다. |
| Minimum-energy control | Klamka, *Controllability and Minimum Energy Control* (2018), https://doi.org/10.1007/978-3-319-92540-0 | fixed linear/LTV system의 reachable terminal displacement에 대해 Gramian pseudoinverse가 최소 quadratic input energy를 준다. | 비선형 뇌의 global reachability나 intrinsic geometry로 승격하지 않는다. |
| Gramian 수치 한계 | Sun & Motter, PRL 110, 208701 (2013), https://doi.org/10.1103/PhysRevLett.110.208701 | Gramian ill-conditioning과 nonlocal control 때문에 이론적 controllability와 실제 수치 제어가 다를 수 있다. | rank만 보고 현실적 제어 가능성을 선언하지 않는다. |
| 뇌 network controllability 경계 | Tu et al., *NeuroImage* 181 (2018), https://doi.org/10.1016/j.neuroimage.2018.04.010 | brain-network controllability 결과는 동역학·입력·모델 가정에 민감하며 degree 해석에 한계가 있다. | 구조 connectome만으로 제어에너지나 인지 메커니즘을 확정하지 않는다. |
| 지연 augmentation | Artstein, IEEE TAC 27 (1982), https://doi.org/10.1109/TAC.1982.1103023 | 지연 제어계를 확대 상태로 환원할 수 있다. | 측정되지 않은 edge delay를 공간거리에서 임의 생성하지 않는다. |
| discrete variable delay | Ben Gaid et al. (2008), https://doi.org/10.3182/20080706-5-KR-1001.00709; Qi et al. (2016), https://doi.org/10.1155/2016/6584523 | delay node/state augmentation 뒤 controllability를 다시 판정해야 한다. | variable delay를 하나의 고정 delay-line과 동일시하지 않는다. |
| 뉴런 비선형성 | Hodgkin & Huxley (1952), https://doi.org/10.1113/jphysiol.1952.sp004764 | 흥분성은 voltage-dependent gating dynamics를 가지므로 단일 공통 hard threshold는 생물물리학 정리가 아니다. | A6.1의 $\theta_i,\phi_i$를 HH에서 직접 유도했다고 주장하지 않는다. |
| 태아·신생아 folding | Dubois et al., *Brain* 131 (2008), https://doi.org/10.1093/brain/awn137 | 주요 cortical folding은 태아기 후반과 신생아기에 빠르게 발달하며 표면·sulcation을 MRI로 계량할 수 있다. | 주름이 성인기까지 완전히 고정되거나 유전자만으로 결정된다고 하지 않는다. |
| 영아기 longitudinal folding | Li et al., *J. Neurosci.* 34 (2014), https://pmc.ncbi.nlm.nih.gov/articles/PMC3960466/ | 첫 2년에도 global/local gyrification이 크게, 지역별로 다르게 변한다. | 출생 시 모든 주름이 완성됐다고 하지 않는다. |
| 청소년기 morphology | Mutlu et al., *PLoS ONE* 8 (2013), https://pmc.ncbi.nlm.nih.gov/articles/PMC3893168/; Alemán-Gómez et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC6618418/ | 청소년기에도 gyrification 감소, sulcal widening/depth 변화 같은 remodeling이 관찰된다. | 청소년기에 수조 개 회로가 물리적 새 주름을 직접 긋는다고 인과 해석하지 않는다. |
| 유전 영향 | Alexander-Bloch et al., PNAS 117 (2020), https://doi.org/10.1073/pnas.1912064117 | cortical folding에는 국소적이고 이질적인 유전 영향이 있다. | heritability를 결정론 또는 개인의 고정된 접힘 도면으로 읽지 않는다. |
| 물리 folding 모델 | Tallinen et al., *Nature Physics* 12 (2016), https://doi.org/10.1038/nphys3632 | differential cortical growth와 층상 연성체의 기계적 불안정이 실제 형상의 후보 메커니즘이다. | 상태공간 Jacobian의 pullback을 물리 조직 응력이나 해부학 곡률로 동일시하지 않는다. |

## 출처가 강제하는 타입 분리

1. **상태공간:** $a\mapsto F_T(a)$의 미분 $J_T$와 $J_T^\top G_TJ_T$.
2. **제어 입력공간:** $u_{0:T-1}$의 비용과 terminal endpoint value $E_T^*$.
3. **해부학 표면:** $X(\sigma,t)$의 first/second fundamental form과 성장·재료 법칙.

세 문헌 계열은 서로 다른 좌표·관측·가정을 사용한다. 공통으로 “metric”이라는 단어가 나타나도 tensor의 domain이 다르므로 등식으로 연결할 수 없다.

## 자료 입력 상태

이번 run은 수학 재정의만 수행한다. edge별 $W_{ij}$, $p_{ij}^n$, $d_{ij}$, neuron별 $\theta_i,\lambda_i$, actuator $B_n$, physical embedding $X(\sigma,t)$의 새 empirical receipt를 열거나 생성하지 않았다. 따라서 A6-P/C는 조건부 수학 후보이며, 실제 피질 주름 메커니즘은 `BLOCKED_INPUT`이다.

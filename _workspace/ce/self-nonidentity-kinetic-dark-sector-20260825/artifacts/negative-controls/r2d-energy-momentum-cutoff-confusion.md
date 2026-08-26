# R2-D 음성대조군 — 에너지 cutoff와 물리 파수의 혼동

Objective-ID: SNKC-K4-ADM-COMPLETION
Failure-Equation: $q_\times\le\Lambda_E$ with $q_\times=c_s\sqrt A/\bar M$ and $\Lambda_E=\Lambda_3c_s^{7/4}$
Minimal-Assumptions: $0<c_s<1$, 정준화된 작은-$c_s$ cubic power counting, 선형 분산 $\omega=c_sq$
Counterexample: $\Lambda_E$는 에너지이고 $q_\times$는 물리 파수이므로 서로 직접 비교할 수 없다; 대응 파수는 $q_{\rm sc}=\Lambda_E/c_s$
First-Failing-Line: 서로 다른 차원의 on-shell 좌표를 같은 cutoff로 놓은 $q_\times\le\Lambda_E$
Failure-Type: 차원
Removed-Claims: $\bar M\gtrsim(\kappa\rho_\infty)^{1/4}c_s^{-3/4}$ 및 현재 $\bar M\sim0.23\,{\rm GeV}$가 필요하다는 과대평가
Preserved-Objective: 배경을 보존하는 higher-spatial-derivative 항이 두-미분 strong coupling 전에 켜지는지 검증한다
Regression-Test: 코드에서 $\Lambda_E=c_sq_{\rm sc}$를 독립 assertion으로 검사하고 $q_\times$는 $q_{\rm sc}$와만 비교한다

올바른 필요조건은

$$
q_\times\le q_{\rm sc}=\Lambda_3c_s^{3/4},
\qquad
\bar M\gtrsim(\kappa\rho_\infty)^{1/4}c_s^{1/4}
$$

이다. 이 음성대조군은 $k^4$ 완성 자체를 증명하지 않으며, 이후 경로가 같은
에너지–파수 혼동을 되풀이하지 못하게 막는다.

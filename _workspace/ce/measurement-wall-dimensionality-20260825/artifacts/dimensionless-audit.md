# Dimensionless audit: finite-duration measurement wall

Status: COMPLETE

Dimension basis: $(M,L,T,\Theta)$. This audit checks dimensional consistency only; it does not establish a physical interpretation.

| expression or core argument | dimension vector | result | required convention |
|---|---:|---|---|
| $\hbar^{-1}\int_{t_0}^{t_1}H_{SA}(t)dt$ in $\exp(-i\cdot)$ | $(0,0,0,0)$ | valid | $[H]=ML^2T^{-2}$ and $[\hbar]=ML^2T^{-1}$ |
| $\Gamma(t)=\int_{t_0}^t\gamma(s)ds$ in $e^{-\Gamma}$ | $(0,0,0,0)$ | valid | $[\gamma]=T^{-1}$ |
| $\eta=1-e^{-\Gamma}$ | $(0,0,0,0)$ | valid | $0\le\eta\le1$ |
| $p_r$ in $\ln p_r$ | $(0,0,0,0)$ | valid | probabilities are dimensionless; use $0\ln0:=0$ by continuity |
| $C_I$, $\overline C_I$ | $(0,0,0,0)$ | valid | measured in nats by logarithm convention |
| $\dot\eta$ | $(0,0,-1,0)$ | rate, not energy | none |
| $C_{\rm wall}=\int\dot\eta\,\overline C_I\,dt$ | $(0,0,0,0)$ | valid | monotone differentiable wall witness |
| $\int C_I(t)dt$ | $(0,0,1,0)$ | time-valued, not energy | needs an independent scale before physical use |
| $k_BT\,D(\rho\|\gamma_T)$ | $(1,2,-2,0)$ | conditional energy bridge | requires bath temperature and Gibbs reference |
| $\hbar\dot C_I$ | $(1,2,-2,0)$ | dimensionally energy | dimensional validity alone does not justify this bridge |

Dimension status: the registered exponential, logarithm, probability, and wall-strength cores are dimensionless. No expression in the operational model supplies an energy density or stress tensor without an added physical scale and action.

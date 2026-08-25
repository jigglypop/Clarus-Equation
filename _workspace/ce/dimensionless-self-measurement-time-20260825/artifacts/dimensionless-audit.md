# Dimensionless audit: self-measurement depth

Status: COMPLETE

Dimension basis: $(M,L,T,\Theta)$. This audit establishes dimensional consistency only.

| expression | dimension vector | result | required convention |
|---|---:|---|---|
| $\eta$ and $1-\eta$ | $(0,0,0,0)$ | valid | channel mixing probability |
| $\theta=-\ln(1-\eta)$ | $(0,0,0,0)$ | valid | $0\leq\eta<1$ |
| $\delta\theta_k$ and $\sum_k\delta\theta_k$ | $(0,0,0,0)$ | valid | additive semigroup coordinate |
| $\theta(t)=\int\gamma(t)dt$ | $(0,0,0,0)$ | conditional | $[\gamma]=T^{-1}$; a clock/rate bridge is extra input |
| $p_a$ in $\ln p_a$ | $(0,0,0,0)$ | valid | $0\ln0:=0$ |
| $\overline C_I$ | $(0,0,0,0)$ | valid | nats |
| $e^{-\theta}d\theta$ | $(0,0,0,0)$ | valid | equals $d\eta$ |
| $C_{\rm self}$ | $(0,0,0,0)$ | valid | bounded information functional, not energy |
| $D_{\rm tr}(\rho,\sigma)$ | $(0,0,0,0)$ | valid | trace norm of density-operator difference |
| $v(\theta)=dD_{\rm tr}/d\theta$ | $(0,0,0,0)$ | valid | speed per dimensionless measurement depth |
| $v_t=dD_{\rm tr}/dt$ | $(0,0,-1,0)$ | rate, not energy | requires physical clock $t$ |
| $L=\int v(\theta)d\theta$ | $(0,0,0,0)$ | valid | state-space path length |
| $d\mu_{\rm self}$ before spatial pushforward | $(0,0,0,0)$ | valid | dimensionless weighted event measure |
| density representation $d\mu/d^3x$ | $(0,-3,0,0)$ | conditional | $F_\theta(a)$ must map into a length-valued space |
| $\epsilon_* C_{\rm self}$ | $(1,-1,-2,0)$ | dimensionally an energy density | requires independent $\epsilon_*$ and does not establish gravity |

The logarithms and exponentials pass the dimensionless gate. No physical time,
energy, energy density or stress tensor follows from $\theta$ or $C_{\rm self}$
without an independent rate/scale and a dynamical action.

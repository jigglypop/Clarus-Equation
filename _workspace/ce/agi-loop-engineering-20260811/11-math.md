# Mathematical lane

Status: COMPLETE

Loop 1 adds the missing local derivatives

\[
\frac{\partial \hat o_{t+1}}{\partial a_t}\ne0,
\qquad
\frac{\partial a_t^*}{\partial g^*}\ne0.
\]

The controller uses transition-internal action effects and a rank-1 posterior. Covariance changes the Kalman update and planning risk; it does not interpolate two completed forecasts.

Loop 2 separates signed task credit from signed homeostasis:

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
e_t=\gamma\lambda e_{t-1}+e_t^{local},
\qquad
\Delta W=\eta\delta_t e_t,
\]

\[
e_i^{homeo}=\bar f_i-f_i^*.
\]

The old squared bootstrap deviation cannot replace the last expression because it removes over/under direction.


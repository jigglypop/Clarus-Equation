# Dimensionless audit

Status: PASS

자연단위 $c=\hbar=1$과 $[x^\mu]=-1$, $[d^4x]=-4$를 사용한다.

| quantity | mass dimension | gate |
|---|---:|---|
| $\theta,\Theta,\psi,\lambda,x,y,q,N,a,E,u,c$ | 0 | PASS |
| $M_{\rm Pl},f,H$ | 1 | PASS |
| $\rho_*,V,\rho_i,p_i$ | 4 | PASS |
| $A=\rho_*/(3M_{\rm Pl}^2H_0^2)$ | 0 | PASS |
| $s=c_{\rm light}/(H_0r_d)$ in conventional BAO units | 0 | PASS |

지수와 로그의 인자는

$$
e^{-\theta},\quad e^{-\Theta},\quad
e^{-\lambda\phi/M_{\rm Pl}},\quad
e^{-3N},\quad e^{-4N},\quad \ln a
$$

로 모두 무차원이다. kinetic density는

$$
f^2(\nabla\Theta)^2
$$

이며 차원은 $2+2=4$, $M_{\rm Pl}^2R$도 $2+2=4$다. 따라서 integrand는
차원 4이고 action은 무차원이다.

$C_{\rm self}$와 $u,c$를 곧바로 에너지 밀도에 더하는 식은 차원 게이트를
통과하지 못한다. 반드시 차원 4의 독립 scale $\rho_*$가 필요하며, 이 감사는
그 scale의 값이나 기원을 제공하지 않는다.

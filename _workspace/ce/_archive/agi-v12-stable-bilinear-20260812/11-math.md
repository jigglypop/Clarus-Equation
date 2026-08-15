# Stability statement

Status: COMPLETE

For fixed input $x_t$, each state coordinate updates as

$$
h_{t+1}=(1-g(x_t))r h_t+g(x_t)\tilde h(x_t),
$$

where $0<r<0.995$ and $0<g<1$. Therefore the state Lipschitz factor is bounded by $r<0.995$.
The bilinear local/cloud term occurs only in the final policy readout and does not alter recurrent
stability. This theorem proves bounded fading memory, not task performance.

# Delay equation delta

Status: COMPLETE

For pre-step activation $a_t$, lifecycle eligibility $q_t$, and the STP state
after its tick update $(u_t^+,x_t^+)$, define the emitted packet

$$e_t=u_t^+\odot x_t^+\odot a_t\odot q_t.$$

For an $L$-slot ring with cursor $k_t$, the update is read-before-write:

$$d_t=B_t[k_t],\qquad r_t=W d_t,$$
$$B_{t+1}[k_t]=e_t,\qquad k_{t+1}=(k_t+1)\bmod L.$$

Thus $e_s$ contributes to recurrent drive exactly at call $s+L$, even if its
source becomes inactive in between. An inactive source emits zero and cannot
be retroactively activated at arrival. All quantities are normalized and
dimensionless. Snapshot continuation must preserve $B_t$ and the unbounded
cursor; reset sets $B=0,k=0$.

# Brain Clarus depth verification alignment

- passed: `True`
- minimal brain depth: `4`
- aligned required layers: 4/4
- optional depth-5 status: `candidate_boundary`

## required layers

| depth | layer | formal equation | observable proxy | prediction gate | ablation/countermodel | aligned |
|---:|---|---|---|---|---|---|
| 1 | `cellular_clarus_cell` | X_{t+1}=Pi_R[B,E,A,I,U,Q,D,S]; E_min>0.45, I_min>0.70, M_min>0.45, D_max<0.40, R>=2 | human multiscale state Y=(E,I,M,T,D,S,R) | full human proliferative/neural pass rates = 1.000 | no membrane/mitochondria/genome/traffic/repair/support/recurrence all <=0.25 | `True` |
| 2 | `tissue_support_field` | S_t enters both Clarus cell Pi_R and brain homeostatic H(q-q*) forcing | tissue/glia/vascular/metabolic support S_t; q_n homeostatic state | no_tissue_support collapses human cell closure; H(q-q*) is required brain forcing | no_tissue_support and no_homeostasis ablation | `True` |
| 3 | `neural_circuit_recurrence` | P_{n+1}=Pi_S[rho P_n+gamma L(W)P_n+...] | weighted chemical/effective graph W and neural activity P_t | L_dyn=MSE(P_{t+1}|P_t)/MSE(P_{t+1}|mean)<1; L_graph<L_flat | binary/flat/shuffled graph and recurrent-baseline countermodels | `True` |
| 4 | `organism_control_loop` | b_{n+1}=Omega(P_n,q_n,E_n); q_{n+1}=q_n+B(E)-C(b,P)-chi(q-q*) | behavior labels/traces, action carrier, body/internal state q_n | L_beh=MSE(y|P,q)/MSE(y|mean)<1 or discrete action carrier passes | no action carrier, timing-only, task/history, continuous alignment boundary | `True` |

## optional depth 5

- layer: `self_model_workspace`
- status: `candidate_boundary`
- formal equation: m_{n+1}=lambda m_n+Psi(P_n,b_n,r_n); W_{n+1}=Pi_W[W+epsilon Phi- mu W]
- reason: not yet locally closed; workspace ablation not available locally

## interpretation

The 4-depth brain claim matches the Clarus verification grammar.  Each required layer has a formal update/projection, a proxy, a prediction or closure gate, and an ablation/countermodel.  Depth 5 is formally writable but remains a workspace/self-model candidate boundary.

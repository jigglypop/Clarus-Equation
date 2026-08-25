# 31-validation — 기회비용 함수 검산

Status: COMPLETE

실행 명령:

```powershell
.codex\hooks\python.cmd python _workspace\ce\quantum-path-opportunity-functional-20260825\artifacts\verify_opportunity_cost.py
```

결과: 14개 검사 모두 통과했다.

주요 수치:

| 항목 | 값 |
|---|---:|
| $H(0.8,0.2)$ | $0.5004024235381879$ nat |
| $-\ln0.2$ | $1.6094379124341005$ nat |
| $-0.2\ln0.2$ | $0.32188758248682003$ nat |
| $D((0,1)\|(1/2,1/2))$ | $0.6931471805599453$ nat |
| $D((0.8,0.2)\|(1/2,1/2))$ | $0.1927447570217575$ nat |
| test energy gap $\Delta=3$의 expected regret | $0.6$ energy unit |
| thermal identity residual | $2.78\times10^{-17}$ energy unit |

차원 검사:

- information: $(0,0,0,0)$
- $k_BT\times$information: $(1,2,-2,0)$
- $\hbar\ln Z$: $(1,2,-1,0)$
- action/time: $(1,2,-2,0)$
- energy-density scale: $(1,-1,-2,0)$

허용오차는 $10^{-12}$였다. 이 결과는 algebra와 차원만 검증하며 실제 중력원을
측정하거나 검증하지 않는다.

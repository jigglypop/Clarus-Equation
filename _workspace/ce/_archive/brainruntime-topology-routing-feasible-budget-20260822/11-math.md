# Budget feasibility

Status: COMPLETE

Because $B\le\min_kM_k$, every cue-specific cluster/path candidate set has at
least $B$ edges. Therefore exact-budget selection is defined for `CLUSTER`,
`PATH_ONLY`, `TOPOLOGY`, and `RETURN_SHUFFLED`; global `WEIGHT` and
`RANDOM_MATCHED` are also feasible because their support contains the learned
graph. The budget is a dimensionless integer computed only from sealed
construction inputs. It changes sparsity strength, not route ranking.

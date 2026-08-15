# Development validation

Status: COMPLETE

## Pre-run

- focused kernel, benchmark, and dimensionless suite: `39 passed in 3.83s`;
- Ruff check: pass;
- Ruff format check: five files formatted;
- exact registration/hash/seed-role audit: pass;
- result path absent.

## Single development execution

- seeds: 64 fresh registered development seeds;
- per seed: 256 train plus 256 disjoint evaluation episodes;
- runtime: `69.9s`;
- result: `GO`;
- no rerun; a post-result runner call verified fail-closed `FileExistsError` before evaluation.

| arm | mean accuracy | mean effective ridge df | mean coefficient L2 |
|---|---:|---:|---:|
| full | 0.6530151367 | 20.9281535967 | 16.9724075859 |
| local only | 0.5017089844 | 20.8713538871 | 10.7697563937 |
| cloud only | 0.5084838867 | 20.8252997128 | 12.4270116157 |
| no memory | 0.5060424805 | 20.7885488898 | 13.5963523302 |

The strongest aggregate factorial control was cloud only. The more conservative per-seed
strongest-control difference was:

$$
0.1284790039,\qquad 95\%\ \mathrm{CI}=[0.1192001343,0.1373291016].
$$

The registered factorial interaction was:

$$
0.1488647461,\qquad 95\%\ \mathrm{CI}=[0.1389770508,0.1586303711].
$$

| intact-readout lesion | lesioned accuracy | paired loss | 95% CI |
|---|---:|---:|---:|
| cross cut on all ticks | 0.4951171875 | 0.1578979492 | [0.1514892578, 0.1647949219] |
| local reset at decision | 0.4977416992 | 0.1552734375 | [0.1476440430, 0.1625366211] |
| cloud reset at decision | 0.5681152344 | 0.0848999023 | [0.0792846680, 0.0906982422] |

All seven preregistered gates passed. All four integrity counters are zero. Independent raw-row
recomputation passed. Confirmation remains unopened.

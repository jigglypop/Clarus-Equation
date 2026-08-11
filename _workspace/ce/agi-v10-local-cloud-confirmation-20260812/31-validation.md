# Confirmation validation

Status: COMPLETE

Pre-run regression: `42 passed in 4.20s`; Ruff check and format pass. The one-shot confirmation
used 64 reserved seeds and completed in `80.6s` with `GO`.

| arm | confirmation accuracy | development accuracy |
|---|---:|---:|
| full | 0.6520996094 | 0.6530151367 |
| local only | 0.4957885742 | 0.5017089844 |
| cloud only | 0.4943847656 | 0.5084838867 |
| no memory | 0.4911499023 | 0.5060424805 |

Seed-paired full-minus-per-seed-strongest-control:

$$
0.1387939453,\qquad 95\%\ \mathrm{CI}=[0.1288436890,0.1497192383].
$$

Factorial interaction:

$$
0.1530761719,\qquad 95\%\ \mathrm{CI}=[0.1411117554,0.1654067993].
$$

| lesion | lesioned accuracy | paired loss | 95% CI |
|---|---:|---:|---:|
| cross cut | 0.4923095703 | 0.1597900391 | [0.1533187866, 0.1663818359] |
| decision local reset | 0.4942626953 | 0.1578369141 | [0.1493530273, 0.1662597656] |
| decision cloud reset | 0.5733032227 | 0.0787963867 | [0.0715942383, 0.0866104126] |

All seven gates pass and all four integrity counters are zero. Independent recomputation and
one-shot refusal pass. This confirms only the registered synthetic mechanism result.

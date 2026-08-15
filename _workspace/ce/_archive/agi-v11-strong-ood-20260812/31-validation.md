# V11 validation

Status: COMPLETE

The one-shot 16-seed development result is `STOP`.

| panel | V10 | Elman-3 | Elman-20 | GRU-20 |
|---|---:|---:|---:|---:|
| ID | 0.660156 | 0.755859 | 0.998779 | 0.998535 |
| noise | 0.593018 | 0.736084 | 0.997559 | 0.998291 |
| horizon | 0.540283 | 0.622803 | 0.787109 | 0.998047 |
| combined | 0.520264 | 0.604980 | 0.781982 | 0.997803 |

V10 minus each seed's stronger Elman-20/GRU-20 contrast:

| panel | mean | 95% CI |
|---|---:|---:|
| ID | -0.338623 | [-0.350098, -0.328369] |
| noise | -0.405762 | [-0.419678, -0.394525] |
| horizon | -0.457764 | [-0.472168, -0.444092] |
| combined | -0.477539 | [-0.493408, -0.462402] |

V10 also lost to compute-matched Elman-3 on every panel. ID and noise contrasts had wholly
negative intervals; horizon and combined means were negative and failed the positive-LCB gate.
V10 fell below the registered accuracy floor on horizon and combined. Ten primary gates failed.

GRU-20 Brier scores were `0.001413`, `0.001758`, `0.001953`, and `0.002197` across the four
panels, while V10's were `0.219473`, `0.243838`, `0.249331`, and `0.273717`.

All integrity counters are zero; raw-row independent recomputation and one-shot refusal pass.

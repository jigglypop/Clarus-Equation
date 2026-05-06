# CE Market Residual Index Gate

Purpose: compare CE residual models across market regimes.

Windows: short=21, medium=63, long=126, forward=21 trading days.

## Best Model By Case

| Asset/case | Best model | Type | Advantage | rho(vol) CE/base | rho(dd) CE/base | top/bottom dd |
|---|---|---|---:|---:|---:|---:|
| SYN_FLASH_CRASH | downside_cascade | downside cascade | +0.079 | 0.107/0.011 | 0.028/-0.034 | 0.015/0.015 |
| SYN_LIQUIDITY_GAP | downside_cascade | downside cascade | -0.078 | 0.246/0.238 | 0.151/0.315 | 0.032/0.011 |
| SYN_MEAN_REVERT | self_recursive | self-recursion | +0.056 | 0.097/0.024 | 0.171/0.132 | 0.026/0.021 |
| SYN_REGIME_SHIFT | equation_residual | direct equation | +0.059 | 0.312/0.249 | 0.330/0.276 | 0.061/0.013 |
| SYN_SLOW_BLEED | downside_cascade | downside cascade | -0.140 | 0.472/0.552 | 0.378/0.577 | 0.025/0.012 |
| SYN_SMOOTH_BULL | entropy_selection | entropy/selection | +0.076 | 0.084/0.024 | -0.057/-0.148 | 0.006/0.006 |
| SYN_TREND_BREAK | downside_cascade | downside cascade | -0.049 | 0.682/0.697 | 0.681/0.763 | 0.042/0.010 |
| SYN_VOL_CYCLE | hybrid_ce | combined | +0.135 | 0.863/0.746 | 0.704/0.551 | 0.040/0.011 |

## Full Model Matrix

| Asset/case | Model | Type | Current stress | Regime | Sel. | rho(vol) CE/base | rho(dd) CE/base | top/bottom vol | top/bottom dd | Advantage |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| SYN_FLASH_CRASH | downside_cascade | downside cascade | 8.98 | calm | 0.9146 | 0.107/0.011 | 0.028/-0.034 | 0.054/0.053 | 0.015/0.015 | +0.079 |
| SYN_FLASH_CRASH | equation_residual | direct equation | 30.52 | calm | 0.7096 | 0.048/0.011 | -0.035/-0.034 | 0.072/0.061 | 0.020/0.020 | +0.018 |
| SYN_FLASH_CRASH | self_recursive | self-recursion | 58.55 | unstable | 0.4430 | -0.057/0.011 | -0.003/-0.034 | 0.054/0.059 | 0.014/0.016 | -0.018 |
| SYN_FLASH_CRASH | multiscale_bridge | multi-scale | 30.70 | calm | 0.7080 | -0.076/0.011 | -0.086/-0.034 | 0.065/0.070 | 0.018/0.023 | -0.069 |
| SYN_FLASH_CRASH | hybrid_ce | combined | 36.54 | watch | 0.6523 | -0.133/0.011 | -0.055/-0.034 | 0.055/0.069 | 0.015/0.023 | -0.082 |
| SYN_FLASH_CRASH | entropy_selection | entropy/selection | 11.54 | calm | 0.8902 | -0.137/0.011 | -0.128/-0.034 | 0.054/0.062 | 0.013/0.019 | -0.121 |
| SYN_LIQUIDITY_GAP | downside_cascade | downside cascade | 35.76 | watch | 0.6598 | 0.246/0.238 | 0.151/0.315 | 0.099/0.049 | 0.032/0.011 | -0.078 |
| SYN_LIQUIDITY_GAP | equation_residual | direct equation | 56.40 | unstable | 0.4635 | 0.130/0.238 | 0.214/0.315 | 0.087/0.053 | 0.029/0.013 | -0.104 |
| SYN_LIQUIDITY_GAP | hybrid_ce | combined | 48.52 | watch | 0.5384 | 0.045/0.238 | 0.264/0.315 | 0.099/0.053 | 0.034/0.011 | -0.122 |
| SYN_LIQUIDITY_GAP | self_recursive | self-recursion | 62.49 | unstable | 0.4055 | 0.025/0.238 | 0.279/0.315 | 0.095/0.054 | 0.032/0.012 | -0.125 |
| SYN_LIQUIDITY_GAP | multiscale_bridge | multi-scale | 54.19 | watch | 0.4844 | 0.053/0.238 | 0.242/0.315 | 0.093/0.053 | 0.032/0.011 | -0.129 |
| SYN_LIQUIDITY_GAP | entropy_selection | entropy/selection | 47.43 | watch | 0.5488 | 0.112/0.238 | 0.175/0.315 | 0.097/0.051 | 0.033/0.013 | -0.133 |
| SYN_MEAN_REVERT | self_recursive | self-recursion | 58.70 | unstable | 0.4415 | 0.097/0.024 | 0.171/0.132 | 0.091/0.090 | 0.026/0.021 | +0.056 |
| SYN_MEAN_REVERT | hybrid_ce | combined | 40.19 | watch | 0.6177 | 0.029/0.024 | 0.129/0.132 | 0.091/0.091 | 0.027/0.023 | +0.001 |
| SYN_MEAN_REVERT | equation_residual | direct equation | 38.48 | watch | 0.6339 | 0.052/0.024 | 0.060/0.132 | 0.090/0.090 | 0.026/0.023 | -0.022 |
| SYN_MEAN_REVERT | multiscale_bridge | multi-scale | 27.26 | calm | 0.7406 | 0.004/0.024 | 0.105/0.132 | 0.089/0.090 | 0.026/0.026 | -0.023 |
| SYN_MEAN_REVERT | entropy_selection | entropy/selection | 20.74 | calm | 0.8027 | -0.036/0.024 | 0.001/0.132 | 0.089/0.089 | 0.025/0.026 | -0.095 |
| SYN_MEAN_REVERT | downside_cascade | downside cascade | 32.42 | calm | 0.6916 | -0.058/0.024 | 0.023/0.132 | 0.089/0.090 | 0.023/0.023 | -0.095 |
| SYN_REGIME_SHIFT | equation_residual | direct equation | 35.90 | watch | 0.6585 | 0.312/0.249 | 0.330/0.276 | 0.147/0.051 | 0.061/0.013 | +0.059 |
| SYN_REGIME_SHIFT | multiscale_bridge | multi-scale | 5.16 | calm | 0.9509 | 0.276/0.249 | 0.233/0.276 | 0.146/0.052 | 0.063/0.016 | -0.008 |
| SYN_REGIME_SHIFT | self_recursive | self-recursion | 46.91 | watch | 0.5538 | 0.327/0.249 | 0.164/0.276 | 0.142/0.049 | 0.068/0.017 | -0.017 |
| SYN_REGIME_SHIFT | hybrid_ce | combined | 33.58 | calm | 0.6805 | 0.256/0.249 | 0.130/0.276 | 0.144/0.051 | 0.064/0.017 | -0.069 |
| SYN_REGIME_SHIFT | entropy_selection | entropy/selection | 19.89 | calm | 0.8107 | 0.112/0.249 | 0.061/0.276 | 0.082/0.054 | 0.024/0.017 | -0.176 |
| SYN_REGIME_SHIFT | downside_cascade | downside cascade | 38.03 | watch | 0.6382 | -0.049/0.249 | -0.003/0.276 | 0.073/0.058 | 0.028/0.017 | -0.289 |
| SYN_SLOW_BLEED | downside_cascade | downside cascade | 6.48 | calm | 0.9383 | 0.472/0.552 | 0.378/0.577 | 0.072/0.047 | 0.025/0.012 | -0.140 |
| SYN_SLOW_BLEED | hybrid_ce | combined | 37.66 | watch | 0.6417 | 0.417/0.552 | 0.303/0.577 | 0.075/0.050 | 0.024/0.015 | -0.205 |
| SYN_SLOW_BLEED | self_recursive | self-recursion | 60.22 | unstable | 0.4270 | 0.384/0.552 | 0.252/0.577 | 0.072/0.050 | 0.023/0.015 | -0.247 |
| SYN_SLOW_BLEED | multiscale_bridge | multi-scale | 21.27 | calm | 0.7976 | 0.325/0.552 | 0.246/0.577 | 0.072/0.050 | 0.022/0.014 | -0.279 |
| SYN_SLOW_BLEED | equation_residual | direct equation | 36.57 | watch | 0.6521 | 0.177/0.552 | 0.116/0.577 | 0.068/0.054 | 0.021/0.016 | -0.418 |
| SYN_SLOW_BLEED | entropy_selection | entropy/selection | 23.53 | calm | 0.7762 | 0.061/0.552 | 0.038/0.577 | 0.061/0.056 | 0.017/0.017 | -0.515 |
| SYN_SMOOTH_BULL | entropy_selection | entropy/selection | 35.99 | watch | 0.6576 | 0.084/0.024 | -0.057/-0.148 | 0.034/0.034 | 0.006/0.006 | +0.076 |
| SYN_SMOOTH_BULL | equation_residual | direct equation | 67.71 | unstable | 0.3558 | 0.050/0.024 | -0.037/-0.148 | 0.035/0.034 | 0.005/0.006 | +0.069 |
| SYN_SMOOTH_BULL | multiscale_bridge | multi-scale | 55.67 | unstable | 0.4704 | 0.045/0.024 | -0.119/-0.148 | 0.036/0.035 | 0.005/0.007 | +0.025 |
| SYN_SMOOTH_BULL | hybrid_ce | combined | 54.56 | watch | 0.4809 | 0.004/0.024 | -0.125/-0.148 | 0.035/0.036 | 0.006/0.007 | +0.002 |
| SYN_SMOOTH_BULL | downside_cascade | downside cascade | 22.96 | calm | 0.7816 | -0.073/0.024 | -0.048/-0.148 | 0.034/0.034 | 0.006/0.006 | +0.002 |
| SYN_SMOOTH_BULL | self_recursive | self-recursion | 67.17 | unstable | 0.3609 | -0.038/0.024 | -0.107/-0.148 | 0.036/0.037 | 0.006/0.007 | -0.011 |
| SYN_TREND_BREAK | downside_cascade | downside cascade | 60.68 | unstable | 0.4227 | 0.682/0.697 | 0.681/0.763 | 0.098/0.054 | 0.042/0.010 | -0.049 |
| SYN_TREND_BREAK | hybrid_ce | combined | 54.68 | watch | 0.4798 | 0.472/0.697 | 0.525/0.763 | 0.096/0.054 | 0.039/0.010 | -0.232 |
| SYN_TREND_BREAK | self_recursive | self-recursion | 70.28 | unstable | 0.3314 | 0.329/0.697 | 0.392/0.763 | 0.091/0.061 | 0.037/0.017 | -0.369 |
| SYN_TREND_BREAK | multiscale_bridge | multi-scale | 52.99 | watch | 0.4958 | 0.329/0.697 | 0.384/0.763 | 0.092/0.064 | 0.038/0.016 | -0.374 |
| SYN_TREND_BREAK | entropy_selection | entropy/selection | 26.25 | calm | 0.7503 | 0.133/0.697 | 0.182/0.763 | 0.088/0.072 | 0.038/0.021 | -0.573 |
| SYN_TREND_BREAK | equation_residual | direct equation | 48.38 | watch | 0.5397 | 0.082/0.697 | 0.131/0.763 | 0.089/0.080 | 0.035/0.026 | -0.624 |
| SYN_VOL_CYCLE | hybrid_ce | combined | 46.11 | watch | 0.5613 | 0.863/0.746 | 0.704/0.551 | 0.133/0.045 | 0.040/0.011 | +0.135 |
| SYN_VOL_CYCLE | multiscale_bridge | multi-scale | 30.50 | calm | 0.7098 | 0.849/0.746 | 0.695/0.551 | 0.132/0.048 | 0.040/0.012 | +0.124 |
| SYN_VOL_CYCLE | self_recursive | self-recursion | 62.56 | unstable | 0.4048 | 0.843/0.746 | 0.693/0.551 | 0.132/0.046 | 0.042/0.011 | +0.119 |
| SYN_VOL_CYCLE | equation_residual | direct equation | 46.03 | watch | 0.5621 | 0.809/0.746 | 0.657/0.551 | 0.133/0.054 | 0.037/0.012 | +0.084 |
| SYN_VOL_CYCLE | entropy_selection | entropy/selection | 49.84 | watch | 0.5258 | 0.252/0.746 | 0.180/0.551 | 0.114/0.084 | 0.029/0.018 | -0.432 |
| SYN_VOL_CYCLE | downside_cascade | downside cascade | 12.54 | calm | 0.8807 | -0.177/0.746 | -0.154/0.551 | 0.096/0.119 | 0.025/0.033 | -0.814 |

Reading:
- equation_residual tests the direct residual equation.
- self_recursive tests whether stress memory and failed recovery matter.
- entropy_selection tests selection failure through sign entropy and liquidity stress.
- downside_cascade tests drawdown-specific cascade risk.
- hybrid_ce combines the above with a recursive memory state.
- This is a risk/regime gate, not a trade recommendation system.

# Mathematical ledger

Status: COMPLETE

The predecessor proves a global weighted-block-sup contraction for the declared bounded map.
This run does not change that map. The empirical estimands are paired per seed:

$$
\Delta_s=A_s^{\mathrm{full}}-\max\left(
A_s^{\mathrm{local}},A_s^{\mathrm{cloud}},A_s^{\mathrm{none}}
\right),
$$

and the factorial interaction

$$
I_s=A_s^{\mathrm{full}}-A_s^{\mathrm{local}}
-A_s^{\mathrm{cloud}}+A_s^{\mathrm{none}}.
$$

Bootstrap resampling is over seed rows, never individual episodes. Lesion losses are paired
intact-full accuracy minus the accuracy from lesioned features under the same frozen full
readout. Mechanical confidence intervals do not upgrade the result beyond this task.

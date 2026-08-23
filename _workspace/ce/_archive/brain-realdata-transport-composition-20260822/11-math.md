# Mathematical check

Status: COMPLETE

All fitted neural coordinates are dimensionless: released dF/F is a ratio and
the train-only chart standardizes each retained ROI before PCA. The affine
coefficients, ridge, SSE ratios, $G$, and operator discrepancy are therefore
dimensionless. Seconds are used only to choose frozen frame windows; no
dimensioned quantity is passed to an exponential, logarithm, or probability.

With row vectors, the composed linear part is $A_{01}A_{12}$ and the composed
intercept is $b_{01}A_{12}+b_{12}$. This is the row-vector equivalent of the
column-vector expression $A_{12}A_{01}x+A_{12}b_{01}+b_{12}$. Comparing only
raw matrices is insufficient because their scale depends on the fitted chart;
held-out prediction is primary. The auxiliary covariance-weighted discrepancy
uses train states only.

The direct two-step map is the decisive matched-state comparator. A composed
map can beat persistence yet fail closure if the direct map is materially
better. Trial derangement detects mere phase means; the intermediate
coordinate permutation detects loss of packet/column correspondence. A pass
still admits smooth calcium autocorrelation, common drive, and low-dimensional
state stabilization as alternative explanations.

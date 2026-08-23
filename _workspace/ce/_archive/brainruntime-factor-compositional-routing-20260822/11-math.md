# Mathematics

Status: COMPLETE

## Frequency obstruction

Let $n_{00},n_{01},n_{10}>0$ be the counts of the three observed joint contexts. Equal A marginals require $n_{00}+n_{01}=n_{10}$, while equal B marginals require $n_{00}+n_{10}=n_{01}$. Subtraction gives $2n_{00}=0$, contradicting $n_{00}>0$. Therefore no positive schedule excluding `11` can balance both factor marginals. Raw Hebbian sums mix context frequency with branch association.

The count-normalized estimator removes that deterministic scale:

$$
\Theta^F_{:,x}=\frac{\sum_{n:x_n=x}u^{F,(n)}}{\#\{n:x_n=x\}}.
$$

It is an empirical mean over local branch-use receipts, not an assertion of unbiased biological estimation.

## Types and dimensions

$E^F,u^F,q^F,C^F,n^F,\Theta^F$, logits, normalized runtime activations, and recurrent weights are dimensionless. Counts and mask cardinalities are unitless. Division by $n_x^F$ is legal only after the positive-count preflight. The argmax and mask are discrete; no derivative through them is claimed. Runtime-energy remains a dimensionless simulator proxy.

The direct product makes joint success and joint route exact conjunctions of independent factor outcomes. It does not model cross-factor interaction. Separate $Y^A,Y^B$ and decoders are necessary: a shared output would permit one factor to affect the other's endpoint.

## Identifiability limits

Three observed joint rows update all four factor columns but only three of four joint-lookup columns. Thus the factor model can produce a defined `11` action while a pure one-hot joint table must abstain. This distinguishes the registered direct-sum representation from joint memorization inside this fixture. Because factorization and candidate branch families are supplied by design, it does not establish that a system discovered those factors or supports.

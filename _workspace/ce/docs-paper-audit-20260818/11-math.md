# Mathematics and dependency audit

Status: COMPLETE

Scope: `docs/**/*.md` as a document collection, with deep checking of
`docs/5_유도/00_선택과_접힘.md`, `docs/코어_독자_가이드.md`, `docs/axium.md`,
`docs/경로적분.md`, the cosmology ledger, and the referenced proof ledger.
This is not an assertion that the 167 Markdown files constitute one paper.

## Result first

The closed conditional core is mathematically coherent under its stated domains:

$$
\alpha_s\longrightarrow s_W^2\longrightarrow\delta\longrightarrow D
\longrightarrow q_{\rm ext}
$$

and the `C-B-LEGACY-01` density readout is arithmetically self-consistent once it
is adopted. The audit found no P0 counterexample to that conditional core.
The two initially found local wording/proof issues are closed by Revision 1;
the remaining P1/P2 structure defects can still make it too easy for a new
reader to confuse a theorem, an adopted physical map, and an archival ledger
entry.

Independent numerical details are in [math-spot-check.md](artifacts/math-spot-check.md).

## Reconstructed dependency chain

1. The Standard-Model neutral mass matrix, with a chosen basis, gives the
   dimensionless quantity $\delta=\sin^2\theta_W\cos^2\theta_W$. The range
   $0\le\delta\le1/4$ is a conditional theorem.
2. The identification of the nontrivial Hodge case with physical spatial
   dimension $d=3$, and the additive fold operator giving $D=d+\delta$,
   are model/physical-map axioms plus direct algebraic outputs. They are not
   consequences of the Hodge theorem or electroweak theory.
3. For an independently stipulated Poisson offspring model with $D>1$, the
   minimal extinction probability is the small solution of
   $q=e^{-D(1-q)}$. Its existence, minimality, Lambert-W form, and local
   stability are conditional theorems.
4. The maps from quantum amplitudes to nonnegative offspring rates, from a
   nonselected residual to a field, and from $q$ to cosmological densities
   are separately marked incomplete or axiomatic. They do not follow from the
   fixed-point theorem.

This separation is present and is the strongest feature of the core documents.

## Domain and dimensional audit

- The exponent $-D(1-q)$ is dimensionless: $D$ is a mean offspring number
  and $q$ is a probability. The declared $\alpha_s$, $s_W^2$, and
  $\delta$ are dimensionless.
- The scalar fixed-point theorem explicitly requires $D>1$; the supplied
  value is $D=3.17775842340997$. The small root lies in $(0,1/D)$, and
  $Dq=0.154587523120074<1$, as required for local contraction.
- The closure calculation additionally assumes a flat late-time model and
  neglects radiation and curvature. It is not a present-day stress-energy
  derivation. The documents state this assumption; it must remain adjacent to
  every displayed $\Omega$ output.
- $\chi A$ in the Euclidean auxiliary semigroup is dimensionless only after
  assigning $[\chi]=[A]^{-1}$, which the core explicitly does. No physical
  time interpretation follows.

## Findings

### P0 — none found in the audited core

No complete counterexample was found against the stated scalar Poisson theorem,
its Lambert-W branch choice, the Hodge $2$-form/$1$-form conditional theorem,
or the arithmetic density partition. The numerical chain reproduces the ledger.

### CLOSED P1-1 — convergence statement now matches its displayed proof

`00_선택과_접힘.md` §0.5.1 now restricts the Banach-contraction conclusion
to $x_0\in[0,1/D]$. This exactly matches the proof ledger's self-map and
contraction domain $[0,1/D]$. No global-convergence assertion is left to be
justified there; the former proof-to-statement gap is closed.

### CLOSED P1-2 — residual object now matches its supplied definition

`00_선택과_접힘.md` §0.5.2 now calls $\mu_{\rm ns}$ a residual measure
restricted to the nonselected candidates. This matches P5 in
`9_등호이전/01_공리와증명.md`: the object may be unnormalised or zero. The
remaining physical identification with a field and with the branching survival
fraction remains explicitly `[미완성]`; no unsupported normalized-probability
claim remains.

### P1-3 — role collision can blur formal status

The same writing workflow currently serves both ledger/audit prose and
paper/lecture prose. The audit found the core labels generally sound, but this
role collision creates a concrete propagation risk: a ledger table can preserve
a status while a narrative cross-reference turns it into an apparently flowing
derivation. The $q\to\Omega_b$ example is protected in the core, but this
must be mechanically checked at handoff boundaries. Separate ownership for
ledger normalization and paper narration is therefore a P1 documentation-control
requirement, not a mathematical counterexample.

### P2-1 — document-family topology is not a paper topology

There are 167 Markdown files across lectures, core theory, engineering,
neuroscience, AGI, Riemann material, pre-equality material, references, ledgers,
and audits. Many files have no formal-status tags; that is acceptable for notes
or specifications, but means the entire tree cannot be rendered as a single
paper merely by applying a universal tag block. Readers need an explicit
document-type and intended-reader declaration before a common style gate runs.

### P2-2 — stale internal research links

The local-link audit found 17 missing links, all in
`docs/7_AGI/28_Nested_Infinite_SCC_V9.md`, to absent `_workspace/ce/...` run
artifacts. These are reproducibility/navigation defects, not evidence against
the mathematics. Archived-run citations should be represented as a stable
provenance reference or explicitly marked unavailable.

### P2-3 — the focused implementation regression was unavailable

`python -m pytest tests/test_bootstrap_solver.py -q -p no:cacheprovider` could
not collect because the available Python 3.14 environment lacks `torch`; the
repository `.venv` executable is absent. The independent calculation passed,
but the harness should make the minimal mathematical check runnable without the
ML dependency or declare the required environment.

## Reproduction

The spot calculation uses only the formulas stated above and is recorded in
`artifacts/math-spot-check.md`. The attempted focused regression command was:

```text
python -m pytest tests/test_bootstrap_solver.py -q -p no:cacheprovider --basetemp <temporary-directory>
```

It failed during collection with `ModuleNotFoundError: No module named 'torch'`.

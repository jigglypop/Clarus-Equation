# H0 source-role readout paper draft

Status: **post-hoc calibration draft; not submission-ready**. The package gate
reproduces tables and classifications but does not establish physical or
statistical validation.

## Working title

A post-hoc source-role conductance diagnostic for Hubble-constant channels.

## Abstract draft

The Hubble tension is usually framed as a disagreement between inferred values
of the same cosmological parameter. We describe an exploratory source-role
conductance diagnostic built from Fisher or covariance structure. Under the
declared map, selected time-delay-lensing, standard-ruler, CMB,
distance-ladder, and standard-siren channels are replayed as global/low,
bridge/intermediate, or local/high families. The map and its endpoint law were
constructed after the relevant channel structures and published \(H_0\) values
were known. Static-role ablations and threshold sweeps therefore test only
internal sensitivity of this calibration; they are not a model-selection test.
No full joint posterior refit, blinded role assignment, or untouched-channel
holdout has been performed. The present result is a reproducible hypothesis
generator for a future preregistered test, not evidence that source role
physically explains the Hubble tension.

## Plain-language significance

The proposal is not that one experiment is simply wrong and another experiment
is simply right. The proposal is that the measured H0 value can depend on where
the observation is anchored in the cosmic inference graph.

CMB and BAO look across the whole cosmic ruler system. They compare distant
structure to an early-universe or large-scale standard ruler, so they behave
like global closure measurements. Distance ladders start from nearby anchors
and climb outward through calibrated objects, so they behave like local
endpoint measurements. Standard sirens naturally sit between those cases:
gravitational waves give an absolute distance, while the host galaxy or
redshift environment supplies the cosmic anchor.

The diagnostic asks whether the same universe could be summarized through
different closures. Whether this is a physical explanation, rather than a
post-hoc classification of known channels, remains an open empirical question.

## Calibration hypothesis

The declared map encodes the following hypothesis:

- global closure channels read a low H0 branch,
- local endpoint channels read a high H0 branch,
- mixed source channels read an intermediate bridge branch.

Within each replay run the source role is loaded before the numeric readout is
computed. Historically, however, the role ontology and endpoints were designed
after the published high/low pattern was known. The current result therefore is
a post-hoc grouping. Only a frozen automatic role extractor applied to an
untouched channel can change that status.

## Methods: source-role conductance

Each observational channel is represented by a labelled Fisher or covariance
payload. If the input is a covariance matrix, it is inverted to a Fisher matrix.
For nodes \(i,j\), we define a normalized edge reliability

\[
r_{ij} = \frac{|F_{ij}|}{\sqrt{F_{ii}F_{jj}}}.
\]

The observable node is connected to two source-role sets:

- \(L\): local endpoint nodes,
- \(G\): global closure nodes.

For direct readout channels, conductance is the sum of normalized edges from
the observable to the target set:

\[
C_L = \sum_{j\in L} r_{oj},\qquad
C_G = \sum_{j\in G} r_{oj}.
\]

For channels where indirect dependencies matter, the path version sums over
paths from the observable to the target set with reliability products weighted
by path depth. The same selector is then used:

\[
q_F = \frac{C_L}{C_L+C_G}.
\]

Thus \(q_F=0\) is a global closure readout, \(q_F=1\) is a local endpoint
readout, and intermediate values are bridge readouts.

The branch value is computed by applying the selector to the two CE readout
endpoints:

\[
H_0(q_F) = H_0\!\left(\log S_{\rm global} - q_F \Delta_{\rm endpoint}\right).
\]

The numeric contract used throughout this draft is

\[
N_e=57.20243399,\qquad \delta_N\sigma_D=0.16925962,
\qquad
H_0(q_F)=66.802746\exp\!\left(\frac{q_F\delta_N\sigma_D}{2}\right),
\]

with \(H_0(0)=66.802746\) and \(H_0(1)=72.702371\) in
\({\rm km\,s^{-1}Mpc^{-1}}\).

The future methodological constraint is that \(L\), \(G\), all thresholds and
both endpoint coefficients must be frozen from source provenance alone before
opening an untouched channel's \(H_0\) result. Existing rows do not satisfy that
historical blinding condition.

## Data provenance

Each row in the readout table must be traceable to a public source and a
reproducible gate. The current provenance map is:

| channel | source role | public source | primary gate | status |
|---|---|---|---|---|
| TDCOSMO-only | local time-delay lens endpoint | public TDCOSMO chain payload plus notebook factor extraction | `h0_tdcosmo_notebook_factor_extract_gate.py` | data-facing |
| TDCOSMO+IFU | local time-delay lens endpoint with IFU kinematic closure | public TDCOSMO chain payload plus notebook factor extraction | `h0_tdcosmo_role_transition_gate.py` | data-facing |
| TDCOSMO+SLACS | global population closure | public TDCOSMO chain payload plus SLACS likelihood factor | `h0_tdcosmo_role_transition_gate.py` | data-facing |
| TDCOSMO+SLACS+IFU | global population closure with IFU kinematic closure | public TDCOSMO chain payload plus SLACS likelihood factor | `h0_tdcosmo_role_transition_gate.py` | data-facing |
| DESI BAO | global standard-ruler closure | `CobayaSampler/bao_data` DESI 2024 mean/covariance | `h0_bao_mean_cov_role_adapter_gate.py` | data-facing |
| Planck CMB | early global acoustic-horizon closure | IRSA Planck PR3 cosmological parameter covariance | `h0_cmb_planck_covariance_adapter_gate.py` | data-facing |
| Pantheon+SH0ES | local distance-ladder endpoint | `PantheonPlusSH0ES/DataRelease` distance table and covariance | `h0_pantheon_shoes_role_adapter_gate.py` | data-facing |
| GW170817 bright siren | bridge distance-redshift anchor | published GW170817 H0 reference plus LVK provenance record | `h0_gw_standard_siren_bridge_gate.py` | scoped bridge abstraction |

This table should appear before the main result tables. It makes clear which
rows are already covariance-backed and which row is currently a scoped bridge
abstraction.

## Figure package

### Figure 1: endpoint source-role split

Purpose: show the Hubble tension split itself.

Rows:

| readout | channels |
|---|---|
| global/low | DESI BAO, Planck CMB, TDCOSMO+SLACS, TDCOSMO+SLACS+IFU |
| local/high | Pantheon+SH0ES, TDCOSMO+IFU, TDCOSMO-only |

Claim supported:

> Under the post-hoc role map, selected Hubble-tension channels replay as
> local/high and global/low families.

Caption: Figure 1 visualizes the endpoint split of the H0 source-role readout.
Each point is placed by the declared source-role selector. The selector was
designed after the channel context was known, so the figure is a calibration
diagram rather than a blinded branch-selection test. The local/high endpoint contains
distance-ladder and time-delay-only channels whose closure is anchored by local
endpoint information. The global/low endpoint contains BAO, CMB, and
population-closure channels whose closure is anchored by global standard-ruler
or hierarchy information. It becomes a branch-selection test only after the
complete role rule is frozen and applied to untouched data.

### Figure 2: three-family readout law

Purpose: show that the rule is not only a binary classifier.

Rows:

| readout | channels |
|---|---|
| global/low | DESI BAO, Planck CMB, TDCOSMO+SLACS, TDCOSMO+SLACS+IFU |
| bridge/intermediate | GW170817 bright standard siren |
| local/high | Pantheon+SH0ES, TDCOSMO+IFU, TDCOSMO-only |

Claim supported:

> Source-role conductance admits global, bridge, and local readout families.

Caption: Figure 2 extends the same selector from the two endpoint families to a
three-family readout. GW170817-like bright standard sirens occupy the
bridge/intermediate family because the luminosity-distance information and
host-redshift anchoring close the measurement from different sides of the
inference graph. The figure should be read as a source-role readout diagram,
not a joint posterior fit. Its purpose is to show that the same conductance law
can represent global/low, bridge/intermediate, and local/high closures.

## Numeric results

The numeric table should be reported with scope. Some rows have an attached H0
reference and pull; some rows are branch-only source-role checks and should not
be presented as posterior refits.

| channel | source role | selector | readout | H0 readout | reference status |
|---|---|---:|---|---:|---|
| DESI BAO | global | 0.000000 | global/low | 66.802746 | branch-only |
| Planck CMB | global | 0.000000 | global/low | 66.802746 | Planck covariance H0 attached |
| TDCOSMO+SLACS+IFU | global | 0.001855 | global/low | 66.813234 | chain H0 attached |
| TDCOSMO+SLACS | global | 0.003704 | global/low | 66.823690 | chain H0 attached |
| GW170817 bright siren | bridge | 0.500000 | bridge/intermediate | 69.690157 | scoped reference H0 attached |
| TDCOSMO-only | local | 0.830134 | local/high | 71.664698 | chain H0 attached |
| TDCOSMO+IFU | local | 0.852221 | local/high | 71.798780 | chain H0 attached |
| Pantheon+SH0ES | local | 1.000000 | local/high | 72.702371 | branch-only |

This table is neither a joint \(H_0\) fit nor an independent validation table.
It records the ordering produced by a post-hoc branch-readout calibration.

## Results narrative

### TDCOSMO

TDCOSMO is the internal transition test. Without SLACS population closure, the
time-delay lensing chains retain local mass-sheet endpoint conductance and
select the high branch. With SLACS population closure, the same lensing family
is pushed into global closure conductance and selects the low branch.

This is important because it is not a comparison between unrelated experiments.
It is a role transition inside one observational family.

### BAO and CMB

DESI BAO and Planck CMB are both global closure channels. BAO uses a standard
ruler over cosmic distance ratios. CMB uses the acoustic angle and early
horizon/distance-to-recombination closure. Both select the global/low branch.

Planck is now backed by a PR3 parameter covariance adapter, not merely a
synthetic role model.

### Pantheon+SH0ES

Pantheon+SH0ES is a local endpoint channel. Its calibrator and Hubble-flow
source labels form a local distance ladder, so it selects the local/high branch.

### GW standard sirens

GW170817-like bright standard sirens are bridge channels. GW amplitude gives an
absolute luminosity distance, while the electromagnetic counterpart and host
environment provide the redshift anchor. This balance selects an intermediate
branch.

## Ablations

Within the declared calibration, the split is not reproduced by assigning every
channel to one fixed role:

- all-local maps fail on global channels,
- all-global maps fail on local channels,
- flipped maps fail on every channel.

Threshold sweeps from broad local/global cuts also preserve the endpoint split.
This checks local threshold sensitivity, but it cannot remove the earlier
post-hoc choice of role ontology, partitions and endpoint law.

## Reviewer objections and safeguards

### Objection 1: the families were assigned after seeing H0

Current answer: this objection is valid. The assignments have defensible
source-provenance descriptions, but the ontology was not preregistered or
blinded. The required safeguard is a versioned automatic extractor and an
untouched release; provenance arguments alone do not supply independence.

### Objection 2: local/global assignment is arbitrary

Current result: static and flipped role ablations show that outputs depend on
the source-aware assignment. They do not prove that the assignment is unique or
non-arbitrary. Competing preregistered role maps and out-of-sample comparison
are still required.

### Objection 3: the classifier threshold was tuned

Current result: the endpoint replay survives the tested threshold sweep. This
is robustness conditional on the chosen features and role map, not protection
against post-hoc feature engineering.

### Objection 4: this is not a full cosmological inference

The draft states this explicitly. The current result is a calibration
diagnostic, not yet a validated branch-selection law, and does not replace full
posterior analyses or likelihood optimization.

### Objection 5: the GW bridge result is still weak

Safeguard: the GW result is marked as bridge evidence with scope. It currently
uses a source-role covariance abstraction and must later be replaced by
event-level posterior samples.

## Predictions and falsification

The paper should make clear predictions before adding more channels. The
central rule is simple: source roles are fixed first, then the H0 family is
read out. If the family only appears after tuning to the published H0 value,
the claim fails.

| target | expected source-role result | what would support the claim | what would weaken or falsify it |
|---|---|---|---|
| GW event-level posterior samples | bridge/intermediate | luminosity-distance information and host-redshift anchoring remain balanced after source roles fixed | event-level samples collapse to a stable global/low or local/high endpoint |
| TRGB/JAGB/CCHP local ladders | local/high or semi-local high | calibrator, anchor, and Hubble-flow labels keep local endpoint conductance dominant | a source-aware ladder map selects global/low without an added population closure |
| BAO+SN inverse-distance ladder | global/low | standard-ruler closure dominates even after SN covariance is attached | the joint covariance selects local/high before any H0 refit |
| CMB covariance variants | global/low | lensing and non-lensing Planck-like covariance adapters stay near the global endpoint | CMB source-role covariance becomes bridge or local/high under the same acoustic map |
| Alternative TDCOSMO notebooks and chains | role-transition split | adding SLACS population closure lowers q_F relative to time-delay-only chains | static role maps explain the rows as well as the source-aware map |

This is where the proposal becomes testable. It is not enough to say that the
existing rows can be arranged into a pleasing story. The next data products
must either keep the same role-to-family behavior or expose the point where the
readout law breaks.

## Required limitations

These limitations must stay in the paper:

1. We have not yet performed a full joint BAO/SN/TDCOSMO posterior refit.
2. The GW bridge gate uses a source-role covariance abstraction, not event-level
   posterior samples yet.
3. The CMB gate reads Planck PR3 parameter covariance, not a fresh Planck
   likelihood optimization.
4. The role ontology and endpoint law were constructed post hoc; current rows
   are not holdouts.
5. The diagnostic is not a replacement for standard cosmological inference
   machinery and is not yet a validated physical law.

## What the paper gains

The present gain is a reproducible schema for expressing a possible structural
explanation. Under the calibration map:

- low branch measurements are global closures,
- high branch measurements are local endpoints,
- intermediate measurements are mixed source-role bridges.

This motivates, but does not answer, the question: can source role predict a
channel family after the rule is frozen and before its \(H_0\) result is seen?

## Next tests

1. Replace the GW bridge abstraction with event-level posterior samples.
2. Add TRGB/JAGB/CCHP distance-ladder variants as local or semi-local channels.
3. Add inverse-distance-ladder BAO+SN joint covariance as a stronger global
   channel.
4. Build a full joint posterior comparison only after the source-role map is
   fixed.

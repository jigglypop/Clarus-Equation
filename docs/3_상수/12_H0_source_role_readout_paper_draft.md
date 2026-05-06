# H0 source-role readout paper draft

Status: draft spine, backed by `examples/physics/h0_readout/h0_paper_package_gate.py`.

## Working title

Source-role conductance separates Hubble-constant measurements into global,
bridge, and local readout families.

## Abstract draft

The Hubble tension is usually framed as a disagreement between inferred values
of the same cosmological parameter. Here we test a different possibility: that
different observational channels read different source-role closures of the
same underlying H0 structure. We define a source-role conductance selector from
Fisher or covariance structure, separating local endpoint conductance from
global closure conductance. Applying this selector to time-delay lensing,
standard-ruler, CMB, distance-ladder, and standard-siren channels yields a
reproducible family split. DESI BAO, Planck PR3 CMB covariance, and
TDCOSMO+SLACS channels select the global/low branch. Pantheon+SH0ES and
TDCOSMO-only or TDCOSMO+IFU channels select the local/high branch. GW170817-like
standard sirens select an intermediate bridge branch. Static role ablations and
classification-threshold sweeps fail to reproduce the split. These results do
not replace a full joint posterior refit, but they show that H0 channels can
separate by source role before such a refit is performed.

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

In this reading, the same universe can be read through different closures. The
Hubble tension is therefore not only a "which number is correct?" problem. It
is a source-role problem: which part of the inference graph is allowed to close
the measurement?

## Core claim

H0 tension is not only a conflict between numerical estimates. It is also a
conflict between readout roles:

- global closure channels read a low H0 branch,
- local endpoint channels read a high H0 branch,
- mixed source channels read an intermediate bridge branch.

The important point is temporal order. The source role is assigned before the
final H0 comparison. This makes the result stronger than a post-hoc grouping of
high and low values.

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

The important methodological constraint is that \(L\) and \(G\) are assigned
from source provenance and likelihood structure before comparing the resulting
branch to an observed H0 value.

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

> Hubble-tension channels split into local/high and global/low branches before
> a joint H0 refit.

Caption: Figure 1 visualizes the endpoint split of the H0 source-role readout.
Each point is placed by its source-role selector rather than by a fitted
cosmological preference chosen after the fact. The local/high endpoint contains
distance-ladder and time-delay-only channels whose closure is anchored by local
endpoint information. The global/low endpoint contains BAO, CMB, and
population-closure channels whose closure is anchored by global standard-ruler
or hierarchy information. The figure is a branch-selection test: source roles
are assigned before H0 comparison, and the endpoint separation is then checked.

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
| DESI BAO | global | 0.000000 | global/low | 67.247245 | branch-only |
| Planck CMB | global | 0.000000 | global/low | 67.247245 | Planck covariance H0 attached |
| TDCOSMO+SLACS+IFU | global | 0.001855 | global/low | 67.257791 | chain H0 attached |
| TDCOSMO+SLACS | global | 0.003704 | global/low | 67.268309 | chain H0 attached |
| GW170817 bright siren | bridge | 0.500000 | bridge/intermediate | 70.151263 | scoped reference H0 attached |
| TDCOSMO-only | local | 0.830134 | local/high | 72.137101 | chain H0 attached |
| TDCOSMO+IFU | local | 0.852221 | local/high | 72.271948 | chain H0 attached |
| Pantheon+SH0ES | local | 1.000000 | local/high | 73.180689 | branch-only |

This table should not be described as a joint H0 fit. It is a branch-readout
table. The important result is the ordering and family separation of
\(q_F\), not a replacement posterior.

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

The split is not produced by assigning every channel to a fixed role:

- all-local maps fail on global channels,
- all-global maps fail on local channels,
- flipped maps fail on every channel.

Threshold sweeps from broad local/global cuts also preserve the endpoint split.
This means the result is not tuned to one arbitrary classifier threshold.

## Reviewer objections and safeguards

### Objection 1: the families were assigned after seeing H0

Safeguard: the role map is assigned from source provenance and likelihood
structure before H0 comparison. TDCOSMO roles are derived from likelihood
factors and public notebook sampler composition. BAO roles are derived from
standard-ruler distance-ratio labels. Planck CMB roles are derived from
acoustic-scale covariance structure. Pantheon+SH0ES roles are derived from
calibrator and Hubble-flow source labels. GW roles are derived from the split
between luminosity distance and host-redshift anchoring.

### Objection 2: local/global assignment is arbitrary

Safeguard: static and flipped role ablations are run. All-local maps fail on
global channels, all-global maps fail on local channels, and flipped maps fail
on every endpoint channel. The split therefore requires source-aware role
assignment.

### Objection 3: the classifier threshold was tuned

Safeguard: the endpoint result survives broad threshold sweeps. The local and
global endpoint channels are far from the bridge midpoint in selector space.

### Objection 4: this is not a full cosmological inference

Safeguard: the paper states this explicitly. The result is a source-role
readout law and a branch-selection test. It does not claim to replace full
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
4. The theory is currently a readout law, not a replacement for all standard
   cosmological inference machinery.

## What the paper gains

The main gain is not a single new H0 number. The gain is a structural
explanation of why H0 measurements cluster where they do:

- low branch measurements are global closures,
- high branch measurements are local endpoints,
- intermediate measurements are mixed source-role bridges.

This changes the question from "which H0 value is correct?" to "which source
role does a given observation use to read H0?"

## Next tests

1. Replace the GW bridge abstraction with event-level posterior samples.
2. Add TRGB/JAGB/CCHP distance-ladder variants as local or semi-local channels.
3. Add inverse-distance-ladder BAO+SN joint covariance as a stronger global
   channel.
4. Build a full joint posterior comparison only after the source-role map is
   fixed.

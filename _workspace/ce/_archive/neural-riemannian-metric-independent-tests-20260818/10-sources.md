# Independent-test source map

Status: COMPLETE  
Checked: 2026-08-18  
Scope: primary papers and official data/technology pages. These sources establish measurement capability or biological precedents; none alone validates the chain `Delta W^s -> Delta g -> future trajectory`.

## Claim tiers

- **Tier A capability**: the source can plausibly provide same-unit or same-synapse structure, intervention, and later trajectory, subject to preregistered linkage and quality checks.
- **Tier B capability**: it provides an important part of the chain, but lacks direct structure, intervention, or complete longitudinal linkage.
- **Tier C context**: it supports a mechanism or measurement component but cannot adjudicate the causal chain.

## Primary-source map

| Route | Source | What is measured | Strongest defensible use | Missing / kill condition |
|---|---|---|---|---|
| Same synapse, longitudinal learning | Hedrick et al., *Learning binds new inputs into functional synaptic clusters via spinogenesis*, 2022, [Nature Neuroscience DOI](https://doi.org/10.1038/s41593-022-01086-6) | Longitudinal two-photon spine imaging during motor learning plus correlated EM and presynaptic partners | **Tier B capability** for structure/function linkage; it lacks a same-synapse perturbation and a locked, independent future-trajectory endpoint under this contract | No direct test of a Riemannian metric; fail if future trials or same-synapse IDs cannot be locked before prediction |
| Same tissue structure/function | MICrONS Consortium, *Functional connectomics spanning multiple areas of mouse visual cortex*, 2025, [official portal](https://www.microns-explorer.org/) and [DOI](https://doi.org/10.1038/s41586-025-08790-w) | Co-registered calcium imaging and EM connectome; tens of thousands of functional units and synapses | **Tier B static structure-function and bridge-calibration capability, not $H_W$**; a frozen static $\Phi(W,c)$ can predict held-out responses | Endpoint structure with no longitudinal rewiring, intervention, or later trajectory; it cannot identify $\Delta W^s\to\Delta g$ or sleep change |
| Longitudinal same-cell state geometry | *Learning produces an orthogonalized state machine in the hippocampus*, 2025, [primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC11964937/) | Thousands of CA1 neurons tracked across learning, with behavior and state-manifold changes | **Tier B** test of Delta g -> future representation/behavior, using frozen chart and future-session prediction | Activity geometry is not structural W; kill if metric is fit from the same future activity used as outcome |
| Causal perturbational system identification | *Active learning of neural population dynamics using two-photon holographic optogenetics*, 2024, [primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC12466919/) | Cellular-resolution stimulation patterns with simultaneous population imaging; stimulation chosen for system identification | **Tier B capability for H_G** and for separating drift/noise from local response ellipsoids; it is not Tier A because it does not itself measure structural W or a same-synapse pre/post change | Does not measure structural W by itself; kill if stimulation is used only for decoder improvement and no held-out perturbation response is scored |
| Neural-system perturbation precedent | Marshel et al., *Cortical layer-specific critical dynamics underlying flexible sensorimotor behavior*, 2019, [Nature](https://doi.org/10.1038/s41586-019-1675-2) | Cell-type/layer-specific optogenetic perturbations with behavior and population activity | **Tier B** causal falsification of direct-dynamics alternatives; perturbation can be an instrument for local metric/drift separation | Perturbation does not identify synaptic W; a behavioral effect alone is not geometric evidence |
| Spontaneous/evoked consistency | Allen Brain Observatory Visual Coding, [official data portal](https://portal.brain-map.org/explore/circuits/visual-coding) | Standardized visual stimuli, spontaneous activity, cellular imaging and electrophysiology across sessions | **Tier B** test whether a metric estimated from spontaneous fluctuations predicts evoked transition covariance and future trials | No learning-induced structural change; spontaneous and evoked data must be split by session and not jointly fit |
| Transition / first-passage tests | Ezaki et al., *Brain states are associated with distinct dynamical landscapes*, 2023, [primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC10503743/) | Neural-state transitions and dynamical landscape/action analysis | **Tier B** test of committor, transition-path and first-passage predictions under an explicit SDE; compare geodesic distance against drift/noise-only models | Landscape/action is not a Riemannian metric unless a bridge law is declared; kill if `tau_B` is replaced by deterministic distance |
| Control-energy separation | Gu et al., *Controllability of structural brain networks*, 2015, [primary report](https://doi.org/10.1038/ncomms9414) | Structural network, linear controllability and control-energy calculations | **Tier C method precedent**; useful as a competing structural predictor and nuisance control | Control Gramian is not process-noise covariance or biological metric; no claim of `g=Q^{-1}` |
| Sleep/replay/renormalization | *Human REM sleep recalibrates neural activity in support of memory formation*, 2023, [primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC10456851/) | Rodent two-photon/electrophysiology plus human EEG; REM-linked recalibration of population dynamics | **Tier B** sleep pre/post test: preregister `Delta g_sleep` and predict next-day trajectories/behavior, with wake and replay controls | Calcium is not direct firing or synaptic weight; no evidence for Ricci flow or forced metric update |
| Sleep synaptic scale | *Evidence for sleep-dependent synaptic renormalization in mouse pups*, 2019, [primary report](https://doi.org/10.1093/sleep/zsz184) | Synaptic structural changes across wake/sleep | **Tier B capability** for structural pre/post sleep comparison | Developmental cohort and no trajectory bridge; cannot test the full chain alone |
| Sleep topology baseline | Gardner et al., *Toroidal topology of population activity in grid cells*, 2022, [Nature DOI](https://doi.org/10.1038/s41586-021-04268-7) | Grid-cell population topology across wake, REM, and slow-wave sleep | **Tier C topology-vs-metric control**; test whether topology persists while local distances or transition statistics change across state | Does not measure structural W or establish a Riemannian metric; topology persistence is not metric invariance |
| Physical-surface baseline | Pang et al., *Geometric constraints on human brain function*, 2023, [Nature DOI](https://doi.org/10.1038/s41586-023-06098-1) | Cortical-surface geometry and Laplace-Beltrami eigenmodes related to macroscale activity | **Tier C anatomical competing model**; include surface eigenmodes/geodesics as a baseline before attributing effects to learned `Delta g` | Macroscale cross-sectional association, not longitudinal synaptic change or causal trajectory mediation |
| Longitudinal neural/behavior control benchmark | LINK dataset, Chestek Lab, [official project page](https://chesteklab.github.io/LINK_dataset/) | 312 sessions over 303 days, chronic neural recordings and synchronized finger kinematics | **Tier B** held-out future-trajectory and invariance benchmark for fixed-identity effective dynamics | No direct W or synaptic intervention; chronic channel identity is not same-cell structural identity |
| Functional-connectivity longitudinal benchmark | Gillon et al., 2023, DANDI 000037, [official DANDI record](https://dandiarchive.org/dandiset/000037) | Longitudinal two-photon M1 activity, ROI tracking and behavior | **Tier B** test of representation and behavior transfer across learning days | Direct synapse and causal perturbation absent; eligibility requires manifest-level identity audit |
| Behavioral manifold intervention | Sadtler et al., *Neural constraints on learning*, 2014, [Nature DOI](https://doi.org/10.1038/nature13665) | BCI decoder perturbations placed within or outside a participant's existing neural-manifold geometry, with subsequent learning curves | **Tier C lower-cost behavioral route**; test whether a preregistered metric length of imposed decoder perturbations predicts learning rate and transfer | Constrains accessible population-manifold directions but has no structural `W`, no direct metric measurement, and no evidence for `W -> g` |

## Dataset and technology capability matrix

The complete machine-readable matrix is
[`artifacts/capability-matrix.csv`](artifacts/capability-matrix.csv). `Unknown`
means that the cited release or method does not establish the field; it is not
imputed from an adjacent experiment.

| ID | Scale | Unit identity | Structural signal | Activity signal | Perturbation | Longitudinal support | Maximum claim |
|---|---|---|---|---|---|---|---|
| Hedrick-2022 | dendritic branch / spines | same spines across learning | live spine turnover plus endpoint CLEM | spine iGluSnFR and behavior | task exposure; no randomized synapse target | yes, same spines | Tier B static/change structure-function component |
| MICrONS-2025 | multi-area mouse cortex | 2P-to-EM cell registration | endpoint EM synapses | repeated visual 2P calcium before fixation | visual stimuli only | functional scans, but no repeated structural $W^s$ | Tier B static bridge only; not $H_W$ |
| Wagenmaker-2024 | 500--700 cortical neurons | within-session targeted cells | none | 20 Hz 2P calcium responses | holographic cellular stimulation | trial blocks, not structural longitudinal | Tier B $H_G$ capability |
| Chen-2025 | up to 100 candidate presynaptic cells and one postsynaptic cell | optically targeted presynaptic cells | causal synaptic presence/strength proxy | whole-cell postsynaptic responses | sequential or compressed holographic stimulation | approximately five-minute maps; repeated-learning support unknown | Tier B local $H_W$ component if repeated |
| Allen-Visual-Coding | mouse visual cortex | ROI/session dependent | none | calcium and electrophysiology | sensory stimuli | repeated sessions; stable identity must be audited | Tier B spontaneous/evoked $H_G$ |
| Gardner-2022 | grid-cell modules | sorted units within recordings | none | spikes in wake, REM and SWS | environments; no circuit intervention | wake/sleep blocks | Tier C topology baseline |
| Pang-2023 | human cortical surface | mesh vertices / parcels | anatomical surface mesh, not synaptic $W^s$ | multiple macroscale activity maps | none | mainly cross-sectional | Tier C physical-fold baseline |
| LINK | chronic motor arrays | channels, not guaranteed cells | none | spikes and kinematics | behavioral task | 303 days | Tier B invariance/trajectory benchmark |
| DANDI-000037 | longitudinal mouse M1 | tracked 2P ROIs subject to audit | none | calcium and behavior | learning task | repeated days | Tier B $H_G$/transfer only |
| Sadtler-2014 | macaque M1 population | within-session recorded units | none | spikes and behavior | randomized BCI decoder mapping | within-session learning | Tier C behavioral accessibility |
| zebrafish all-optical platform | larval whole brain | registered cells/regions within experiment | atlas or endpoint structure only | volumetric calcium and behavior | targeted optogenetics | feasible repeated activity; repeated $W^s$ unknown | Tier B $H_G$ platform |
| C. elegans combined platform | selected synapses to whole nervous system | named neurons and fluorescent synapses | live synapse label or atlas connectome | calcium and behavior | single-synapse ablation or patterned optogenetics | feasible for selected synapses; no cited all-in-one dataset | proposed Tier A pilot, not existing evidence |

The technology rows combine only capabilities that must still be integrated in
one preregistered experiment. Relevant primary demonstrations include in-vivo
holographic synaptic mapping ([Chen et al., 2025](https://doi.org/10.1038/s41593-025-02024-y)),
freely moving zebrafish whole-brain imaging plus optogenetics
([primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC10776927/)), and
live single-synapse ablation/long-term imaging in *C. elegans*
([primary report](https://pmc.ncbi.nlm.nih.gov/articles/PMC2535809/)). None is an
already completed test of the full chain.

## Operational routes and independent falsifiers

1. **Same-unit longitudinal structural route (highest priority).** Measure pre/post synapse or spine identity, a frozen activity chart, and future trials. Fit `Phi` only on pre-intervention data. Test `H_W: Delta W -> Delta g`, then `H_C` on later trajectory distributions. Nulls: direct `v,Q`, firing-rate/gain, Euclidean covariance, shuffled structure, and parameter-matched metric. Kill the claim if Delta W adds no out-of-sample proper-score gain or if the sign reverses across animals.

2. **Perturbational response-ellipsoid route.** Apply randomized holographic stimulation to identified cells, estimate local response covariance and drift from held-out trials, and test whether the predeclared metric predicts response ellipsoid deformation. This separates a metric-like noise geometry from a decoder or retrospective covariance summary. Kill if the metric does not predict perturbational responses beyond unconstrained `v,Q`.

3. **Spontaneous-to-evoked route.** Estimate `g` from spontaneous blocks only, then predict evoked transition distributions in a different block/session. A same-block score is circular. Kill if the result disappears under stimulus, gain, or state-label permutation.

4. **First-passage route.** Define target basin `B`, estimate committor and `p(tau_B)` from held-out trajectories, and compare geodesic, Freidlin-Wentzell action, drift-barrier, and noise-only models. A distance decrease is not a time prediction without an explicit coupling law. Kill if geodesic gains vanish after controlling for drift, barrier, and diffusion.

5. **Sleep/replay route.** Acquire structure or synaptic proxy before learning, immediately after learning, after sleep, and after matched wake. Predict next-day trajectories and behavior. Replay/sham, sleep deprivation, and gain/noise controls are required. Treat Ricci-flow language as a model hypothesis, never as an interpretation of sleep data.

6. **Topology-versus-metric route.** Hold graph support/topology fixed while perturbing weights or effective gains, and separately create/delete a shortcut while preserving local covariance. Test whether future transitions respond to support changes, edge weights, or local metric deformation. A topology-only predictor winning would falsify the claim that metric deformation is the necessary explanatory object.

7. **Cross-scale triangulation.** Use MICrONS for structure-function mapping, longitudinal optical/neuropixels datasets for future trajectories, and perturbation datasets for causal response. These are complementary, not replicates. A claim is Tier A only when the same experimental unit supplies the linked measurements or the bridge is validated by an explicitly held-out transfer experiment.

## Measurement boundary

Connectomes, calcium movies, covariance matrices, decoder manifolds, and graph distances establish that a quantity can be computed. They do not establish that the brain uses that quantity as a Riemannian metric. The decisive evidence must be a predeclared `Phi`, independent future outcomes, explicit chart covariance, and superiority over direct-dynamics and gain/noise controls at the animal or subject level.

## Pre-existing folds and baseline geometry

The statement "the brain already has folds" must be split into three typed
objects. They are not interchangeable.

### A. Physical cortical folding and laminar geometry

The cortical sheet is an anatomical surface with sulci and gyri. Surface-mesh geodesic distance, mean curvature, cortical depth, layer identity, cell type, and axon path length are measurable covariates. Primary work has used surface geodesics and folding metrics to relate cortical geometry to connectivity and neural dynamics ([Goulas et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27807628/); [Human cortical dynamics reflect graded contributions of local geometry and network topography, 2025](https://doi.org/10.1038/s41467-025-67740-2)). Layer-specific circuit mapping shows that synaptic wiring depends on laminar position and cell subclass ([Harris et al., 2021, DOI 10.1126/science.abj5861](https://doi.org/10.1126/science.abj5861)); the BICCN cellular anatomy resource supplies registered cell types, cortical depth and projection wiring ([Tasic et al., 2021](https://doi.org/10.1038/s41586-021-03970-w)).

These data support a **physical baseline scaffold**, not yet a neural-state Riemannian metric. A valid test should include surface geodesic distance, cortical depth/layer, cell type, and estimated wiring length as nuisance predictors. Otherwise a purported `g` effect may merely restate anatomy.

### B. Pre-existing developmental/connectivity-derived state geometry

Define a baseline field `g0(z)` from a pre-learning period, fixed anatomical connectome, cell-type wiring, or a mechanistic model. It must be estimated before the intervention and then frozen. `g0` is distinct from physical surface geometry: it is a state-space object induced by a declared chart and map from anatomy/dynamics. The mouse cortical connectome demonstrates that baseline anatomical graphs have dense, weight-specific and area-specific organization ([Oh et al., 2014](https://doi.org/10.1038/nature13186); [Gamanut et al., 2018](https://doi.org/10.1016/j.neuron.2017.12.037)). MICrONS supplies a stronger same-tissue structure/function resource ([MICrONS Consortium, 2025](https://doi.org/10.1038/s41586-025-08790-w)). None of these papers establishes that their graph is the subject's neural-state metric.

Before intervention, freeze four separate baseline-producer IDs: anatomical
`P-h`, structural-connectivity `P-W`, pre-period dynamical `P-D`, and combined
`P-C`. Each producer gets its own complexity budget and must predict only
pre-intervention calibration outcomes during selection. Estimate relative
deformation against every retained producer; do not select a producer using the
post-intervention trajectory. A pseudo-intervention placed wholly inside the
pre-period must yield a placebo deformation compatible with zero, and every
producer must also face an equally flexible flat-pullback null.

The required decomposition is therefore

```text
physical anatomy / lamina / wiring -> g0(z)
learning, context, sleep, intervention -> Delta g(z,c)
later drift, noise, trajectory -> p(x_0:T, tau_B | g0 + Delta g, v, Q)
```

The first arrow is a model bridge that must be predeclared and compared with direct `v,Q` alternatives. The second is the biological claim under test. Within-subject differencing (`post - pre`) and baseline-period estimation are required; cross-sectional condition means are insufficient.

### C. Deformation versus coordinate artifact

Every nonconstant baseline field must be tested against a flat nonlinear-coordinate pullback null. A Euclidean metric written in nonlinear coordinates can have nonconstant components and nonzero Christoffel symbols while retaining zero Riemann curvature. Thus the experiment should compare:

1. surface/laminar baseline covariates only;
2. a frozen `g0` derived from pre-period anatomy or connectivity;
3. `g0 + Delta g` with the preregistered learning/context/sleep update;
4. a flat pullback metric generated by a smooth coordinate warp;
5. unconstrained drift/noise and gain-only controls.

The metric claim survives only if the deformation predicts independent future trajectories or first-passage distributions after these controls, and the result is stable under chart transformation, cell re-identification, and animal-level resampling.

### E17 boundary

E17 included none of the anatomical surface mesh, cortical folding, cortical depth, laminar identity, cell type, axon length, or direct structural `W^s` measurements. It also did not estimate a nonconstant pre-existing `g0(z)` and then measure a post-learning `Delta g`. Its sessionwise fitted activity dynamics and covariance candidates therefore cannot be interpreted as evidence that physical cortical folds or a developmental baseline field generated the candidate metric. At most, E17 is a retrospective feasibility result for activity-derived metric-like summaries.

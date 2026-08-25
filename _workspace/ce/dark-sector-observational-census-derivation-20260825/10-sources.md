# Primary-source observational census

Status: COMPLETE

Access date for all links: 2026-08-25.  Rows are kept as published likelihood
outputs; no central values from different likelihoods are combined into a new
tuple.  “Covariance” means that the collaboration supplies a joint chain,
likelihood, or covariance product suitable for reproducing the quoted joint
fit.  A derived density fraction is labelled derived even when it is printed
in a table.

## 1. Planck 2018 baseline CMB

Primary source: [Planck Collaboration VI, A&A 641 A6 (2020),
doi:10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910);
[official Planck parameter tables](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters).
The baseline below is TT,TE,EE+lowE+lensing, base-ΛCDM, 68% marginalized
intervals (Planck table convention):

|parameter|value|status/covariance|
|---|---:|---|
|ω_b=Ω_bh²|0.02237 ± 0.00015|direct posterior; joint chains/covariance available|
|ω_c=Ω_ch²|0.1200 ± 0.0012|direct posterior; joint chains/covariance available|
|H₀|67.36 ± 0.54 km s⁻¹ Mpc⁻¹|direct posterior; joint chains/covariance available|
|Ω_m|0.3153 ± 0.0073|derived from sampled densities and H₀; covariance available through chains|
|Ω_Λ (flat)|0.6847 ± 0.0073|derived as 1−Ω_m in flat model; not an independent sample|
|Ω_k|0 (fixed in baseline)|model assumption; non-flat runs are separate|
|σ₈|0.8111 ± 0.0060|derived posterior; covariance available through chains|
|S₈=σ₈(Ω_m/0.3)^0.5|≈0.832 ± 0.013|derived posterior; correlation with σ₈,Ω_m available through chains|

These are external CMB inputs, not CE predictions.  The Planck table and
likelihood products provide the parameter covariance; the one-dimensional
numbers above must not be treated as an independent diagonal tuple.

## 2. DESI DR2 BAO and supernova combinations

Primary source: [DESI Collaboration, “DESI DR2 Results II”,
arXiv:2503.14738](https://arxiv.org/abs/2503.14738); [official DR2 data and
likelihood/release page](https://data.desi.lbl.gov/doc/releases/dr2/).
The DR2 BAO sample contains >14 million galaxies/quasars plus Lyα BAO; exact
redshift-bin vectors and covariance are in the release products.  The paper
uses CamSpec CMB likelihood (with robustness checks using Plik and LiteBIRD–
HFI) and the three named uncalibrated SN samples.

Official DR2 baseline is Planck PR3 low-l TT/EE + Planck PR4 NPIPE CamSpec
high-l + Planck-PR4/ACT-DR6 lensing.  Plik, HiLLiPoP and LoLLiPoP are
robustness variants; the earlier LiteBIRD wording is superseded.  Additional
flat-ΛCDM primary table rows are DESI-only Ωm=0.2975±0.0086; DESI+BBN
Ωm=0.2977±0.0086, H0=68.51±0.58 km s⁻¹ Mpc⁻¹; and DESI+BBN+θ* Ωm=0.2967±0.0045,
H0=68.45±0.47 km s⁻¹ Mpc⁻¹.  These are released joint products, not locally
pinned covariance files.

|model and exact combination|published result|covariance/status|
|---|---|---|
|flat ΛCDM, DESI DR2 BAO alone|BAO–CMB parameter discrepancy 2.3σ in (Ω_m,H₀r_d); DESI-only Ω_m and H₀r_d are table/chain outputs|full DESI BAO covariance and chains available; the 2.3σ uses the published 2×2 posterior covariances|
|flat ΛCDM, DESI DR2 BAO+CMB|Ω_m=0.3027±0.0036; H₀=68.17±0.28 km s⁻¹ Mpc⁻¹; correlation r=−0.975|joint fit; covariance explicitly published (2×2 for this pair), chains available|
|flat ΛCDM, DESI DR2 BAO+BBN|H₀ tension with SH0ES =4.5σ; BBN fixes early-universe calibration, so this is not BAO-only|joint BAO covariance and BBN prior; full posterior product available|
|w₀w_aCDM, DESI DR2 BAO+CMB|w₀=−0.42±0.21; w_a=−1.75±0.58; preference 3.1σ|joint released chain/covariance; not locally pinned|
|w₀w_aCDM, DESI DR2 BAO+CMB+Pantheon+|w₀=−0.838±0.055; w_a=−0.62^{+0.22}_{−0.19}; preference 2.8σ|joint released chain/covariance; not locally pinned|
|w₀w_aCDM, DESI DR2 BAO+CMB+Union3|w₀=−0.667±0.088; w_a=−1.09^{+0.31}_{−0.27}; preference 3.8σ|joint released chain/covariance; not locally pinned|
|w₀w_aCDM, DESI DR2 BAO+CMB+DES-SN5YR (DESY5)|w₀=−0.752±0.057; w_a=−0.86^{+0.23}_{−0.20}; preference 4.2σ|joint released chain/covariance; not locally pinned|
|ΛCDM, DESI DR2 BAO+CMB|Σmν<0.064 eV (95%, physical positive-mass prior)|joint posterior; model/prior dependent, not a CE scalar-mass limit|
|w₀w_aCDM, DESI DR2 BAO+CMB|Σmν<0.16 eV (95%)|joint posterior; relaxed by DE freedom|

The DR2 paper explicitly reports no joint DESI+SN ΛCDM parameter tuple for
the three SN samples; only the published marginal Ω_m discrepancies are
quoted (Pantheon+ 1.7σ, Union3 2.1σ, DESY5 2.9σ).  This is a source-level
availability limitation, not permission to combine marginals.

### Current DR2 Lyα full-shape/AP update (2026)

The cutoff also includes the DESI collaboration's [DR2 Results IV,
arXiv:2607.27410](https://arxiv.org/abs/2607.27410) and its [official DR2
paper/data index](https://data.desi.lbl.gov/doc/papers/dr2/), released 2026-07-30.
This is a distinct primary likelihood from the 2025 DR2 galaxy/quasar BAO
paper and must not be folded into its parameter tuple:

|model/data|published result|covariance/status|
|---|---|---|
|Lyα auto + Lyα–quasar cross full-shape/AP, joint with Lyα BAO, z_eff=2.33|D_H/r_d=8.600±0.066 and D_M/r_d=39.32±0.33|joint AP+BAO likelihood/covariance in DESI release; direct distance-ratio posterior|
|flat ΛCDM, Lyα measurements+BBN|H₀=66.5±1.3 km s⁻¹ Mpc⁻¹|joint Lyα likelihood with BBN prior; external early-universe calibration|
|flat ΛCDM, Lyα AP|Ω_m=0.325±0.018; 1.4σ above DESI galaxy/quasar BAO|marginal posterior; full chains supplied, but not a synthetic DESI-DR2 BAO tuple|
|w₀w_aCDM, DESI DR2 BAO+Lyα full-shape+CMB|w₀=−0.54^{+0.19}_{−0.21}; w_a=−1.39^{+0.61}_{−0.50}; preference 2.7σ|joint released chain/covariance; not locally pinned|
|w₀w_aCDM, same plus DES-Dovekie SNe|w₀=−0.821±0.054; w_a=−0.65±0.20; preference 3.2σ|joint released chain/covariance; not locally pinned|

The Lyα AP result moves the DESI-versus-CMB discrepancy slightly from 2.4σ to
2.2σ and therefore weakens (rather than strengthens) the earlier DR2
galaxy/quasar-BAO dynamical-DE indication.  A companion [three-redshift Lyα
BAO primary analysis, arXiv:2607.19619](https://arxiv.org/abs/2607.19619)
reports D_V/r_d=30.26±0.39 to 32.22±0.47 and D_M/D_H=3.96±0.15 to
5.63^{+0.22}_{−0.24} over z_eff=2.13,2.40,2.81; it is a consistency/extension
product, not a replacement for the single-bin DR2 Results IV likelihood.

The earlier [DESI DR1 primary paper, arXiv:2404.03002](https://arxiv.org/abs/2404.03002)
is retained because it is present in the frozen CE corpus: DESI+CMB gives
Ω_m=0.307±0.005 and H₀=67.97±0.38; wCDM DESI-only gives
w=−0.99^{+0.15}_{−0.13}; w₀w_a preference is 2.6σ for DESI+CMB and
2.5σ/3.5σ/3.9σ after Pantheon+/Union3/DESY5.  These are DR1, not DR2, and
must not be substituted for the current DR2 rows.

## 3. Structure and local calibration comparators

The current primary weak-lensing/large-scale-structure row is DES Y6, not
Y3.  Primary source: [DES Y6 3x2pt, arXiv:2601.14559](https://arxiv.org/abs/2601.14559)
and [DES Y6 release page](https://www.darkenergysurvey.org/des-y6-cosmology-results-papers/).

|model/data|value|covariance/status|
|---|---:|---|
|Y6-only flat ΛCDM 3x2pt|S8=0.789±0.012; Ωm=0.333^{+0.023}_{−0.028} (68%)|direct posterior; DES release provides chains/likelihood, not locally pinned in this run|
|Y6-only flat wCDM 3x2pt|S8=0.782^{+0.021}_{−0.020}; Ωm=0.325^{+0.032}_{−0.035}; w=−1.12^{+0.26}_{−0.20} (68%)|direct joint posterior; released covariance/chains, not locally pinned|
|Y6 3x2pt+CMB+low-z (flat ΛCDM)|S8=0.806^{+0.006}_{−0.007}; Ωm=0.302±0.003; h=0.683^{+0.003}_{−0.002}|joint released posterior; external CMB/low-z inputs, not locally pinned|
|same joint combination, wCDM|w=−0.981^{+0.021}_{−0.022}|joint released posterior; no significant preference over ΛCDM|

|probe|model/data combination|value|covariance/status|
|---|---|---:|---|
|DES Collaboration, [DES Y3 cosmic shear + galaxy clustering + galaxy–galaxy lensing, arXiv:2105.13549](https://arxiv.org/abs/2105.13549)|flat ΛCDM, DES Y3 3×2pt|S₈=0.776^{+0.017}_{−0.017} (68%)|historical frozen-CE comparator only; joint DES likelihood/covariance exists but is not locally pinned|
|SH0ES, [Riess et al. 2024, arXiv:2404.08038](https://arxiv.org/abs/2404.08038)|SMC Cepheids plus four geometric anchors and HST Cepheid distance ladder|H₀=73.17±0.86 km s⁻¹ Mpc⁻¹ (68%)|internal ladder covariance available; external calibration comparator, not a CMB/BAO joint fit|

The DES Y3 row is a low-redshift structure probe and is not a measurement of
the CE residual scalar itself.  SH0ES is deliberately kept external because
the DESI+BBN tension statement assumes standard pre-recombination physics.

## 4. Scalar/fuzzy-DM mass and fraction limits

|primary result|model/data combination|published limit|covariance/status|
|---|---|---:|---|
|Iršič et al., [MNRAS 466 (2017), arXiv:1703.04683](https://arxiv.org/abs/1703.04683)|fuzzy/ultralight scalar all-DM transfer function; XQ-100+HIRES/MIKE Lyα flux power, hydrodynamical simulations|m_fdm>2.0×10⁻²¹ eV (2σ; conservative thermal-history case); 3.75×10⁻²¹ eV for smoother power-law thermal history|simulation nuisance covariance and likelihood are model-specific; not a universal scalar bound|
|Winch et al., [HST/JWST UV luminosity functions + Planck, arXiv:2404.11071](https://arxiv.org/abs/2404.11071)|ULA fraction f_a of DM, HST UVLF (24,000 sources, 4<z<10)+Planck CMB|all-DM excluded for m_a<10⁻²¹·⁶ eV; for 10⁻²⁶<m_a/eV<10⁻²³, f_a<0.22 (95%); stated bound applies −26<log₁₀(m_a/eV)<−23|joint HST/Planck likelihood with marginalized halo–UV nuisance model; covariance supplied by the analysis, but astrophysical model dependence is substantial|

These are primary scalar/fuzzy-DM constraints, not detections.  They constrain a
specified scalar transfer function or UV-halo mapping; they do not measure the
CE history-to-stress normalization M_* or the vacuum offset V_Λ.

## 5. CE-internal numerical claims and provenance (superseded legacy table)

The short table immediately below is retained only for historical traceability;
its 0.0486 and 0.315 labels are deprecated.  The normalized census in §6a is
the sole active CE-internal record and uses runtime Ω_b=0.0487 and normalized
forward-boundary Ω_m=0.31096890310968905.

The following values occur in the frozen CE cosmology corpus and are therefore
recorded separately from observations:

|claim/input in CE corpus|value|status|
|---|---:|---|
|CE ratio audit Ω_b|0.0486 (exact code default)|CE internal/adopted comparison input; not a no-input prediction|
|CE ratio audit Ω_c/Ω_DM|0.2623 (code/runtime tuple)|provenance-conflicted legacy value; quarantined by contract and not observational evidence|
|CE ratio audit Ω_Λ|0.6891 (code/runtime tuple)|same quarantined legacy tuple; not observational evidence|
|forward-model default Ω_m|0.315|external Planck-like baseline used by code|
|forward-model default σ₈|0.811|external baseline input|
|forward-model default H₀|code-specific; usually 67.4 km s⁻¹ Mpc⁻¹|external calibration input|
|flat closure|Ω_b+Ω_DM+Ω_DE=1|exact only after flatness and component completeness are adopted; derived, not a measured CE equality|
|historical CE ratio `R=α_s D_eff` and q_ext-derived abundances|no independently sourced observational value|bridge is NOT_AVAILABLE; probability values cannot be promoted to density fractions|

The CE-internal rows are audit inputs/claims and are never combined with the
Planck or DESI rows.  In particular, the rounded `(0.0487,0.2623,0.6891)`
runtime tuple is excluded as required by the contract.

## 6. Closure and limitations

## 6a. Expanded CE-internal inventory (not observations)

The following audit inputs are included so the census closes over every
numeric dark-sector claim in the frozen CE corpus.  They are code/document
values, not primary observational measurements:

|CE item|exact value or result|status/provenance|
|---|---:|---|
|alpha_s closure input|α_s=0.11789; PDG comparison 0.1180±0.0009|external coupling input; CE uses it before cosmology construction|
|fixed-point chain at α_s=0.11789|δ=0.1777584234; D_eff=3.1777584234; ε²=0.04864671964|derived by `cosmology_discrimination_gates.py`; ε² is the fixed-point output, not Ω_b observation|
|3-layer ratio|R=0.3806281404|CE candidate, using documented coupling-ratio sum 1.015; status OPEN against NLO|
|NLO ratio|R=0.3857942086|CE candidate `R=α_sD_eff+(α_sD_eff)^2/(4π)`; status OPEN against 3-layer|
|closed-form approximation|R=0.37789|explicitly approximate (0.72% below 3-layer), not accepted as exact 3-layer result|
|3-layer derived fractions|Ω_m=0.3109272131; Ω_Λ=0.6890727869|CE-derived within this candidate; not observationally calibrated and not the runtime tuple|
|ratio-audit runtime Ω_b|0.0487|correct runtime tuple value; CE internal/adopted comparison input, not a prediction|
|historical runtime dark tuple|0.0487, 0.2623, 0.6891|rounded/provenance-conflicted legacy tuple; quarantined and not evidence|
|q_ext / survival|q_ext and 1−q_ext are genealogical probabilities; finite positive-decay bootstrap is absorbing without drive|no independently sourced probability→energy normalization; survival-to-density bridge NOT_AVAILABLE|
|R=α_sD_eff abundance route|R candidates above|α_s and D_eff do not derive Ω_DM/Ω_DE without the missing history-to-stress scale and matching rule|
|LO/NLO/3-layer decision|LO/3-layer/NLO are alternative CE bookkeeping routes; no observational kill measurement frozen in corpus|model ambiguity remains OPEN; cannot combine route outputs|
|forward-model boundary|`(S_res,T_muν,initial data,other species) -> H(z),D(z),P(k,z) -> observables`|normalized boundary is an external-input forward model; not a CE-only prediction|
|DESI legacy 13-vector BAO gate|full-covariance χ²/ν/p output exists in the frozen CE forward harness|diagnostic reproduction only; exact covariance/data snapshot is not locally pinned in this source lane|
|BAO scale-fit diagnostic|one fitted H₀r_d scale factor, Δχ²/AIC/BIC and equivalent r_d/H₀ outputs|explicitly fitted-scale diagnostic, not prediction; local reproduction not claimed here|
|early-r_d result|early-universe sound-horizon adapter/BBN-r_d result is an external boundary input|not a CE derivation of r_d; likelihood/data version not locally pinned|
|modern-likelihood flags|all modern-likelihood provenance flags are `false` in the scorecard audit|flag result is a CE audit datum; it does not validate the physical bridge|
|scorecard status|cosmology rows distinguish external Planck/SN/BAO inputs, fitted scales, and rejected legacy tuples|scorecard is not an observational source and is not used to create a hybrid tuple|

### Canonical executed CE values

The active normalized record is: α=.11789, sin²θW=.23122206826075514,
δ=.17775842340997383, D=3.1777584234099736, q=.048646719644028225,
survival=.9513532803559718, Dq=.15458752312007412.  Alpha and sin²θW are
inputs/model choice; q is a conditional branching output.  The runtime tuple
(.0487,.2623,.6891) has raw sum 1.0001 and is UNRESOLVED/quarantined.

Model alternatives (scientific default=None) are LO
(.048646719644028225,.25927170943410105,.6920815709218708), 3-layer manuscript
(.048646719644028225,.2622797333,.6890735470), 3-layer approximation
(.048646719644028225,.26228049346744653,.6890727868885254), and NLO
(.048646719644028225,.26484927098139216,.6865040093745797).  They are mutually
exclusive bookkeeping routes, not a combined prediction.

The normalized historical forward boundary is raw Ωm=.311, ΩΛ=.6891, giving
Ωm=.31096890310968905 and ΩΛ=.6890310968903111.  Defaults are runtime
densities with external H0=67.4, rd=147.09 Mpc, σ8=.811, w0=−1, wa=0.
The full-covariance DESI 13-vector gives χ²=37.10026085715347 for 13 dof,
p=.000399573259824 (REJECT).  The fitted-scale diagnostic gives scale
.98647693346963, χ²=12.6083468622414 for 12 dof, p=.398138192515, equivalent
rd=149.106375435 Mpc and H0=68.3239493122; this is explicitly not a prediction.
The early-rd value 151.318753028 gives χ²=40.4682255438671,
p=.000116176098098 (REJECT).

The scorecard rows are ωb h²=.0221005274 versus Planck .02237±.00015
(−1.796σ), ωDM h²=.1191565948 versus .11933±.00091 (−.191σ), ΩΛ=.6891
versus .6847±.0073 (+.603σ), and versus DESI flat-derived .6973±.0036
(−2.28σ).  Its old w0=−.769 is +1.25σ versus current DESI+Pantheon+.
All are seen/non-holdout marginal comparisons without joint covariance; all
modern-likelihood flags are false.  The DESI vector provenance is CobayaSampler/
bao_data tag v2.6 commit b7b8a36e9bccb063081f811f323cada21ab5fbdd (2025-03-20),
mean SHA256 9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585,
covariance SHA256 252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509;
embedded floats match numerically but source bytes are not locally pinned.

The numerical CE forward boundary and DESI 13-vector/covariance artifacts are
linked by the frozen repository harness but are not locally pinned primary
likelihood files in this source lane.  Accordingly, this census records their
existence and status without claiming local covariance reproduction.

All five parameter families are populated: composition (Planck and DESI),
expansion/calibration (Planck, DESI, SH0ES), dark-energy dynamics (DESI DR2
with all three named SN combinations), structure (Planck and DES Y6, with Y3
retained only as historical comparator), and
scalar/fuzzy-DM mass/fraction (Lyα and HST/Planck).  All six probe families in
contract §4 are represented.  Official covariance/chain availability is
recorded row-by-row; where a primary paper reports only a significance or
marginal discrepancy, that limitation is retained.  No hybrid parameter tuple
has been constructed.

This census establishes empirical inputs and comparators only.  It does not
evaluate or prove the CE mathematical claim; that belongs to `11-math.md` and
the audit lane.

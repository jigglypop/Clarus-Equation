# SNKC quench background holdout data (v1)

Sealed-evaluation holdout inputs for preregistration manifest
`experiments/preregistration/cosmology_snkc_quench_bg_v1.json`
(ledger `SNKC-R2-THEATER-QUENCH-BG-PREREG-10`).

Access date: **2026-08-30** (Asia/Seoul). Files downloaded verbatim with
`curl -fsSL` from the pinned upstream commits below; no row was edited,
reordered, or reformatted. First contact of this repository with these
numerical values is the sealed evaluation of the manifest above.

## Sources

### 1. eBOSS DR16 / BOSS DR12 consensus BAO (primary holdout)

- Upstream: `https://github.com/CobayaSampler/bao_data`, commit `bb0c1c9`
  (official Cobaya packaging of the SDSS final consensus BAO likelihood files;
  Alam et al. 2021, Phys. Rev. D 103, 083533).
- Raw URL pattern:
  `https://raw.githubusercontent.com/CobayaSampler/bao_data/bb0c1c9/<file>`

| file | sha256 |
|---|---|
| `sdss_DR12_LRG_BAO_DMDH.dat` | `ccdbe5ad44016ea09e10e30f0178eb75b756417730b7b53c611ac896623c5f81` |
| `sdss_DR12_LRG_BAO_DMDH_covtot.txt` | `fd2a67856f0ffa7267cff5245b579dcdb4b5cd461849377a2f8e9582a7679544` |
| `sdss_DR16_LRG_BAO_DMDH.dat` | `b3317e7590799fad71a9a707023d0743c14d87399d6bb4129965d6a5732d91be` |
| `sdss_DR16_LRG_BAO_DMDH_covtot.txt` | `1a45e106f8e2bbf8742a6c3d4a9c11bdc288801fc6824e0db8cfbab4290f6160` |
| `sdss_DR16_QSO_BAO_DMDH.txt` | `9d3a43515d009d5c836728d4af1f1d02887fcdd874aba098c597f1f47693bbe6` |
| `sdss_DR16_QSO_BAO_DMDH_covtot.txt` | `c0d8bab47132045139c5bbd0ebfd8464434e1354371ceeeca70bb90ecbcee383` |
| `sdss_DR16_ELG_BAO_DVtable.txt` | `ebbd6b7a2946cf1903bac9e699702e6aa57a631799bb70421c8e7a55cb3d2c1f` |
| `sdss_DR16_LYAUTO_BAO_DMDHgrid.txt` | `40cee3a1c9dc58616ba7151ab9d020b0014238249409cd1ace71af14674e37e0` |
| `sdss_DR16_LYxQSO_BAO_DMDHgrid.txt` | `653e2cea43a742d12090e9b7eacaf74dc7af7d7f6153a1a4c696d6303a7fb952` |

Effective redshifts (fixed by the source publication, not fit here):
DR12 LRG bins z = 0.38, 0.51; DR16 LRG z = 0.698; DR16 QSO z = 1.48;
DR16 ELG z = 0.845 (DV/rd likelihood table); Lya auto and LyaxQSO z = 2.334
(DM/rd x DH/rd likelihood-ratio grids, 50x50).

Notes (declared before evaluation):

- **DR12 two-bin file (double-count avoidance).** The original BOSS DR12
  consensus had three bins (0.38, 0.51, 0.61); in the SDSS final consensus the
  z = 0.61 bin is merged into the eBOSS DR16 LRG measurement. The 2-bin
  `sdss_DR12_LRG_BAO_DMDH` file used here is the official double-count-free
  combination and is the one shipped with the DR16 likelihoods.
- **P1 — DR16 LRG covtot errors ~10% smaller than the individual paper.**
  `sqrt(diag)` of `sdss_DR16_LRG_BAO_DMDH_covtot.txt` gives (0.328, 0.533)
  versus (0.37, 0.56) quoted in the LRG paper alone. This is because the
  Cobaya file carries the SDSS *consensus combination* (Fourier + configuration
  space, systematics-included consensus of Alam et al. 2021), not the single
  analysis. Recorded as P1: resolved by adopting the official consensus values;
  no result-driven choice was made.

### 2. Cosmic chronometers, Moresco 15-point homogeneous subset (secondary holdout)

- Upstream: `https://gitlab.com/mmoresco/CCcovariance`, commit `88141333`
  (Moresco et al. 2020, arXiv:2003.07362; covariance recipe per the official
  `CCcovariance` notebooks).
- Raw URL pattern:
  `https://gitlab.com/mmoresco/CCcovariance/-/raw/88141333/data/<file>`

| file | sha256 |
|---|---|
| `HzTable_MM_BC03.dat` | `32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98` |
| `data_MM20.dat` | `577ac2f346e346fe7cf94daa7b7000c05d04ebc8a029cda31e0d8643b956a485` |

`HzTable_MM_BC03.dat`: 15 homogeneous H(z) points (z, Hz, errHz, stat and
metallicity contributions, reference). `data_MM20.dat`: percent systematic
contributions (IMF, stellar library, SPS model `mod`, odd-one-out `mod_ooo`)
tabulated in z, to be interpolated to the data redshifts and combined as
outer products `cov^c_ij = H_i f^c_i H_j f^c_j` with `f = percent/100`
on top of `diag(errHz^2)`, exactly as in the official notebook recipe.

## Protocol pointer

The evaluation protocol (Gaussian 8x8 primary block P, extended variant X
with ELG + Lya likelihood tables, CC GLS with full covariance, kill rule
delta chi2 > +9 triggered by either P or X) is implemented in
`examples/physics/kinetic_dark_sector_quench_holdout_eval.py`.
Provenance: the manifest of ledger `SNKC-R2-THEATER-QUENCH-BG-PREREG-10`
(model, holdout identity, nuisances, +9 threshold) was frozen before
holdout access; the P/X decomposition and the either-P-or-X strengthening
were clarified AFTER manifest freeze and holdout acquisition, BEFORE any
model-vs-data computation — a documented, conservative, verdict-invariant
deviation recorded in the run audit.  Data-file sigma and structure were
seen at acquisition time; the sealed evaluation is the first
model-vs-data scoring of these numbers in this repository.

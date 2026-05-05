# Mouse IBL/OpenAlyx larger registered panel audit

## equation reset

The current mouse equation is not a region-only or unit-only claim.
The supported term is the mixed current-window readout:

$$
y_t^{(s)}
\sim
X_t^{(s,\mathrm{task})}
+R_t^{(s,\mathrm{hybrid})}
+U_t^{(s)}\mid X_t^{(s,\mathrm{task})},R_t^{(s,\mathrm{hybrid})}.
$$

The strict temporal extension remains a counterexample, not a promoted term:

$$
U_{t-\ell}^{(s)}\mid
X_t^{(s,\mathrm{task})},R_t^{(s,\mathrm{hybrid})},U_t^{(s)}
\quad\mathrm{not\ yet\ supported}.
$$

So the next step is a larger registered panel that can repeat the mixed-readout tests before fitting heavier directed/state-space coupling models.

## metadata search

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- project: `ibl_neuropixel_brainwide_01`
- scanned sessions: 5
- mixed-readout ready sessions: 5
- temporal-GLM metadata ready sessions: 5

Required metadata/files:

| block | requirement |
|---|---|
| task/history \(X_t\) | `_ibl_trials.table.pqt` |
| movement targets | `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy` |
| current unit \(U_t\) | `spikes.times.npy`, `spikes.clusters.npy` |
| hybrid region \(R_t\) | `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`; strict acronyms are kept when present |

## top registered candidates

| rank | eid | session | probes | strict probes | channel probes | metrics probes | score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | 2 | 2 | 2 | 2 | 308 |
| 2 | `4720c98a-a305-4fba-affb-bbfa00a724a4` | `danlab/Subjects/DY_014/2020-07-14/001` | 2 | 1 | 2 | 2 | 268 |
| 3 | `d2832a38-27f6-452d-91d6-af72d794136c` | `wittenlab/Subjects/ibl_witten_29/2021-06-08/001` | 1 | 1 | 1 | 1 | 153 |
| 4 | `cf43dbb1-6992-40ec-a5f9-e8e838d0f643` | `danlab/Subjects/DY_011/2020-02-08/001` | 1 | 1 | 1 | 1 | 153 |
| 5 | `ee13c19e-2790-4418-97ca-48f02e8013bb` | `danlab/Subjects/DY_008/2020-03-04/001` | 1 | 1 | 1 | 1 | 153 |

## lab and subject spread

| item | count |
|---|---:|
| lab `danlab` | 3 |
| lab `angelakilab` | 1 |
| lab `wittenlab` | 1 |
| subject `NYU-30` | 1 |
| subject `DY_014` | 1 |
| subject `ibl_witten_29` | 1 |
| subject `DY_011` | 1 |
| subject `DY_008` | 1 |

## next gate

- panel audit passed: `False`
- recommended candidate count: 5
- next executable gate: repeat block-regularized mixed readout on this larger panel, then rerun strict temporal GLM only if the mixed term remains stable.

# Mouse IBL/OpenAlyx Neuropixels audit

목표는 mouse Neuropixels/IBL 단계에서 region-level decision/action gate를 열 수 있는 최소 공개 데이터 bridge가 있는지 확인하는 것이다.
이 audit는 spike array를 내려받지 않고 OpenAlyx metadata만 확인한다.

## source

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- project: `ibl_neuropixel_brainwide_01`
- ONE dataset type names are used for search; file names may include `_ibl_` prefixes and revisions.

## bridge requirements

| tier | required dataset types | meaning |
|---|---|---|
| core | `trials.table, wheel.timestamps, spikes.times, spikes.clusters` | trial, wheel, spike time/cluster bridge |
| strict region | `clusters.brainLocationAcronyms_ccf_2017` | cluster-level CCF acronym labels |
| channel fallback | `clusters.channels, channels.brainLocationIds_ccf_2017` | channel CCF ids plus cluster-channel map |

## search results

| query | sessions found |
|---|---:|
| core + strict cluster region | 29 |
| core + channel-region fallback | 483 |

## first strict candidates

| eid | session | probes | strict ready | channel fallback ready |
|---|---|---|---|---|
| `d2832a38-27f6-452d-91d6-af72d794136c` | `wittenlab/Subjects/ibl_witten_29/2021-06-08/001` | `alf/probe00/pykilosort` | True | True |
| `dc21e80d-97d7-44ca-a729-a8e3f9b14305` | `wittenlab/Subjects/ibl_witten_26/2021-01-31/001` | `alf/probe00/pykilosort` | True | True |
| `8c2f7f4d-7346-42a4-a715-4d37a5208535` | `wittenlab/Subjects/ibl_witten_26/2021-01-29/001` | `alf/probe01/pykilosort` | True | True |
| `952870e5-f2a7-4518-9e6d-71585460f6fe` | `wittenlab/Subjects/ibl_witten_27/2021-01-19/001` | `alf/probe00/pykilosort`, `alf/probe01/pykilosort` | True | True |
| `c728f6fd-58e2-448d-aefb-a72c637b604c` | `wittenlab/Subjects/ibl_witten_27/2021-01-16/003` | `alf/probe00/pykilosort`, `alf/probe01/pykilosort` | True | True |

## first candidate dataset coverage

| item | files |
|---|---|
| trial table | `alf/#2025-03-03#/_ibl_trials.table.pqt`<br>`alf/_ibl_trials.table.pqt` |
| wheel timestamps | `alf/_ibl_wheel.timestamps.npy` |
| spikes times | `alf/probe00/pykilosort/#2024-03-22#/spikes.times.npy`<br>`alf/probe00/pykilosort/#2024-05-06#/spikes.times.npy`<br>`alf/probe00/pykilosort/spikes.times.npy` |
| spikes clusters | `alf/probe00/pykilosort/#2024-03-22#/spikes.clusters.npy`<br>`alf/probe00/pykilosort/#2024-05-06#/spikes.clusters.npy`<br>`alf/probe00/pykilosort/spikes.clusters.npy` |
| cluster region acronyms | `alf/probe00/pykilosort/clusters.brainLocationAcronyms_ccf_2017.npy` |
| cluster channels | `alf/probe00/pykilosort/#2024-03-22#/clusters.channels.npy`<br>`alf/probe00/pykilosort/#2024-05-06#/clusters.channels.npy`<br>`alf/probe00/pykilosort/clusters.channels.npy` |
| channel region ids | `alf/probe00/pykilosort/#2024-05-06#/channels.brainLocationIds_ccf_2017.npy`<br>`alf/probe00/pykilosort/channels.brainLocationIds_ccf_2017.npy` |

## verdict

- metadata bridge ready: `True`
- next gate: `region_binned_decision_action_decoding`

이 결과는 mouse 단계의 방정식 항을 닫은 것이 아니라, 다음 region-binned decision/action decoding gate를 열 수 있음을 뜻한다.

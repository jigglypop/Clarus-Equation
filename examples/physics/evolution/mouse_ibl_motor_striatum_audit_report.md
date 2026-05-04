# Mouse IBL/OpenAlyx motor-striatum audit

첫 mouse strict-session gate가 thalamus/visual/hippocampus 중심이었으므로, 이번 audit는 strict session 안에서 motor cortex와 striatal-complex coverage를 찾는다.
Spike arrays는 내려받지 않고 cluster acronym과 cluster metrics만 사용한다.

## source

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- project: `ibl_neuropixel_brainwide_01`
- strict sessions scanned: 29
- target families: `basal_ganglia_output, motor_cortex, septal_subpallium, striatal_complex`

## candidate probes

| eid | session | collection | target clusters | target spikes from metrics | target family spike counts |
|---|---|---|---:|---:|---|
| `4720c98a-a305-4fba-affb-bbfa00a724a4` | `danlab/Subjects/DY_014/2020-07-14/001` | `alf/probe01/pykilosort` | 449 | 16530280 | `{"septal_subpallium": 14090847, "striatal_complex": 2439433}` |
| `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `alf/probe01/pykilosort` | 209 | 8137407 | `{"septal_subpallium": 3638363, "striatal_complex": 4499044}` |
| `cf43dbb1-6992-40ec-a5f9-e8e838d0f643` | `danlab/Subjects/DY_011/2020-02-08/001` | `alf/probe00/pykilosort` | 98 | 4211707 | `{"motor_cortex": 4211707}` |
| `ee13c19e-2790-4418-97ca-48f02e8013bb` | `danlab/Subjects/DY_008/2020-03-04/001` | `alf/probe00/pykilosort` | 74 | 3713674 | `{"striatal_complex": 3713674}` |
| `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `alf/probe00/pykilosort` | 78 | 2649647 | `{"motor_cortex": 2649647}` |
| `158d5d35-a2ab-4a76-87b0-51048c5d5283` | `danlab/Subjects/DY_009/2020-03-03/001` | `alf/probe00/pykilosort` | 5 | 119196 | `{"basal_ganglia_output": 119196}` |
| `f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1` | `wittenlab/Subjects/ibl_witten_19/2020-07-22/001` | `alf/probe01/pykilosort` | 1 | 838 | `{"motor_cortex": 838}` |

## multi-probe candidates

| eid | session | collections | target families | target clusters | target spikes from metrics |
|---|---|---|---|---:|---:|
| `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `alf/probe01/pykilosort`, `alf/probe00/pykilosort` | `motor_cortex, septal_subpallium, striatal_complex` | 287 | 10787054 |

## verdict

- selected next gate eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- collections: `alf/probe01/pykilosort`, `alf/probe00/pykilosort`
- families: `motor_cortex, septal_subpallium, striatal_complex`

이 세션은 같은 behavioral trial table에서 motor cortex probe와 striatal-complex probe를 동시에 걸 수 있으므로, 다음 gate는 multi-probe region/action decoder다.

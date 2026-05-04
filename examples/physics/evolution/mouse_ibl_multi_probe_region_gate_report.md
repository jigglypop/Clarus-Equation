# Mouse IBL/OpenAlyx multi-probe motor-striatal region gate

Motor-striatum audit에서 선택한 NYU-30 session의 두 probe를 같은 trial table 위에 결합했다.
각 probe에서 region family와 acronym firing-rate feature를 만들고, probe별 feature를 이어 붙여 global-rate baseline과 비교한다.

## source

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- collections: `alf/probe00/pykilosort`, `alf/probe01/pykilosort`

## loaded arrays

- trials: 933
- wheel samples: 1309392

| probe | spikes | cluster slots | acronym rows | unknown fraction | target-family spikes |
|---|---:|---:|---:|---:|---:|
| `alf/probe00/pykilosort` | 24114519 | 908 | 643 | 0.191488 | 2649647 |
| `alf/probe01/pykilosort` | 31802063 | 561 | 209 | 0.744123 | 8137407 |

## feature construction

For probe \(p\), trial \(i\), and region group \(g\):

$$
x_{ipg}=\frac{1}{b_i-a_i}\sum_k \mathbf 1[t_{pk}\in[a_i,b_i]]\mathbf 1[G_p(c_{pk})=g].
$$

The multi-probe region vector is

$$
R_i^{\mathrm{multi}}=[x_{i,\mathrm{probe00},:},x_{i,\mathrm{probe01},:}].
$$

The compared models are `family_collapsed`, `family_by_probe`, `motor_striatal_family_by_probe`, `acronym_by_probe`, and `global_rate`.

## probe region summaries

### alf/probe00/pykilosort

| family | spikes |
|---|---:|
| `prefrontal_cortex` | 9911494 |
| `other` | 6935739 |
| `unknown` | 4617639 |
| `motor_cortex` | 2649647 |
| `somatosensory_cortex` | 0 |
| `visual_cortex` | 0 |
| `visual_thalamus` | 0 |
| `somatosensory_thalamus` | 0 |
| `striatal_complex` | 0 |
| `septal_subpallium` | 0 |

| acronym group | spikes |
|---|---:|
| `probe00:unknown` | 4617639 |
| `probe00:PL6a` | 3749742 |
| `probe00:ILA6a` | 3302122 |
| `probe00:DP` | 3039615 |
| `probe00:ACAd6a` | 2859630 |
| `probe00:AON` | 2004862 |
| `probe00:TTd` | 1891262 |
| `probe00:MOs6a` | 1758004 |
| `probe00:MOs5` | 891643 |

### alf/probe01/pykilosort

| family | spikes |
|---|---:|
| `unknown` | 23664656 |
| `striatal_complex` | 4499044 |
| `septal_subpallium` | 3638363 |
| `motor_cortex` | 0 |
| `somatosensory_cortex` | 0 |
| `visual_cortex` | 0 |
| `visual_thalamus` | 0 |
| `somatosensory_thalamus` | 0 |
| `basal_ganglia_output` | 0 |
| `prefrontal_cortex` | 0 |

| acronym group | spikes |
|---|---:|
| `probe01:unknown` | 23664656 |
| `probe01:ACB` | 2803057 |
| `probe01:SI` | 2702837 |
| `probe01:OT` | 1280274 |
| `probe01:LSv` | 755388 |
| `probe01:STR` | 415713 |
| `probe01:LSr` | 180138 |

## decoder results

### choice_sign

- window: `stimulus_20_320ms`
- definition: choice > 0

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `family_collapsed` | 930 | `{"0": 417, "1": 513}` | 0.712854 | 0.791844 | 0.497251 | 0.004975 | 0.213534 | True |
| `family_by_probe` | 930 | `{"0": 417, "1": 513}` | 0.775737 | 0.851160 | 0.496264 | 0.004975 | 0.276417 | True |
| `motor_striatal_family_by_probe` | 930 | `{"0": 417, "1": 513}` | 0.624427 | 0.650034 | 0.497894 | 0.004975 | 0.125107 | True |
| `acronym_by_probe` | 930 | `{"0": 417, "1": 513}` | 0.836748 | 0.910556 | 0.495400 | 0.004975 | 0.337428 | True |
| `global_rate` | 930 | `{"0": 417, "1": 513}` | 0.499320 | 0.518565 | 0.500412 | 0.691542 | 0.000000 | False |

### first_movement_speed

- window: `stimulus_20_320ms`
- definition: firstMovement_times - stimOn_times <= median
- median seconds: 0.151490

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `family_collapsed` | 833 | `{"0": 416, "1": 417}` | 0.637411 | 0.691760 | 0.499162 | 0.004975 | 0.031189 | True |
| `family_by_probe` | 833 | `{"0": 416, "1": 417}` | 0.661424 | 0.711285 | 0.499547 | 0.004975 | 0.055202 | True |
| `motor_striatal_family_by_probe` | 833 | `{"0": 416, "1": 417}` | 0.570213 | 0.615759 | 0.494481 | 0.004975 | -0.036009 | False |
| `acronym_by_probe` | 833 | `{"0": 416, "1": 417}` | 0.673460 | 0.713608 | 0.499607 | 0.004975 | 0.067239 | True |
| `global_rate` | 833 | `{"0": 416, "1": 417}` | 0.606222 | 0.625300 | 0.501346 | 0.004975 | 0.000000 | False |

### wheel_action_direction

- window: `first_movement_-100_200ms`
- definition: wheel position displacement from first movement to min(response, first movement + 250 ms)
- min absolute wheel displacement: 0.001000

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `family_collapsed` | 930 | `{"0": 522, "1": 408}` | 0.708404 | 0.789737 | 0.501398 | 0.004975 | 0.195642 | True |
| `family_by_probe` | 930 | `{"0": 522, "1": 408}` | 0.760339 | 0.850429 | 0.502493 | 0.004975 | 0.247577 | True |
| `motor_striatal_family_by_probe` | 930 | `{"0": 522, "1": 408}` | 0.640382 | 0.693815 | 0.500447 | 0.004975 | 0.127620 | True |
| `acronym_by_probe` | 930 | `{"0": 522, "1": 408}` | 0.851462 | 0.920076 | 0.504272 | 0.004975 | 0.338700 | True |
| `global_rate` | 930 | `{"0": 522, "1": 408}` | 0.512762 | 0.578558 | 0.499815 | 0.009950 | 0.000000 | False |

## verdict

- gate passed: `True`
- passed non-global rows: ['choice_sign:family_collapsed', 'choice_sign:family_by_probe', 'choice_sign:motor_striatal_family_by_probe', 'choice_sign:acronym_by_probe', 'first_movement_speed:family_collapsed', 'first_movement_speed:family_by_probe', 'first_movement_speed:acronym_by_probe', 'wheel_action_direction:family_collapsed', 'wheel_action_direction:family_by_probe', 'wheel_action_direction:motor_striatal_family_by_probe', 'wheel_action_direction:acronym_by_probe']

이 gate는 첫 thalamic/visual probe 결과를 motor/striatal multi-probe session으로 일반화하는 중간 단계다. 단, 아직 단일 session이므로 최종 mouse 항은 여러 session 반복으로 닫아야 한다.

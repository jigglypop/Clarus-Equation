# Mouse IBL/OpenAlyx region-binned decision/action gate

첫 strict IBL/OpenAlyx Neuropixels 세션을 실제로 내려받아 trial table, wheel, spike time, spike cluster, cluster-region acronym을 연결했다.
목표는 영역별 발화율이 choice/action 변수를 단순 majority와 global firing-rate baseline보다 잘 예측하는지 확인하는 것이다.

## source

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- probe collection: `alf/probe00/pykilosort`

## loaded arrays

| item | value |
|---|---:|
| trials | 663 |
| wheel samples | 1126509 |
| spikes | 34822347 |
| max observed cluster id + 1 | 700 |
| strict acronym rows | 648 |
| spikes with strict acronym row | 33789015 |
| spikes assigned unknown | 1033332 |
| unknown spike fraction | 0.029674 |

## feature construction

For trial \(i\), region group \(g\), and window \([a_i,b_i]\), the feature is a duration-normalized spike count:

$$
x_{ig}=\frac{1}{b_i-a_i}\sum_k \mathbf 1[t_k\in[a_i,b_i]]\mathbf 1[G(c_k)=g].
$$

The compared models are:

| model | feature set |
|---|---|
| `region_family` | anatomical family bins: visual cortex, visual thalamus, somatosensory thalamus, hippocampus, cingulate, other, unknown |
| `acronym_region` | high-spike CCF acronyms plus unknown |
| `global_rate` | one scalar total firing rate, used as flat baseline |

The decoder is a z-scored ridge linear classifier evaluated with deterministic stratified cross-validation. The null model shuffles labels and repeats the same CV routine.

$$
\hat w=\arg\min_w\|y-Xw\|_2^2+\lambda\|w\|_2^2,\qquad
\mathrm{BA}=\frac12\left(\frac{TP}{P}+\frac{TN}{N}\right).
$$

## region groups

| family | mean stimulus-window rate |
|---|---:|
| `motor_cortex` | 0.000000 |
| `somatosensory_cortex` | 0.000000 |
| `visual_cortex` | 391.573655 |
| `visual_thalamus` | 1866.445450 |
| `somatosensory_thalamus` | 3891.709402 |
| `striatal_complex` | 0.000000 |
| `septal_subpallium` | 0.000000 |
| `basal_ganglia_output` | 0.000000 |
| `prefrontal_cortex` | 0.000000 |
| `hippocampus` | 988.471594 |
| `cingulate` | 36.616390 |
| `other` | 0.000000 |
| `unknown` | 211.618904 |

Top acronym groups by spike count:

| acronym group | spikes |
|---|---:|
| `PO` | 15814791 |
| `LP` | 8712114 |
| `CA1` | 2031843 |
| `VISa5` | 1724705 |
| `VPM` | 1697249 |
| `unknown` | 1033332 |
| `VPLpc` | 824549 |
| `DG-po` | 796917 |
| `DG-sg` | 785649 |
| `CA3` | 532485 |
| `DG-mo` | 371329 |
| `VISa6a` | 309037 |
| `cing` | 188347 |

## decoder results

### choice_sign

- window: `stimulus_20_320ms`
- definition: choice > 0

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `region_family` | 663 | `{"0": 300, "1": 363}` | 0.582259 | 0.611570 | 0.500529 | 0.004975 | 0.078072 | True |
| `acronym_region` | 663 | `{"0": 300, "1": 363}` | 0.632548 | 0.687778 | 0.501580 | 0.004975 | 0.128361 | True |
| `global_rate` | 663 | `{"0": 300, "1": 363}` | 0.504187 | 0.531745 | 0.500748 | 0.149254 | 0.000000 | False |

### first_movement_speed

- window: `stimulus_20_320ms`
- definition: firstMovement_times - stimOn_times <= median
- median seconds: 0.200485

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `region_family` | 625 | `{"0": 312, "1": 313}` | 0.748756 | 0.830804 | 0.500294 | 0.004975 | 0.067195 | True |
| `acronym_region` | 625 | `{"0": 312, "1": 313}` | 0.847971 | 0.916452 | 0.502408 | 0.004975 | 0.166411 | True |
| `global_rate` | 625 | `{"0": 312, "1": 313}` | 0.681561 | 0.754680 | 0.500018 | 0.004975 | 0.000000 | False |

### wheel_action_direction

- window: `first_movement_-100_200ms`
- definition: wheel position displacement from first movement to min(response, first movement + 250 ms)
- min absolute wheel displacement: 0.001000

| model | n | class counts | BA | AUC | perm BA mean | p(BA>=obs) | delta global | pass |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `region_family` | 661 | `{"0": 342, "1": 319}` | 0.672194 | 0.733726 | 0.500501 | 0.004975 | 0.078659 | True |
| `acronym_region` | 661 | `{"0": 342, "1": 319}` | 0.709179 | 0.760170 | 0.500026 | 0.004975 | 0.115644 | True |
| `global_rate` | 661 | `{"0": 342, "1": 319}` | 0.593535 | 0.614640 | 0.500477 | 0.004975 | 0.000000 | False |

## verdict

- gate passed: `True`
- passed non-global rows: ['choice_sign:region_family', 'choice_sign:acronym_region', 'first_movement_speed:region_family', 'first_movement_speed:acronym_region', 'wheel_action_direction:region_family', 'wheel_action_direction:acronym_region']

해석:

- 첫 strict session에서 region/acronym activity는 choice, wheel action direction, movement speed를 global-rate baseline보다 잘 예측한다.
- 특히 acronym-level region bin은 세 target 모두에서 permutation gate를 통과한다.
- 단, 이 probe의 강한 영역은 thalamus, visual cortex, hippocampus 쪽이다. motor/striatal loop를 닫은 것이 아니므로 mouse 단계 전체를 닫으려면 다음에는 다중 probe 또는 motor/striatum 포함 세션을 추가해야 한다.

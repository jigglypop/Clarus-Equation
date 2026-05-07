# Clarus cell glia support operator gate

- passed: `True`
- claim level: `empirical_glia_support_context_branch`
- operators supported: `S,D,Q,U,R`
- microglia source: [Drager et al. 2022](https://www.nature.com/articles/s41593-022-01131-4)
- astrocyte source: [Leng et al. 2022](https://www.nature.com/articles/s41593-022-01180-9)

## decision

- `microglia_primary_ok`: `True`
- `microglia_state_shift_ok`: `True`
- `astrocyte_screen_ok`: `True`
- `astrocyte_cropseq_ok`: `True`
- `microglia_branch_ok`: `True`
- `astrocyte_branch_ok`: `True`

## summary

| branch | metric | value |
|---|---|---:|
| microglia primary screens | support genes with hits | 18 |
| microglia primary screens | phenotype classes hit | 4 |
| microglia CROP-seq | support genes shifting states | 7 |
| astrocyte screens | regulators with hits | 19 |
| astrocyte screens | phenotype classes hit | 3 |
| astrocyte CROP-seq | regulators changing support outputs | 15 |
| astrocyte CROP-seq | covered support output genes | 18 |

## microglia phenotype hits

- `AXL`: survival:-0.471, phagocytosis_crispri:-0.304
- `C1QA`: phagocytosis_crispri:-0.116, phagocytosis_crispra:0.058
- `C1QB`: phagocytosis_crispra:0.075
- `C1QC`: survival:0.045
- `CD33`: phagocytosis_crispri:-0.122
- `CDK12`: activation:1.070, phagocytosis_crispri:-0.977, phagocytosis_crispra:0.568
- `CDK8`: survival:1.907, activation:-2.268, phagocytosis_crispra:-0.066
- `CSF1R`: survival:-1.963, activation:0.454, phagocytosis_crispri:0.732, phagocytosis_crispra:-1.125
- `CSF2RA`: survival:-1.365, activation:0.407, phagocytosis_crispri:0.542
- `CSF2RB`: survival:-1.503, activation:0.511, phagocytosis_crispri:0.719, phagocytosis_crispra:-1.785

## microglia state-shift examples

- `CDK12`: max abs shift `2.892`, total abs shift `6.962`
- `MAPK14`: max abs shift `2.446`, total abs shift `4.933`
- `MED1`: max abs shift `2.069`, total abs shift `6.081`
- `CSF2RB`: max abs shift `0.986`, total abs shift `2.935`
- `CSF1R`: max abs shift `0.685`, total abs shift `3.995`
- `CSF2RA`: max abs shift `0.638`, total abs shift `2.703`
- `TGFBR2`: max abs shift `0.604`, total abs shift `2.915`

## astrocyte screen hits

- `RIPK1`: phagocytosis_inflammatory:0.754, vcam1_inflammatory:0.246, phagocytosis_inflammatory:-0.245, vcam1_inflammatory:-1.641
- `CEBPB`: phagocytosis_vehicle:-0.419, phagocytosis_inflammatory:-0.334, vcam1_inflammatory:1.971
- `MAP3K7`: phagocytosis_vehicle:0.455, phagocytosis_inflammatory:0.480, vcam1_inflammatory:-0.342
- `STAT3`: phagocytosis_vehicle:-0.322, phagocytosis_inflammatory:0.602, vcam1_inflammatory:1.910
- `CHUK`: phagocytosis_inflammatory:0.465, vcam1_inflammatory:-1.298
- `ETS1`: phagocytosis_inflammatory:0.635, vcam1_inflammatory:-0.471
- `FOXC2`: phagocytosis_inflammatory:-0.480, vcam1_inflammatory:-0.846
- `IKBKG`: phagocytosis_inflammatory:0.276, vcam1_inflammatory:-0.773
- `IRAK4`: phagocytosis_inflammatory:0.172, vcam1_inflammatory:1.045
- `IRF1`: phagocytosis_inflammatory:0.512, vcam1_inflammatory:-0.932

## astrocyte CROP-seq support-output hits

- `RELA`: `15` outputs; CCL2:-14.538, SOD2:-6.071, IL32:-6.589, CXCL8:-12.395, ICAM1:-2.546
- `RIPK1`: `14` outputs; CCL2:-10.279, SOD2:-4.461, IL32:-4.566, ICAM1:-1.889, CXCL8:-8.558
- `IKBKG`: `13` outputs; CCL2:-11.387, SOD2:-3.879, CXCL8:-9.384, ICAM1:-1.858, CCL20:-8.334
- `CEBPB`: `12` outputs; VCAM1:3.155, CCL2:6.663, ICAM1:1.416, IL32:2.742, SAA1:-3.547
- `MAP3K7`: `11` outputs; SOD2:-4.441, CCL2:-9.521, CXCL8:-9.502, CCL20:-8.586, IL32:-3.361
- `STAT3`: `11` outputs; CXCL10:20.006, IL32:8.999, IFIT3:2.203, STAT1:-1.778, IFIT1:1.333
- `CEBPD`: `9` outputs; IL32:2.471, VCAM1:1.348, SAA1:-3.714, C3:-0.806, CCL2:3.791
- `NFKB1`: `8` outputs; C3:1.087, CSF1:-1.034, VCAM1:-1.174, SOD2:-1.567, ICAM1:-0.882
- `NFKB2`: `8` outputs; VCAM1:3.835, IL32:4.898, CXCL10:4.195, CCL2:4.999, ICAM1:1.021
- `CHUK`: `7` outputs; CCL2:-4.220, IL32:-2.020, ICAM1:-0.975, SOD2:-1.572, CSF1:-0.819

## claim boundary

This supports the neural Clarus-cell S operator as glia-mediated support context. It does not prove in-vivo whole-brain closure or complete neuron-glia circuit dynamics.

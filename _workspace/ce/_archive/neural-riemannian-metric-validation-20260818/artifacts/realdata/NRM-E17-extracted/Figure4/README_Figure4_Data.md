# DataSummary_CaImagingDendrites.mat - Data Structure

## Overview

Longitudinal calcium imaging data from Layer 5b pyramidal neuron dendrites tracked across 5 consecutive sessions. Used to analyze representational stability and drift during rule learning.

**Recording:** Dendritic calcium imaging, 60 frames per trial  
**Figure:** Figure 4 - Longitudinal tracking during rule learning  
**Animals:** Same dendrites tracked across multiple days

## Experimental Conditions

### TEST Condition (Relearning Paradigm)
5 sessions tracking rule transitions:
- **Session 1:** Rule A 
- **Session 2:** Rule A 
- **Session 3:** Rule A 
- **Session 4:** Rule B 
- **Session 5:** Rule A 

### CONTROL Condition (Stability Baseline)
5 consecutive sessions of Rule A only (no rule changes)
- Purpose: Distinguish learning-related changes from natural representational drift

## Variable Naming Convention

Variables follow pattern: `[type]_[category]_[condition]`

**Condition suffix:**
- `_test` = Relearning paradigm (A→A→A→B→A)
- `_ctr` = Control (A→A→A→A→A)

## Data Structures

### 1. Coding Direction Variables (`cd_choice_*`, `cd_stim_*`)

**Fields:** `Lproj`, `Rproj`  
**Dimensions:** d × 60 (per session, 5 sessions total)

- **d:** Number of dendrites tracked that session
- **60:** Frames per trial
- **Lproj:** Projection onto coding direction for left instructed trials
- **Rproj:** Projection onto coding direction for right instructed trials

**Types:**
- `cd_choice_*`: Coding direction during choice/report period
- `cd_stim_*`: Coding direction during instruction/sample period

**Example structure:**
```matlab
cd_choice_test(1).Lproj  % Day 1 left projections (d×60)
cd_choice_test(1).Rproj  % Day 1 right projections (d×60)
% ... up to cd_choice_test(5) for Day 5
```

### 2. Significance Variables (`sig_*`)

**Fields:** `Choice`, `Stimuli`, `Delay`  
**Dimensions:** 5 × d (binary matrix)

- **Rows (5):** Session/day number
- **Columns (d):** Dendrite number
- **Values:** 1 = significantly coding, 0 = not significant

**Meaning:**
- `sig_test.Choice(3, 15) = 1` means dendrite 15 was significantly choice-selective on Day 3
- `sig_ctr.Delay(2, 8) = 0` means dendrite 8 was not delay-coding on Day 2

### 3. Selectivity Variables (`type_sum_*`)

**Structure:** Array of 5 elements (one per day), each with fields:

| Field | Dimensions | Description |
|-------|------------|-------------|
| `nu_type1` | 60 × d | Average activity for right instructed trials |
| `nu_type2` | 60 × d | Average activity for left instructed trials |
| `sel_types` | 60 × d | Selectivity = \|type1 - type2\| |

**Same as Figure 3 data, but with 5 days:**
- Row dimension (60): Time points across trial
- Column dimension (d): Individual dendrites
- Values: Calcium event rates or selectivity magnitude

**Example:**
```matlab
type_sum_test(1).nu_type1  % Day 1 right trials (60×d)
type_sum_test(1).nu_type2  % Day 1 left trials (60×d)
type_sum_test(1).sel_types % Day 1 selectivity (60×d)
```


### Additional Required Variable
```matlab
x_df               % Time vector for trial (length 60)
```


## Related Scripts

- `Figure4_Code.m` - Main analysis script
- `DeleteFullZero.m` - Remove inactive dendrites
- `SortROIsByPeak.m` - Sort and visualize by selectivity timing
- `coding_arranger.m` - Extract epoch-specific activity
- `rep_drift_selec.m` - Quantify representational drift
- `Plot_Selec_CD_Avg.m` - Visualize across all days



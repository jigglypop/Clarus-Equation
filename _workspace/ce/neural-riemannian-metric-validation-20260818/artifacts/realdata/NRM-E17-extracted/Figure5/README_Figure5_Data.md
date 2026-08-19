# Figure 5 Data - NDNF Interneuron Coding

## Overview

GCaMP calcium imaging from NDNF interneuron somas across rule learning sessions. Two datasets analyze (1) selectivity and representational drift, and (2) transition error coding.

**Recording:** NDNF interneuron soma calcium imaging  
**Figure:** Figure 5 - NDNF interneurons during rule learning

## Dataset 1: Selectivity and Representational Drift

**File:** `Figure5Data\_Selectivity\_RepDrift.mat`

### Structure

Identical to Figure 4 data structure but recorded from NDNF interneurons instead of L5b dendrites.

**Variables:**

```matlab
type\_sum\_all      % Selectivity across 5 sessions
sig\_all           % Significance flags (Choice, Stimuli, Delay)
cd\_choice\_all     % Choice period coding direction (5 sessions)
cd\_stim\_all       % Instruction period coding direction (5 sessions)
x\_df              % Time vector for trial
```

### Experimental Paradigm

5 sessions of rule learning:

* **Day 1:** Rule A
* **Day 2:** Rule A
* **Day 3:** Rule A
* **Day 4:** Rule B
* **Day 5:** Rule A

### Variable Details

Same as Figure 4 (see README\_Figure4\_Data.md for complete description):

* **type\_sum\_all:** 5-element array, each with `nu\_type1`, `nu\_type2`, `sel\_types`
* **sig\_all:** Fields `Choice`, `Stimuli`, `Delay` (5 × n binary matrices)
* **cd\_choice\_all / cd\_stim\_all:** 5-element array, each with `Lproj`, `Rproj`



## Dataset 2: Transition Error Coding

**File:** `Figure5Data\_TransitionError.mat`

### Purpose

Analyzes NDNF interneuron activity during rule transition sessions, specifically examining how these cells encode unexpected errors when animals apply the wrong rule.

### Variables

#### A→B Transition (Rule A to Rule B)

```matlab
ConAB\_Summ        % Trial-averaged activity during A→B errors
prob\_startAB      % Significance probabilities for error coding
selec\_startAB     % Selectivity during errors
```

#### B→A Transition (Rule B to Rule A)

```matlab
ConBA\_Summ        % Trial-averaged activity during B→A errors
prob\_startBA      % Significance probabilities for error coding
selec\_startBA     % Selectivity during errors
```

### Structure Details

#### ConAB\_Summ / ConBA\_Summ

Trial-averaged NDNF activity during transition error trials.

**Fields:**

* `selec`: Selectivity time course (180 × n)
* `selecabs`: Absolute selectivity (180 × n)
* Other activity measures

**Dimensions:** 180 × n

* **180:** Frames per trial (6 seconds at 30 Hz)
* **n:** Number of NDNF interneurons

#### prob\_startAB / prob\_startBA

Statistical significance for different coding types.

**Fields:**

* `resp`: P-value for response/choice period coding
* `stim`: P-value for stimulus/instruction period coding
* `err`: P-value for error-specific coding

**Dimensions:** n × 1 (one value per interneuron)

#### selec\_startAB / selec\_startBA

Direction and magnitude of error selectivity.

**Fields:**

* `err`: Error selectivity value for each interneuron

**Values:**

* Positive: Interneuron increases activity on errors
* Negative: Interneuron decreases activity on errors
* Magnitude: Strength of error selectivity



## Related Helper Functions

Same functions as Figure 4:

* `DeleteFullZero.m` - Remove inactive cells
* `SortROIsByPeak.m` - Sort and visualize
* `coding\_arranger.m` - Extract epochs
* `rep\_drift\_selec.m` - Quantify drift
* `Plot\_Selec\_CD\_Avg.m` - Visualize all sessions
* `trans\_gcamp\_general.m` - Analyze transition errors

## 


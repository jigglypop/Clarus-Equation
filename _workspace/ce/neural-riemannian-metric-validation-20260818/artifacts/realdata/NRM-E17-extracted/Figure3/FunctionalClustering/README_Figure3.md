# Functional Clustering Analysis - Figure 3



### Two Types of Correlations

#### 1. Noise Correlation (Coactivity)
- **Definition:** Pearson correlation of raw calcium activity between two synapses
- **Reflects:** Shared input, global processes, or common modulatory signals
- **Interpretation:** 
  - High coactivity = synapses receive similar input patterns
  - Independent of task structure

#### 2. Signal Correlation (Coding Correlation)
- **Definition:** Correlation of task-related activity patterns between synapses
- **Reflects:** Functional similarity in information encoding
- **Interpretation:**
  - High signal correlation = synapses encode similar task features
  - Depends on task structure and synapse selectivity

## Experimental Design

### Comparisons

#### Comparison 1: Rule A vs Rule B
- **Purpose:** Test if functional clustering depends on task complexity
- **Rule A:** Bidirectional task (complex) - requires left/right discrimination
- **Rule B:** Unidirectional task (simple) - always lick left
- **Hypothesis:** Clustering may be stronger during complex rule requiring integration

#### Comparison 2: Instruction-Selective vs Choice-Selective Synapses
- **Purpose:** Test if different synapse types show different clustering properties
- **Instruction-selective:** Synapses encoding sensory instruction (air puff side)
- **Choice-selective:** Synapses encoding motor choice (lick direction)
- **Hypothesis:** Choice-selective synapses may show stronger clustering

### Distance Threshold
- **Maximum distance analyzed:** 20 μm
- **Rationale:** Functional clustering is most relevant for nearby synapses

## Input Data Structures

### Primary Data Files

| File | Purpose | Contains |
|------|---------|----------|
| `SpatialCorr_RuleARuleB.mat` | Rule comparison | Correlation vs distance for both rules |
| `SpatialCorrInstructionVSChoiceStats.mat` | Synapse type comparison | Correlation vs distance + nearest neighbor data |

### Data Matrix Structure

All correlation matrices follow the same format:

#### Matrix Dimensions: n×2

| Column | Content | Units | Description |
|--------|---------|-------|-------------|
| Column 1 | Distance | μm (pre-scaling) | Spatial separation between synapse pairs |
| Column 2 | Correlation | -1 to 1 | Correlation coefficient (coactivity or signal) |



## Function Usage

### Basic Syntax

```matlab
summary = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
                                 branch_signal, branch_signal_RuleA)
```

### Full Syntax with Nearest Neighbor

```matlab
summary = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
                                 branch_signal, branch_signal_RuleA, ...
                                 inst_noise_nearneigh, resp_noise_nearneigh)
```

### Example Usage

```matlab
%% Analysis 1: Compare Rule A vs Rule B
load('SpatialCorr_RuleARuleB.mat')

% Run analysis
stats_rules = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
                                     branch_signal, branch_signal_RuleA);


%% Analysis 2: Compare Instruction vs Choice Selective Synapses
load('SpatialCorrInstructionVSChoiceStats.mat')

% Run analysis with nearest neighbor comparison
stats_types = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
                                     branch_signal, branch_signal_RuleA, ...
                                     inst_noise_nearneigh, resp_noise_nearneigh);


## Output Structure

### Summary Statistics

The function returns a structure with the following fields:

| Field | Description | Test Type |
|-------|-------------|-----------|
| `RHO_signal` | Spearman ρ for signal correlation vs distance (Rule B) | Correlation |
| `PVAL_signal` | P-value for signal correlation (Rule B) | Correlation |
| `RHO_noise` | Spearman ρ for coactivity vs distance (Rule B) | Correlation |
| `PVAL_noise` | P-value for coactivity (Rule B) | Correlation |
| `RHO_signal_RuleA` | Spearman ρ for signal correlation vs distance (Rule A) | Correlation |
| `PVAL_signal_RuleA` | P-value for signal correlation (Rule A) | Correlation |
| `RHO_noise_RuleA` | Spearman ρ for coactivity vs distance (Rule A) | Correlation |
| `PVAL_noise_RuleA` | P-value for coactivity (Rule A) | Correlation |
| `l_noise` | Test statistic for Rule A vs B coactivity | Mann-Whitney U |
| `p_noise` | P-value for Rule A vs B comparison | Mann-Whitney U |
| `pnearest` | P-value for nearest neighbor comparison | Mann-Whitney U |




## Dependencies

### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox (for `corr`, `fitlm`, `ranksum`)

### Required Helper Functions

| Function | Purpose |
|----------|---------|
| `plot_bar_errors()` | Create bar plots with error bars and statistical comparison |
| `violinplot()` | Generate violin plots showing full distributions |

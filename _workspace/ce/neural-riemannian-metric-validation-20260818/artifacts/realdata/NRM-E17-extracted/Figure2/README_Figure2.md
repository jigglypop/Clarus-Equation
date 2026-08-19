# Calcium Imaging Analysis - Figure 2

## Overview

This repository contains code for analyzing dendritic calcium activity in Layer 5b pyramidal neurons during chemogenetic manipulation of NDNF (Neuron-Derived Neurotrophic Factor) interneurons. The analysis examines how NDNF activation affects calcium signaling in dendritic spines and shafts during an instructed delayed response task.


## Experimental Design

### Conditions
- **Saline (Control):** Baseline calcium activity
- **DCZ (NDNF Activation):** Chemogenetic activation of NDNF interneurons using Deschloroclozapine

### ROI Types
- **Dendritic Spines:** Small protrusions on dendrites receiving synaptic input
- **Dendritic Shafts/Branches:** Main dendritic trunk showing global calcium activity

### Trial Types
- **Left Instructed Trials:** Air puff stimulus delivered to left whisker pad
- **Right Instructed Trials:** Air puff stimulus delivered to right whisker pad

## Input Data Structures

### 1. Transient Properties Data

Loaded via `DataSummarizer('matrix', variable_name, 'vert')`:

| Variable | Description | Structure |
|----------|-------------|-----------|
| `branch_amp` | Amplitude of calcium transients in dendritic branches | Structure with `.Sal` and `.DCZ` fields |
| `spine_amp` | Amplitude of calcium transients in dendritic spines | Structure with `.Sal` and `.DCZ` fields |
| `branch_freq` | Frequency of calcium transients in branches | Structure with `.Sal` and `.DCZ` fields |
| `spine_freq` | Frequency of calcium transients in spines | Structure with `.Sal` and `.DCZ` fields |

Each field (`.Sal`, `.DCZ`) contains a cell array where each cell corresponds to one imaging session/animal.

### 2. Behavioral Data Structure: `cont_data`

Loaded via `DataSummarizer('matrix', 'cont_data', 'horz')`:

Structure array with multiple sessions, each containing:

#### Main Fields

| Field Name | Structure | Description |
|------------|-----------|-------------|
| `Sal` | Substructure | Behavioral data during saline (control) condition |
| `DCZ` | Substructure | Behavioral data during DCZ (NDNF activation) condition |

#### Subfields (within `.Sal` and `.DCZ`)

| Field | Type | Column 1 | Column 2 | Description |
|-------|------|----------|----------|-------------|
| `TrialTypes` | t×1 | Trial instruction | N/A | 1=Left instructed, 0=Right instructed |
| `Choice` | t×2 | Animal's choice | Timestamp | 1=Licked left, 0=Licked right |
| `Outcomes` | t×2 | Trial outcome | Timestamp | -1=Impulsive, 0=Error, 1=Correct, 3=Omission |
| `DirOut` | t×1 | Performance | N/A | 1=Correct, 0=Incorrect |
| `spine_local` | t×r | Isolated spine calcium activity | N/A | Activity of r spines across t timepoints |
| `spine_all` | t×r | All spine events | N/A | Includes both local and global events |
| `branch` | t×r | Branch calcium activity | N/A | Activity of r branches across t timepoints |

**Note:** 
- `t` = number of timepoints in trial (e.g., imaging frames)
- `r` = number of ROIs (regions of interest)

### 3. Selectivity Data Structure: `dff_types`

Loaded via `DataSummarizer('matrix', 'dff_types', 'horz')`:

Structure organized by experimental condition and ROI type:

```
dff_types(session).Condition.ROI_type.DataType
```

#### Organization

| Level 1 | Level 2 | Level 3 | Description |
|---------|---------|---------|-------------|
| `Sal` or `DCZ` | `spine` or `branch` | `type1` | Mean activity for right instructed trials (t×r matrix) |
| | | `type2` | Mean activity for left instructed trials (t×r matrix) |
| | | `selec` | Trial-type selectivity across time (t×r matrix) |

#### Matrix Dimensions

- **t:** Number of timepoints across trial duration
- **r:** Number of ROIs (spines or branches)

#### Field Definitions

| Field | Formula | Description |
|-------|---------|-------------|
| `type1` | Mean(Right trials) | Average calcium activity during right instructed trials |
| `type2` | Mean(Left trials) | Average calcium activity during left instructed trials |
| `selec` | \|type1 - type2\| | Absolute difference between right and left trial activity |

**Selectivity** quantifies how well an ROI discriminates between trial types. High selectivity indicates the ROI responds differently to left vs. right instructions.


## Key Metrics

### Effect Size

**Formula:**
```
Effect Size = (Activity_Saline - Activity_DCZ) / (Activity_Saline + Activity_DCZ)
```

**Interpretation:**
- **1:** Complete suppression by DCZ
- **0:** No effect
- **-1:** Complete enhancement by DCZ

### Geometric Mean Activity

**Formula:**
```
Geometric Mean = √(Amplitude × Frequency)
```

**Purpose:** Combines two aspects of calcium signaling into a single metric

### Selectivity

**Formula:**
```
Selectivity(t) = |Activity_Right(t) - Activity_Left(t)|
```

**Interpretation:**
- High values indicate strong trial-type discrimination
- Low values indicate similar responses to both trial types

## Dependencies

### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox

### Custom Functions (must be in MATLAB path)
- `DataSummarizer()` - Loads and organizes data from multiple files
- `iam_lazy()` - Extracts specific calcium activity fields from `cont_data`
- `iam_lazyV2()` - Extracts specific fields from `dff_types`
- `plot_bar_errors()` - Creates bar plots with error bars and statistics
- `stdshade()` - Plots mean with shaded standard error
- `plot_epochs()` - Marks trial epoch boundaries on plots

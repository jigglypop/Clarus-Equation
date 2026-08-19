# Optogenetic Behavioral Analysis - Data Structure Documentation

## Overview

This repository contains code for analyzing behavioral performance during an instructed delayed response task with optogenetic manipulations. The analysis examines the effects of optogenetic stimulation at different task epochs on behavioral metrics including performance accuracy, reaction times, and licking behavior.

## Input Data Structure: `cont_data`

`cont_data` is a structure array with 40 entries, where each entry corresponds to behavioral data from one animal during one experimental session.

### Data Dimensions

- **Number of entries**: 40 (animal × session combinations)
- **Number of fields per entry**: 22
- **Trial structure**: Most fields are t×2 matrices, where:
  - `t` = number of trials in the session
  - Column 1 (t,1) = categorical values indicating event types
  - Column 2 (t,2) = timestamps when events occurred/began (in seconds)

### Key Fields

| Field Name    | Structure | Column 1 (t,1) Description | Column 2 (t,2) Description |
| ------------- | --------- | -------------------------- | -------------------------- |
| `Opto`        | t×1       | Optogenetic condition:<br>1 = No stimulation (control)<br>2 = Stimulation during sample/instruction period<br>3 = Stimulation during delay period<br>4 = Stimulation during report period | N/A |
| `TrialTypes`  | t×1       | Trial instruction:<br>1 = Left instructed trial<br>0 = Right instructed trial | N/A |
| `Choice`      | t×2       | Animal's choice:<br>1 = Licked left<br>0 = Licked right | Timestamp of choice initiation |
| `Outcomes`    | t×2       | Trial outcome:<br>-1 = Impulsive lick (premature response)<br>0 = Incorrect choice<br>1 = Correct choice<br>3 = Omission (no response) | Timestamp of outcome determination |
| `DirOut`      | t×1       | Binary performance:<br>1 = Correct trial<br>0 = Incorrect trial | N/A |
| `Confidence`  | t×1 or t×2 | Measure of choice confidence | Timestamp (if applicable) |
| `licks_trial` | t×1 or t×2 | Number of licks per trial | Timestamp information (if applicable) |

### Additional Fields

The structure contains additional fields capturing other behavioral and task-related variables. The exact structure may vary, but fields generally follow the t×2 format described above, with categorical values in the first column and timestamps in the second column when applicable.

## Task Structure

The behavioral task consists of several epochs:

1. **Sample/Instruction Period**: Animals receive a sensory cue indicating the correct response direction
2. **Delay Period**: A waiting period between instruction and response opportunity
3. **Report Period**: Animals report their choice by licking left or right

Optogenetic stimulation can occur during any of these three epochs (or not at all in control trials).

## Data Preprocessing

The analysis script performs the following preprocessing steps:

1. **Trial Selection**: 
   - Excludes initial baseline trials (before first Opto ≠ 1)
   - Excludes omitted trials (Outcome = 3)
   - Excludes impulsive trials (Outcome = -1)

2. **Session Quality Control**:
   - Requires minimum 7 trials per condition (left and right) per opto condition
   - Sessions not meeting this criterion are flagged and excluded from analysis

3. **Trial Categorization**:
   - Separates left-instructed and right-instructed trials
   - Groups trials by optogenetic condition

## Output Metrics

The analysis calculates the following behavioral metrics:

| Metric     | Description | Units |
| ---------- | ----------- | ----- |
| `DirOut`   | Proportion of correct choices | 0-1 (fraction) |
| `RT`       | Reaction time (choice latency) | seconds |
| `Licks`    | Average number of licks per trial | count |
| `Confi`    | Average confidence measure | variable |

Each metric is calculated for:
- **All trials combined** (Allstats)
- **Left-instructed trials only** (Lstats)
- **Right-instructed trials only** (Rstats)

## Statistical Analysis

The script performs one-way ANOVA to test for effects of optogenetic condition on each behavioral metric, separately for:
- Combined trials
- Left-instructed trials
- Right-instructed trials

## Visualization

The analysis generates a 3×3 grid of violin plots showing:
- **Top row**: Combined trials (all conditions)
- **Middle row**: Right-instructed trials (red axes)
- **Bottom row**: Left-instructed trials (blue axes)
- **Columns**: Different behavioral metrics (Performance, RT, Licks)

## Usage

```matlab
% Load your data
load('cont_data_example.mat')

% Run analysis
analyze_opto_behavior  % or run the script sections individually
```

## Dependencies

- MATLAB R2016b or later
- Statistics and Machine Learning Toolbox
- `violinplot` function (available from MATLAB File Exchange)

## Notes

- Impulsive and omitted trials are quantified separately but excluded from main behavioral analyses
- Sessions with fewer than 7 valid trials per condition per side are excluded from statistical analysis
- NaN values in reaction time and other continuous measures are handled using `nanmean`


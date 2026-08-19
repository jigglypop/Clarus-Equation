# SelectivityStatsGluSNFR.mat - Data Structure

## Overview

**Recording:** 100 Hz, 6-second trials = 600 frames per trial  
**Figure:** Figure 3 - Selectivity Analysis

## Data Structure

Each rule (RuleA, RuleB, RuleA_prime) contains three matrices:

### Matrix Format: s × 600
- **s** = number of synapses
- **600** = time points (6 seconds × 100 Hz)

### Fields

| Field | Description |
|-------|-------------|
| `dff_type1` | Average ΔF/F for **right instructed trials** |
| `type2` | Average ΔF/F for **left instructed trials** |
| `selec` | Selectivity = \|type1 - type2\| |

## Epoch Frame Numbers

| Epoch | Time (s) | Frames |
|-------|----------|--------|
| Initial Delay | 0.0-1.2 | 1-120 |
| Instruction | 1.2-1.9 | 121-190 |
| Delay | 1.9-2.9 | 191-290 |
| Go Cue | 2.9-3.0 | 291-300 |
| Report | 3.0-6.0 | 301-600 |



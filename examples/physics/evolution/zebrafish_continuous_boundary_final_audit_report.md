# Zebrafish continuous boundary final audit

This consolidates the existing continuous-alignment probes into one final boundary verdict.

## verdict

- timestamp-certified continuous ready: `False`
- verdict: `data_boundary`
- final continuous gate pass: `False`

## supported discrete bridges

| bridge | possible |
|---|---|
| `activity_to_behavior_frame_possible` | `True` |
| `activity_to_direction_possible` | `True` |

## alignment evidence

| item | value |
|---|---:|
| e2 neural matrix | True |
| behavior bout frame label | True |
| stage tracking txt | True |
| neural mat has coordinates | False |
| neural mat has e2 timestamp | False |
| laser-schedule matches | 1/10 |
| timestamp-certified alignments | 0/10 |
| candidate inferred alignments | 1/10 |
| supplementary has e2 timestamp variable | False |
| supplementary has e2-resampled behavior | False |

## candidate inferred decoding

| target | best lag e2 frames | R2 | mse/base | shift p | candidate |
|---|---:|---:|---:|---:|---|
| speed | 10 | 0.123460 | 0.876540 | 0.066667 | `True` |
| turn | 150 | 0.010998 | 0.989002 | 0.066667 | `False` |

## interpretation

- Activity-to-bout-frame and activity-to-direction gates remain supported.
- The inferred alignment gives a weak speed candidate, but it is not timestamp-certified.
- Turn is not supported even under the inferred alignment.
- Continuous movement decoding therefore remains blocked by missing e2 timestamp or e2-resampled speed/turn/heading trace.

# BA-TR22 validation

Focused test:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_three_event_relevance_no_go.py -q -p no:cacheprovider
1 passed in 3.79s
```

Fresh calibration seed `109001` produced the preregistered no-go witness.
Fresh development seeds `109101..109116` produced:

- witness rows: 16/16;
- pair-only composition: 64/64;
- exact three-route/three-target identity: 64/64;
- desired pair alone in the presence of the matched distractor: 0/64;
- first-arrival H positive count: exactly three;
- delivered packet receipt: `[0,0,0,3,0,0,0]`;
- written packet receipt: `[0,3,0,0,0,0,0]`;
- zero-store gate: 16/16.

Confirmation stays sealed because this is a constructive no-go, not a
candidate efficacy result.


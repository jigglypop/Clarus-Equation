# BA-TR23 validation

Focused test:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_context_packet_relevance_gate.py -q -p no:cacheprovider
1 passed in 3.99s
```

Fresh calibration seed `110001` passed. Fresh development seeds
`110101..110116` produced:

- row gate: 16/16;
- learned context gate: 64/64;
- oracle pair gate: 64/64 and bit-exact with learned;
- cyclic context shuffle: 0/64;
- fixed context-0 gate: 16/64, exactly one of four contexts per seed;
- no-context all-input rule: 0/64;
- exact three-event one-shot receipt, gate immutability, and zero stores:
  16/16.

Confirmation remains sealed.


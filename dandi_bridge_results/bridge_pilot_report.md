# DANDI 001695 direct bridge pilot

Units: `{'CA3': 21, 'CA1': 80, 'RSC': 9}`; duration `1800.0s`; bin `0.05s`.

| path | rank | lag ms | test ΔNLPD base-bridge | shift-bridge | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| CA3→CA1 | 8 | 50 | 0.00803 | 0.00825 | [0.00475, 0.01182] |
| CA1→RSC | 5 | 200 | 0.01209 | 0.01124 | [0.00514, 0.01918] |
| CA1→CA3 | 8 | 50 | 0.03753 | 0.03864 | [0.03182, 0.04341] |
| RSC→CA1 | 8 | 50 | 0.01583 | 0.01874 | [0.01152, 0.02011] |

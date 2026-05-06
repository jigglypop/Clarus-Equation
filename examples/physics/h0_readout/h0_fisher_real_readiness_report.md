# H0 Fisher Real Readiness Gate

## Manifest status

| quantity | value |
|---|---:|
| tracked channels | 4 |
| tracked channel files ready | 4 |
| synthetic channels | 4 |
| real channels | 0 |
| real-ready channels | 0 |

## Real covariance roadmap

| priority | channel class | expected role | required source | current status | next action |
|---:|---|---|---|---|---|
| 1 | BAO+SN inverse distance ladder | global standard-ruler closure | public covariance/compressed likelihood with ruler and SN nuisance roles | not tracked in Fisher JSON bundle | convert labelled BAO+SN covariance into observable/local/global role graph |
| 2 | SH0ES-style local distance ladder | local calibrator endpoint closure | public ladder covariance with Cepheid/TRGB/SN calibration blocks | not tracked in Fisher JSON bundle | recover calibration graph instead of final scalar H0 only |
| 3 | GW standard sirens | mixed distance-redshift bridge | event-level distance-redshift posterior or population covariance | synthetic smoke channel only | ingest event/posterior covariance and split distance vs redshift anchors |
| 4 | CMB acoustic-scale inference | early global horizon closure | public parameter covariance/likelihood with acoustic-scale roles | synthetic global-horizon smoke channel only | map acoustic-scale covariance nodes to global horizon role |

## Verdict

Fisher/covariance IO is ready, but real covariance closure is still a data boundary.

Do not promote the q-selector to a real covariance result until `real_ready_channels > 0`.

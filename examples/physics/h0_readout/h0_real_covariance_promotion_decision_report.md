# H0 real covariance promotion decision gate

- passed: `True`
- channels: 4
- real-ready: 0

| file | source | class | IO valid | source URL | version pin | promotion | blockers |
|---|---|---|---|---|---|---|---|
| `gw_like_fisher.json` | `synthetic` | `mixed_distance_redshift_bridge` | `True` | `False` | `False` | `not-promoted` | synthetic source |
| `gw_like_covariance.json` | `synthetic` | `mixed_distance_redshift_bridge` | `True` | `False` | `False` | `not-promoted` | synthetic source |
| `cmb_global_fisher.json` | `synthetic` | `global_horizon` | `True` | `False` | `False` | `not-promoted` | synthetic source |
| `local_endpoint_fisher.json` | `synthetic` | `local_endpoint` | `True` | `False` | `False` | `not-promoted` | synthetic source |

## Verdict

Current channels pass IO but are not promoted as real covariance evidence because they are synthetic.

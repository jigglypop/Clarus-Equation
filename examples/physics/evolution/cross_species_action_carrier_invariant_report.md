# Cross-species action carrier invariant

This gate reads existing species reports and asks whether action output is repeatedly carried by a restricted target-linked channel.

## summary

- passed: `True`
- passed stages: 4/4
- invariant: action output is carried by a restricted, weighted, target-linked channel; the carrier becomes more specialized from domain flow to action/memory loop to direction channel to split mammalian unit/probe carriers

## stage evidence

| stage | carrier | status | key evidence | passed |
|---|---|---|---|---|
| C. elegans | weighted stimulus-output domain channel | `proxy` | 8/8 weighted stages; mean matched/wrong 3.213504 | `True` |
| Drosophila adult | celltype/action/memory loop | `connectome` | memory/action loop observed/random 3.738545; p 0.012987 | `True` |
| Zebrafish | left/right perturbation-to-direction activity channel | `discrete_direction` | behavior effect ratio 157.562119; activity AUC 1.000000 | `True` |
| Mouse IBL | split speed probe00/block and wheel probe00/top16 carrier | `candidate_panel` | full 9/12; drop_top_ccf 4/11; drop_probe 3/6; only_probe 7/9; only_top_units 6/9; full 8/12; drop_top_ccf 10/11; drop_probe 4/6; only_probe 6/9; only_top_units 7/9 | `True` |

## caveats

- C. elegans evidence is connectome proxy, not trial behavior.
- Drosophila evidence is structural connectome, not trial dynamics.
- Zebrafish evidence is discrete laser/activity direction, not continuous movement.
- Mouse evidence is an IBL candidate-panel carrier split, not a universal mammalian atlas.

## equation update

$$
\boxed{
\mathcal A_{\mathrm{carrier}}
:
d\mapsto C_d,
\qquad
y_{d,t}=g_d(C_d(t),X_t,\hat H_t)
}
$$

The invariant is not a fixed anatomical location. It is a rule: action variables become readable when the model respects the stage-specific carrier.

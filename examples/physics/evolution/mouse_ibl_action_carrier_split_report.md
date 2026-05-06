# Mouse IBL/OpenAlyx speed/wheel action carrier split

This meta-gate reads the already-generated action ablation reports and asks whether speed and wheel use the same carrier.

## inputs

- region/top-block ablation: `mouse_ibl_action_subspace_region_ablation_report.md`
- probe00 ablation: `mouse_ibl_action_subspace_probe_ablation_report.md`
- fold-local top-unit sufficiency: `mouse_ibl_action_top_unit_sufficiency_report.md`

## carrier verdict

| target | carrier | key pattern | passed |
|---|---|---|---|
| `first_movement_speed` | probe00/block-distributed speed carrier | full 9/12; drop_top_ccf 4/11; drop_probe 3/6; only_probe 7/9; only_top_units 6/9 | `True` |
| `wheel_action_direction` | compact fold-local probe00 top-unit wheel carrier | full 8/12; drop_top_ccf 10/11; drop_probe 4/6; only_probe 6/9; only_top_units 7/9 | `True` |

## interpretation

- Speed is probe00/block-dependent but not closed by the fold-local top 16 probe00 units.
- Wheel is compact enough to close on fold-local top 16 probe00 units, and the top anatomical block is not required.
- The split passes because the expected carrier patterns differ for both targets.

## equation update

$$
\boxed{
\Phi_{\mathrm{action},t}^{(s)}
=
\Phi_{\mathrm{speed},t}^{(s,\mathrm{probe00/block})}
\oplus
\Phi_{\mathrm{wheel},t}^{(s,\mathrm{probe00/top16})}
}
$$

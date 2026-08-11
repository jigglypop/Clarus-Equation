# V5 development-only route pilots

Status: COMPLETE

These calculations use only the 20 historical V5 validation seeds.  They are
route-selection data, not V7 evidence.  Chart errors were normalized by the
training-only standard deviations
`[0.5328761311, 0.5650979567, 1.8315263080, 1.5086286134]`.

## R2 scalar latent reliability -- rejected

- mean fitted alpha: `0.915606` (range `0.601308..1.0`; 7/20 at 1)
- scaled sparse H20 normalized RMSE: `0.503290`
- unscaled V5 parent: `0.501682`
- persistence: `0.501760`
- stable adaptive dense: `0.447237`
- paired improvement versus V5: `-0.001607`, lower 95% `-0.011580`

The single gain worsened the parent and was removed from consideration before
any V7 registration or seed was opened.

## R1 symmetric three-expert consensus -- closure candidate only

- sparse/adaptive/persistence mean weights: `0.349168/0.307410/0.343422`
- sparse consensus H20 normalized RMSE: `0.448700`
- unscaled V5 parent: `0.501682`
- persistence: `0.501760`
- stable adaptive dense: `0.447237`
- symmetrically reweighted dense consensus: `0.448983`
- no-sparse adaptive+persistence consensus: `0.441678`
- paired lower 95% versus V5/persistence at n=20: `-0.005836/-0.003009`
- paired mean sparse contribution versus no-sparse: `-0.007022`

The consensus may repair mean parent error at larger n, but it cannot support a
sparse-causal contribution claim on development data.  It is retained only as
a fresh-seed closure/falsification gate with the no-sparse comparison primary.

## R3 hard prefix selector -- rejected

- H20 normalized RMSE: `0.454244`
- stable adaptive dense: `0.447237`
- target-window oracle ceiling: `0.384479` (diagnostic only)
- prefix/target winner agreement: `0.40`
- paired improvement versus adaptive dense: `-0.007007`

The prefix ranker did not predict seed-level winners reliably and was removed.

No V7 validation or test seeds were simulated in these pilots.

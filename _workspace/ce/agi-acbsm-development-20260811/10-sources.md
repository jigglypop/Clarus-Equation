# Sources

Status: COMPLETE

No external data or new development/test block was used. The model was built
from the committed V8 synthetic family, inherited observational-training
episodes 45100..45107, frozen sparse/dense mechanisms, and training-only
normalization scales.

The preliminary score uses eight leave-one-episode-out folds. Every held
episode is excluded from residual-dynamics fitting, and the episode—not its 22
rolling H20 windows—is the independent summary unit.

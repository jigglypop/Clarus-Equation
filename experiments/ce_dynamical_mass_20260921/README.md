# CE-JS3: dynamical mass, minimal bulk degree, and the remaining input

The authoritative proof is
[chapter 32](../../paper/후속연구_기록과_상태선택/32_동적_질량의_최소_안정화_차수와_남는_자유도.md).

The bulk-monomial model has an explicit shrinking-radius counterpath for degree
below five, and a compact negative sublevel/global-minimum proof for degree above
five. The minimum sufficient even degree in this class is six. Degree five and
different localization/scaling assumptions are not classified by that theorem.

The new scalar has a unique positive conditional mass minimum at fixed radius.
The supplied six-variable local certificate is for the **two-light/one-heavy**
branch. The desired one-light/two-heavy branch remains a saddle at the same
coefficient. Do not present dynamical mass generation as selection of all natural
constants: Yukawa, sextic and lower-term matching remain inputs.

From the repository root:

    .venv\Scripts\python.exe experiments/ce_dynamical_mass_20260921/derive_dynamic_mass.py
    .venv\Scripts\python.exe experiments/ce_dynamical_mass_20260921/certify_dynamic_minimum.py

The first script imports the existing JS2 kernel and records that source hash.
The second independently encloses the infinite six-variable function using
outward intervals, analytic tails and the chapter's third-derivative bound.
The exploratory coefficient scan is not a preregistered observational test.
Cutoff validity, higher-loop matching and the origin of the new operator are open.
The added scalar's own KK winding is at the same loop order. Its nonzero force
at the displayed point is reported separately; it is excluded from the local
certificate. The global degree/existence proof survives this extra bounded
negative determinant, but the displayed stationary point is not the stationary
point of that enlarged function.

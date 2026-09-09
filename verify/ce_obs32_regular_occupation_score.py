"""Conditional cold regular branches, not a matched RT31 cosmology."""
import json
from pathlib import Path
import numpy as np
from ce_obs32_occupation_background import ce, OccupationSpectrum, OccupationBackground, matter_stationary_modes


def run():
    fractions = json.loads(Path('verify/ce_rt31_occupation_bridge.json').read_text())['occupation_fractions']
    *_, zs, observed, cov = ce.load_data()
    chol = np.linalg.cholesky(cov)
    def prediction(bg):
        return np.array([bg.distances(z)['F_AP'] for z in zs])
    def score(pred):
        residual = np.linalg.solve(chol, pred-observed)
        return float(np.sqrt(residual@residual/len(zs)))
    baseline = score(prediction(ce.Background(ce.Parameters(.35, 0.))))
    sp = OccupationSpectrum(ce.Parameters(.35, 0.), fractions)
    rows = []
    for mode in matter_stationary_modes(sp):
        starts = []
        predictions = []
        for ai in [.02, .01, .005]:
            displacement = mode['forced_a3_coefficient']*ai**3
            p = ce.Parameters(.35, mode['theta']+displacement, ai=ai)
            bg = OccupationBackground(p, fractions, initial_velocity=3*displacement)
            pred = prediction(bg)
            predictions.append(pred)
            starts.append(dict(ai=ai, AP_rmse=score(pred), final_theta=bg.quantities(0.)['theta']))
        error = float(np.max(abs(predictions[-1]-predictions[-2])))
        assert error < 1e-7
        # Independent stiff solver checks the full six-dimensional prediction.
        alternate = OccupationBackground(p, fractions, initial_velocity=3*displacement, method='Radau')
        solver_error = float(np.max(abs(prediction(alternate)-predictions[-1])))
        assert solver_error < 1e-8
        rows.append(dict(stationary_theta=mode['theta'], starts=starts,
                         AP_prediction=predictions[-1].tolist(), delta_AP_rmse=starts[-1]['AP_rmse']-baseline,
                         initial_time_refinement_max_AP_difference=error, independent_solver_max_AP_difference=solver_error))
    return dict(baseline_AP_rmse=baseline, cases=rows, fitted_parameters=[],
                assumptions=['Freeze RT31 t25 population ratios; omit coherence and momentum pressure.',
                             'Use OBS32 scales and matter coefficient, not RT31 scales or absolute abundance.',
                             'Set all homogeneous mode amplitudes to zero; this is not derived from generation.',
                             'Report both stationary branches without selecting the better score.'],
                full_joint_rmse=None, state_matching_complete=False, independent_holdout=False)


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

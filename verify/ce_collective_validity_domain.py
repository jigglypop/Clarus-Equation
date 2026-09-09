"""Intersect disclosed approximation and spatial-scale requirements.

No observational optimizer; required stiffness is a conditional lower bound,
not a prediction or an adopted benchmark. Both EFT approximation tolerances
and diagnostic spatial scales are explicit.
"""
import json
from pathlib import Path
import numpy as np


def run():
    folder=Path(__file__).resolve().parent
    ap=json.loads((folder/'ce_collective_ap.json').read_text())
    scales=json.loads((folder/'ce_collective_scale_relation.json').read_text())
    mass_cases={row['N']:row for row in ap['mass_exchange_cases']}
    rows=[]
    for scale in scales['rows']:
        N=scale['N'];case=mass_cases[N]
        for amplitude_limit in [.1,.03]:
            # theta excursion ~ R^-1/2, m/H ~ R^1/2, lambda_J ~ R^-1/4.
            amp_floor=128*(case['initial_harmonic_link_excursion']/amplitude_limit)**2
            adiabatic_floor=128*(100/case['initial_gear_mass_over_H'])**2
            for length in [1.,.01]:
                jeans_floor=128*(scale['jeans_wavelength_Mpc']/length)**4
                lower=max(amp_floor,adiabatic_floor,jeans_floor)
                labels={'small_amplitude':amp_floor,'rapid_oscillation':adiabatic_floor,
                        'spatial_scale':jeans_floor}
                rows.append(dict(N=N,maximum_link_amplitude=amplitude_limit,
                                 minimum_initial_mass_over_H=100,target_Jeans_length_Mpc=length,
                                 lower_bounds=labels,combined_R_H_lower_bound=lower,
                                 controlling_condition=max(labels,key=labels.get),
                                 local_f_GeV=scale['local_f_GeV']))
    return dict(inputs_source=['ce_collective_ap.json','ce_collective_scale_relation.json'],
                rows=rows,selected_N=None,selected_R_H=None,observational_rmse=None,
                scope='Leading harmonic scaling constraints, not full nonlinear sufficiency.',
                limitations=['Amplitude and frequency tolerances are computational criteria, not observations.',
                             'Changing R changes small finite-lock normalization corrections omitted in these estimates.',
                             'Initial state, UV stability, coupled perturbations and all-force matching remain required.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in result['rows'] if r['maximum_link_amplitude']==.1
                      and r['target_Jeans_length_Mpc']==.01],indent=2))

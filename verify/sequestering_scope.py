"""Algebraic scope of arXiv:1505.01492v2 eq. (10), not a CE completion."""
import json
from pathlib import Path
import sympy as sp


def run():
    v, average_v, rho, average_rho, residual, shift = sp.symbols(
        'v average_v rho average_rho residual shift')
    stress = sp.diag(-rho-v, -v, -v, -v)
    source = stress-sp.eye(4)*((-average_rho-4*average_v)/4+residual)
    shifted = source.subs({v:v+shift, average_v:average_v+shift}, simultaneous=True)
    assert sp.simplify(shifted-source) == sp.zeros(4)
    density = sp.expand(-source[0,0])
    assert sp.diff(density, v) == 1
    assert sp.diff(density, residual) == 1
    return dict(source='https://arxiv.org/html/1505.01492v2', equation=10,
        scope='pressureless matter plus local potential; no solved spacetime average or flux',
        effective_density=str(density), constant_shift_cancels=True,
        local_potential_dependence_at_fixed_average=1, residual_constant_dependence=1,
        joint_rmse=None, fitted_parameters=0,
        conclusion='Constant vacuum shifts cancel, but local potential variation and an undetermined residual remain.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

"""Independent numerical checks for the derivations added to the main paper.

Checks conditional identities, not empirical validity or a state-selection law.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np


def run():
    rows = []
    with mp.workdps(60):
        for r_string in ['0.15', '0.35']:
            eps = mp.mpf(r_string)
            def xs(s, t):
                return [s+2*eps*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
            def U(s, t):
                def raw(t1):
                    return sum(x*x*(mp.log(x)-mp.mpf('1.5')) for x in xs(s,t1))
                return (raw(t)-raw(mp.pi))/(32*mp.pi**2)
            mass = lambda t: sum(mp.sqrt(x) for x in xs(1,t)[1:])/(2*mp.sqrt(1-eps))
            curvature_error = abs(mp.diff(mass, 0, 2)-eps*(2-5*eps)/(36*(1-eps)**2))
            assert curvature_error < mp.mpf('1e-50')
            for theta_string in ['0', '0.7', '1.8']:
                theta = mp.mpf(theta_string)
                x = xs(1,theta)
                xx = np.array([float(v) for v in x])
                shift = np.roll(np.eye(3),1,axis=1)
                matrix = np.eye(3)+float(eps)*(np.exp(1j*float(theta)/3)*shift+
                                               np.exp(-1j*float(theta)/3)*shift.T)
                eigen_error = float(np.max(abs(np.linalg.eigvalsh(matrix)-np.sort(xx))))
                determinant_error = abs(mp.fprod(x)-(1-3*eps**2+2*eps**3*mp.cos(theta)))
                moment_error = max(abs(sum(x)-3), abs(sum(v*v for v in x)-(3+6*eps**2)))
                uss = mp.log(mp.fprod(x)/mp.fprod(xs(1,mp.pi)))/(16*mp.pi**2)
                usss = (sum(1/v for v in x)-sum(1/v for v in xs(1,mp.pi)))/(16*mp.pi**2)
                derivative_error = max(abs(mp.diff(lambda s: U(s,theta),1,2)-uss),
                                       abs(mp.diff(lambda s: U(s,theta),1,3)-usss))
                # Integrate the spectral density directly over nu up to infinity.
                A = [-2*eps*mp.sin((theta+2*mp.pi*j)/3)/3 for j in range(3)]
                Z_integral = sum(aj**2/(8*mp.pi**2)*mp.quad(
                    lambda nu: mp.sqrt(1-4*v/nu**2)/nu**3,
                    [2*mp.sqrt(v),4*mp.sqrt(v),mp.inf]) for aj,v in zip(A,x))
                Z_local = sum(aj**2/v for aj,v in zip(A,x))/(96*mp.pi**2)
                Z_error = abs(Z_integral-Z_local)
                assert eigen_error < 1e-14
                assert max(moment_error,determinant_error,derivative_error,Z_error) < mp.mpf('1e-50')
                rows.append(dict(r=float(eps),theta=float(theta),matrix_eigenvalue_error=eigen_error,
                                 moment_error=float(moment_error),determinant_error=float(determinant_error),
                                 fixed_epsilon_derivative_error=float(derivative_error),
                                 spectral_integral_Z_error=float(Z_error),
                                 symmetric_mass_curvature_error=float(curvature_error)))
    return dict(precision_decimal_digits=60,rows=rows,
                scope='Conditional algebra and spectral integral; no observational fit or physical proof.')


if __name__ == '__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

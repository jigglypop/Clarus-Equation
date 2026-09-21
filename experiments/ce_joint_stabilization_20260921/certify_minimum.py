"""Outward interval enclosure of the JS2 five-variable local minimum.

This certifies the stated infinite one-loop function, not loop truncation error
or the choice of EFT. The analytic third-derivative bound is derived in chapter 31.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath
import numpy as np
from mpmath import iv

from derive_stabilization import Model


def lower(x):
    return float(np.nextafter(float(x.a), -np.inf))


def upper(x):
    return float(np.nextafter(float(x.b), np.inf))


def absolute_upper(x):
    return max(abs(lower(x)), abs(upper(x)))


def certify():
    here = Path(__file__).resolve().parent
    data = json.loads((here/'results.json').read_text(encoding='utf-8'))
    assert data['inputs']['neutral_Dirac_mass'] == .5
    assert data['source_sha256'] == hashlib.sha256((here/'derive_stabilization.py').read_bytes()).hexdigest()
    iv.dps = 45
    nmax, massive_cutoff = 8192, 64
    u, r, _, q, _ = [iv.mpf(str(v)) for v in data['point']]
    y = iv.mpf('.18')*u*iv.exp(2*r)
    p = 3/(64*iv.pi**6)*iv.exp(-6*r)
    zero = iv.mpf(0)
    w0 = wy = wyy = sn = sr = srr = zero
    wa, way = [zero]*3, [zero]*3
    waa = [[zero for _ in range(3)] for _ in range(3)]
    for n in range(1, nmax+1):
        wn, weight = 2*iv.pi*n, iv.mpf(1)/(n**5)
        ct, st = iv.cos(wn*q), iv.sin(wn*q)
        trace = 1+2*ct
        f = fy = fyy = zero
        if n <= massive_cutoff:
            z = wn*iv.sqrt(y)
            ez = iv.exp(-z)
            f = ez*(1+z+z*z/3)
            fy = -wn**2*(1+z)*ez/6
            fyy = wn**4*ez/12
            zn = iv.pi*n*iv.exp(r)
            en = iv.exp(-zn)
            sn += weight*en*(1+zn+zn*zn/3)
            sr += weight*(-en*zn*zn*(1+zn)/3)
            srr += weight*en*(zn**4-2*zn**3-2*zn**2)/3
        w0 += weight*(trace*trace-1-2*f*trace)
        wy += -2*weight*fy*trace
        wyy += -2*weight*fyy*trace
        ga = -2*wn*st*(trace-f)*weight
        gya = 2*fy*wn*st*weight
        wa[1] += ga
        wa[2] -= ga
        way[1] += gya
        way[2] -= gya
        diagonal0 = 2*wn*wn*(f-2*ct)*weight
        diagonal1 = 2*wn*wn*(1-(trace-f)*ct)*weight
        off01 = 2*wn*wn*ct*weight
        off12 = 2*wn*wn*(2*ct*ct-1)*weight
        local = [[diagonal0, off01, off01], [off01, diagonal1, off12],
                 [off01, off12, diagonal1]]
        for i in range(3):
            for j in range(3):
                waa[i][j] += local[i][j]
    er, dy = iv.exp(-r), y/u
    h, hp = 10*(u-iv.mpf('.5'))**2, 20*(u-iv.mpf('.5'))
    full = w0+4*sn
    gradient = [er*hp+p*dy*wy, -er*h+p*(-6*full+2*y*wy+4*sr)]+[p*v for v in wa]
    matrix = [[zero for _ in range(5)] for _ in range(5)]
    matrix[0][0] = 20*er+p*dy*dy*wyy
    matrix[0][1] = matrix[1][0] = -er*hp+p*dy*(-4*wy+2*y*wyy)
    matrix[1][1] = er*h+p*(36*full-20*y*wy+4*y*y*wyy-48*sr+4*srr)
    for i in range(3):
        matrix[0][i+2] = matrix[i+2][0] = p*dy*way[i]
        matrix[1][i+2] = matrix[i+2][1] = p*(-6*wa[i]+2*y*way[i])
        for j in range(3):
            matrix[i+2][j+2] = p*waa[i][j]
    # The floating eigenvectors only select a basis. Interval Gram/Gershgorin
    # inequalities below certify that this basis is invertible and positive.
    _, _, numerical_hessian, _ = Model(nmax=nmax, mass=.5).evaluate(data['point'])
    _, qmatrix = np.linalg.eigh(numerical_hessian)
    basis = [[iv.mpf(str(qmatrix[i, j])) for j in range(5)] for i in range(5)]
    rotated, gram = [], []
    for i in range(5):
        rotated.append([sum(basis[k][i]*matrix[k][l]*basis[l][j]
                            for k in range(5) for l in range(5)) for j in range(5)])
        gram.append([sum(basis[k][i]*basis[k][j] for k in range(5)) for j in range(5)])
    def gersh_lower(a):
        return min(lower(a[i][i])-sum(absolute_upper(a[i][j]) for j in range(5) if i != j)
                   for i in range(5))
    def gersh_upper(a):
        return max(upper(a[i][i])+sum(absolute_upper(a[i][j]) for j in range(5) if i != j)
                   for i in range(5))
    # Conservative final outward padding for the few binary64 additions/divisions.
    gram_min, gram_max = gersh_lower(gram)-1e-14, gersh_upper(gram)+1e-14
    assert gram_min > .99
    center_lower = gersh_lower(rotated)/gram_max-1e-13
    # On the stated box, all massive value/gradient/Hessian tails at 64 are
    # bounded by 10^8 sum_{n>64} n^4 exp(-2n), conservatively below 10^-36.
    massive_tail = 1e-36
    gradient_tail = upper(p*iv.sqrt((iv.mpf(12)/nmax**4)**2+
                                   3*(4*iv.pi/nmax**3)**2))+massive_tail
    hessian_tail = upper(p*(iv.mpf(72)/nmax**4+48*iv.sqrt(3)*iv.pi/nmax**3+
                            12*iv.pi**2/nmax**2))+massive_tail
    gradient_bound = float(np.nextafter(np.sqrt(sum(absolute_upper(v)**2 for v in gradient)), np.inf))
    gradient_bound += gradient_tail+1e-25
    radius, third_derivative_bound = 1e-8, 60.
    convexity = center_lower-hessian_tail-third_derivative_bound*radius-1e-15
    assert convexity > 0
    assert gradient_bound < convexity*radius
    assert .49 < data['point'][0]-radius < data['point'][0]+radius < .51
    assert .10 < data['point'][1]-radius < data['point'][1]+radius < .14
    distance = float(np.nextafter(gradient_bound/convexity, np.inf))
    report = {
        'claim': 'one strict local minimum of the infinite JS2 winding potential in a five-dimensional ball',
        'center': data['point'], 'ball_radius': radius,
        'interval_decimal_precision': iv.dps, 'massless_winding_cutoff': nmax,
        'massive_winding_cutoff': massive_cutoff, 'massive_tail_bound': massive_tail,
        'interval_gradient': [str(v) for v in gradient],
        'interval_Hessian': [[str(v) for v in row] for row in matrix],
        'basis_Gram_lower': gram_min, 'basis_Gram_upper': gram_max,
        'center_Hessian_lower': center_lower, 'Hessian_tail_bound': hessian_tail,
        'gradient_tail_bound': gradient_tail, 'total_gradient_upper': gradient_bound,
        'third_derivative_bound_on_box': third_derivative_bound,
        'box_u': [.49, .51], 'box_r': [.10, .14],
        'full_ball_Hessian_lower': convexity, 'boundary_outward_margin': convexity*radius-gradient_bound,
        'distance_to_unique_stationary_point_upper': distance, 'certified': True,
        'limitations': ['specified one-loop EFT only', 'not global uniqueness', 'not a natural-value prediction',
                        'mpmath interval computation, not a proof-assistant formalization'],
        'mpmath_version': mpmath.__version__,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source_sha256': data['source_sha256'],
        'input_results_sha256': hashlib.sha256((here/'results.json').read_bytes()).hexdigest()}
    (here/'minimum_certificate.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['certified', 'full_ball_Hessian_lower',
                                                  'total_gradient_upper', 'boundary_outward_margin',
                                                  'distance_to_unique_stationary_point_upper']}, indent=2))


if __name__ == '__main__':
    certify()

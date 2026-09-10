"""CE-REL-LINK1: exact quantum geometry, projection costs and force response.

Coordinates, Hamiltonian, gap, external kinetic mass and preparation are inputs.
The results do not identify a gauge sector with a fundamental force or gravity.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as s

from ce_dimension_correspondence import exact_rectangle, frame_geometry
from ce_state_dimensions import exact_matrix, is_zero


def projected_bridge(frame, derivatives, mass=1):
    """Pointwise Stiefel geometry and flat-space projected kinetic scalar, hbar=1."""
    geom = frame_geometry(frame, derivatives)
    v = s.Matrix(frame)
    mass = s.sympify(mass)
    if mass.is_positive is not True or mass.has(s.Float) or mass.free_symbols:
        raise ValueError('mass must be an exact positive number')
    q = s.eye(v.rows)-v*v.H
    bs = [q*s.Matrix(a) for a in derivatives]
    phi = s.simplify(sum((b.H*b for b in bs), s.zeros(v.cols))/(2*mass))
    bounds = []
    for (i, j), f in geom['curvature'].items():
        determinant = s.simplify(geom['metric'][i, i]*geom['metric'][j, j]-geom['metric'][i, j]**2)
        residual = s.simplify(4*determinant-s.trace(f.H*f))
        bounds.append({'plane': (i, j), 'gram_determinant': determinant,
                       'frobenius_bound_residual': residual})
    return {**geom, 'born_huang_scalar_hbar1': phi, 'curvature_area_bounds': bounds}


def spectral_bridge(hamiltonian, state, energy, h_derivatives):
    """QGT via the exact reduced resolvent of a witnessed simple eigenstate.

    The spectral formula works for any simple eigenvalue; an adiabatic
    preparation and slow driving are additionally required for force response.
    """
    h = exact_matrix(hamiltonian, 'Hamiltonian')
    psi = exact_rectangle(state, 'state')
    if psi.shape != (h.rows, 1) or not is_zero((psi.H*psi)[0]-1):
        raise ValueError('state must be a normalized column of Hamiltonian size')
    energy = s.sympify(energy)
    if energy.free_symbols or energy.has(s.Float) or energy.is_real is not True or energy.is_finite is not True:
        raise ValueError('energy must be exact finite real')
    if any(not is_zero(a) for a in h*psi-energy*psi):
        raise ValueError('state is not an eigenstate at the supplied energy')
    derivatives = [exact_matrix(a, 'Hamiltonian derivative') for a in h_derivatives]
    if any(a.shape != h.shape for a in derivatives):
        raise ValueError('Hamiltonian derivative dimensions differ')
    p = psi*psi.H
    q = s.eye(h.rows)-p
    augmented = h-energy*s.eye(h.rows)+p
    if is_zero(augmented.det()):
        raise ValueError('degenerate eigenvalue: simple-eigenstate branch is undefined')
    resolvent = s.simplify(q*augmented.inv()*q)
    state_derivatives = [-resolvent*a*psi for a in derivatives]
    out = projected_bridge(psi, state_derivatives)
    tensor = s.Matrix(len(derivatives), len(derivatives),
                      lambda a, b: s.simplify((psi.H*derivatives[a]*resolvent**2*derivatives[b]*psi)[0]))
    out.update({'qgt': tensor, 'reduced_resolvent': resolvent})
    return out


def ricci_from_metric(metric, coordinates):
    """Direct Christoffel contraction, independent of product-space formulas."""
    metric = s.Matrix(metric)
    n = len(coordinates)
    if metric.shape != (n, n) or metric != metric.T:
        raise ValueError('metric must be a symmetric coordinate-sized matrix')
    inv = metric.inv()
    gamma = [[[s.simplify(sum(inv[i, a]*(s.diff(metric[a, j], coordinates[k])
                  + s.diff(metric[a, k], coordinates[j])-s.diff(metric[j, k], coordinates[a]))
                  for a in range(n))/2) for k in range(n)] for j in range(n)] for i in range(n)]
    ricci = s.Matrix(n, n, lambda j, k: s.trigsimp(sum(
        s.diff(gamma[a][j][k], coordinates[a])-s.diff(gamma[a][j][a], coordinates[k])
        + sum(gamma[a][a][b]*gamma[b][j][k]-gamma[a][k][b]*gamma[b][j][a]
              for b in range(n)) for a in range(n))))
    return ricci, s.trigsimp(s.trace(inv*ricci))


def run_audit():
    root = Path(__file__).resolve().parents[1]
    ledger = root/'paper/검증_원장/관계기하_힘_연결_조사원장.md'
    prefix = ledger.read_text(encoding='utf-8').split('## 2.')[0].rstrip()+'\n'
    prereg_sha = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def eq(a, b):
        if isinstance(a, s.MatrixBase):
            return a.shape == b.shape and all(is_zero(v) for v in a-b)
        return is_zero(a-b)

    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)

    check('frozen preregistration', prereg_sha == '6d50d79baa35a32333f737a57ab43253f7f6126dd445b64465eeee84b4b90959')
    ledger_text = ledger.read_text(encoding='utf-8')
    extra_prereg = {}
    for section, expected in [(2, '91007a01cd1efb4b177566debc4b1c53a2c19fc78a95e3849aa4a199430f6431'),
                              (3, '5ed9f545645703b403f4e47b874b5b76fb916fbf845aa3e2e08fc2509d60c7bf')]:
        heading = f'## {section}.'
        block = (heading+ledger_text.split(heading, 1)[1]).split(f'## {section+1}.', 1)[0].rstrip()+'\n'
        digest = hashlib.sha256(block.encode('utf-8')).hexdigest()
        check(f'frozen additional preregistration section {section}', digest == expected)
        extra_prereg[str(section)] = digest
    x, y, z = s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)
    zero, one = s.Matrix([1, 0]), s.Matrix([0, 1])
    theta, phi = s.symbols('theta phi', real=True)
    psi = s.Matrix([s.cos(theta/2), s.exp(s.I*phi)*s.sin(theta/2)])
    q = s.eye(2)-psi*psi.H
    ds = [psi.diff(theta), psi.diff(phi)]
    tensor = s.Matrix(2, 2, lambda i, j: s.simplify((ds[i].H*q*ds[j])[0]))
    metric = tensor.applyfunc(lambda a: s.trigsimp(s.re(a)))
    f = s.trigsimp(-2*s.im(tensor[0, 1]))
    check('sphere metric from symbolic normalized state', eq(metric, s.diag(s.Rational(1, 4), s.sin(theta)**2/4)))
    check('sphere curvature sign convention', eq(f, -s.sin(theta)/2))
    check('metric curvature determinant inequality saturates', eq(metric.det(), f*f/4))
    # On the sphere chart 0<theta<pi, the area element is sin(theta)/4.
    area = s.integrate(s.sin(theta)/4, (theta, 0, s.pi))*2*s.pi
    chern = s.integrate(f, (theta, 0, s.pi))
    check('topological area lower bound, C=-1 and area=pi', area == s.pi and chern == -1)
    sigma_energy = s.integrate(s.sin(theta)/2, (theta, 0, s.pi))*2*s.pi
    check('unit-sphere sigma energy saturates twice the Chern area bound', sigma_energy == 2*s.pi*abs(chern))
    # The north chart ket need not be global; its projector is smooth at both poles.
    projector = s.simplify(psi*psi.H)
    check('sphere projector at chart poles', eq(projector.subs(theta, 0), zero*zero.H)
          and eq(projector.subs(theta, s.pi), one*one.H))

    # CE-REL-PROD4, preregistered separately: add the second qubit phase.
    eta, xi = s.symbols('eta xi', real=True)
    second = s.Matrix([s.cos(eta/2), s.exp(s.I*xi)*s.sin(eta/2)])
    product = s.kronecker_product(psi, second)
    coordinates = [theta, phi, eta, xi]
    point = {theta: s.pi/3, phi: 0, eta: s.pi/6, xi: 0}
    prod_geom = projected_bridge(product.subs(point), [product.diff(a).subs(point) for a in coordinates])
    g4 = s.diag(1, s.sin(theta)**2, 1, s.sin(eta)**2)/4
    check('product state four-dimensional metric from exact jets', eq(prod_geom['metric'], g4.subs(point)))
    check('product Berry curvature in first sphere', eq(prod_geom['curvature'][0, 1], s.Matrix([[-s.sqrt(3)/4]])))
    check('product Berry curvature in second sphere', eq(prod_geom['curvature'][2, 3], s.Matrix([[-s.Rational(1, 4)]])))
    check('product mixed Berry planes vanish', all(eq(prod_geom['curvature'][i, j], s.zeros(1))
          for i, j in [(0, 2), (0, 3), (1, 2), (1, 3)]))
    riccis = []
    for k, expected_scalar in enumerate([0, 8, 8, 16], start=1):
        ricci, scalar_curvature = ricci_from_metric(g4[:k, :k], coordinates[:k])
        riccis.append(ricci)
        check(f'product restriction k={k} Ricci scalar', eq(scalar_curvature, expected_scalar))
    check('sphere Ricci tensor is four times metric', eq(riccis[1], 4*metric))
    check('sphere scalar curvature area density matches minus four Berry form', eq(8*s.sin(theta)/4, -4*f))
    check('four-dimensional product is Euclidean Einstein', eq(riccis[3], 4*g4))
    fprod = s.zeros(4)
    fprod[0, 1], fprod[1, 0] = -s.sin(theta)/2, s.sin(theta)/2
    fprod[2, 3], fprod[3, 2] = -s.sin(eta)/2, s.sin(eta)/2
    inv4 = g4.inv()
    raised = inv4*fprod*inv4
    volume = s.sin(theta)*s.sin(eta)/16  # positive chart 0<theta,eta<pi
    dual = s.Matrix(4, 4, lambda i, j: s.simplify(volume*sum(
        s.LeviCivita(i, j, a, b)*raised[a, b] for a in range(4) for b in range(4))/2))
    check('product Berry form is Euclidean self-dual', eq(dual, fprod))
    check('product Berry form is closed', all(eq(s.diff(fprod[j, k], coordinates[i])
          + s.diff(fprod[k, i], coordinates[j])+s.diff(fprod[i, j], coordinates[k]), 0)
          for i in range(4) for j in range(4) for k in range(4)))
    norm2 = s.simplify(sum(fprod[i, j]*raised[i, j] for i in range(4) for j in range(4)))
    stress = s.simplify(fprod*inv4*fprod.T-g4*norm2/4)
    check('Euclidean self-dual field norm and zero Maxwell stress', eq(norm2, 16) and eq(stress, s.zeros(4)))
    check('supplied Einstein Maxwell equations with Lambda four', eq(riccis[3]-g4*16/2+4*g4, stress))
    wedge = 2*(fprod[0, 1]*fprod[2, 3]-fprod[0, 2]*fprod[1, 3]+fprod[0, 3]*fprod[1, 2])
    wedge_integral = s.integrate(wedge, (theta, 0, s.pi), (eta, 0, s.pi))*(2*s.pi)**2
    check('product first Chern class square is two', eq(wedge_integral/(2*s.pi)**2, 2))
    check('product second Chern character integral is one', eq(wedge_integral/(8*s.pi**2), 1))
    ym_action = s.integrate(norm2*volume/4, (theta, 0, s.pi), (eta, 0, s.pi))*(2*s.pi)**2
    check('Euclidean Yang Mills action saturates topological bound at coupling one', eq(ym_action, abs(wedge_integral)/2) and eq(ym_action, 4*s.pi**2))

    for angle in [s.pi/6, s.pi/3, s.pi/2]:
        pn = psi.subs({theta: angle, phi: 0})
        dn = [a.subs({theta: angle, phi: 0}) for a in ds]
        direct = projected_bridge(pn, dn)
        hn = -(s.sin(angle)*x+s.cos(angle)*z)/2
        hd = [-(s.cos(angle)*x-s.sin(angle)*z)/2, -s.sin(angle)*y/2]
        spec = spectral_bridge(hn, pn, -s.Rational(1, 2), hd)
        check(f'direct derivative versus spectral QGT at {angle}', eq(direct['metric'], spec['metric'])
              and eq(direct['curvature'][0, 1], spec['curvature'][0, 1]))

    for gap in [1, 2, 4]:
        # H(theta,eta)=-gap/2 U_y(theta)(Z+eta Y)U_y(theta)^dagger at zero.
        bridge = spectral_bridge(-gap*z/2, zero, -s.Rational(gap, 2), [-gap*x/2, -gap*y/2])
        check(f'common eigenstates and geometry independent of gap {gap}', eq(bridge['metric'], s.eye(2)/4)
              and eq(bridge['curvature'][0, 1], s.Matrix([[-s.Rational(1, 2)]])))
    for radius in [1, 2, 4]:
        cone = spectral_bridge(radius*z, one, -radius, [x, y, z])
        check(f'cone radial direction null at r={radius}', eq(cone['metric'], s.diag(s.Rational(1, 4*radius**2), s.Rational(1, 4*radius**2), 0)))
    rejects('degenerate cone origin rejected', lambda: spectral_bridge(s.zeros(2), zero, 0, [x, y, z]))
    rejects('non-eigenstate rejected', lambda: spectral_bridge(z, (zero+one)/s.sqrt(2), 0, [x]))
    rejects('invalid external mass rejected', lambda: projected_bridge(zero, [one], 0))

    # Same metric and rank permit zero, nonzero, and non-Abelian curvatures.
    v = s.eye(3)[:, :1]
    e1, e2 = s.eye(3)[:, 1:2], s.eye(3)[:, 2:3]
    for label, bs in [('real', [e1, e2]), ('complex', [e1, s.I*e1]), ('intermediate', [e1, (s.I*e1+e2)/s.sqrt(2)])]:
        bridge = projected_bridge(v, bs)
        check(f'PSD curvature area bound {label}', bridge['curvature_area_bounds'][0]['frobenius_bound_residual'].is_nonnegative is True)
        check(f'common metric rank {label}', eq(bridge['metric'], s.eye(2)))
    v2 = s.eye(3)[:, :2]
    b1, b2 = s.Matrix([[0, 0], [0, 0], [1, 0]]), s.Matrix([[0, 0], [0, 0], [0, 1]])
    nonabelian = projected_bridge(v2, [b1, b2, s.I*b1])
    f12, f13 = nonabelian['curvature'][0, 1], nonabelian['curvature'][0, 2]
    check('curvatures in rank-two frame do not commute', not eq(f12*f13-f13*f12, s.zeros(2)))
    check('all non-Abelian area bounds hold', all(b['frobenius_bound_residual'].is_nonnegative is True for b in nonabelian['curvature_area_bounds']))
    null = projected_bridge(v2, [b1, 2*b1, s.I*v2])
    check('pure gauge null direction annihilates curvature', eq(null['curvature'][0, 2], s.zeros(2))
          and eq(null['curvature'][1, 2], s.zeros(2)))
    check('dependent real directions carry no curvature plane', eq(null['curvature'][0, 1], s.zeros(2)))
    for k in [1, 2, 3, 4]:
        skew = s.zeros(k)
        for j in range(k//2):
            skew[2*j, 2*j+1], skew[2*j+1, 2*j] = 1, -1
        check(f'antisymmetric maximal rank in dimension {k}', skew.rank() == 2*(k//2))
    f4 = s.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    pfaffian = f4[0, 1]*f4[2, 3]-f4[0, 2]*f4[1, 3]+f4[0, 3]*f4[1, 2]
    check('four dimensions allow a nonzero wedge square', pfaffian == 1 and f4.det() == pfaffian**2)

    # Project the differential kinetic operator directly, independently of the norm split.
    coord = s.symbols('q', real=True)
    amplitude = s.Function('a')(coord)
    real_ket = s.Matrix([s.cos(coord**2/2), s.sin(coord**2/2)])
    scalar = s.simplify((real_ket.diff(coord).H*real_ket.diff(coord))[0]/2)
    lhs = s.simplify((real_ket.H*(-(real_ket*amplitude).diff(coord, 2)/2))[0])
    check('real state has zero connection and a spatial scalar potential', eq((real_ket.H*real_ket.diff(coord))[0], 0) and eq(scalar, coord**2/2))
    check('projected kinetic operator retains Born-Huang scalar', eq(lhs, -amplitude.diff(coord, 2)/2+scalar*amplitude))
    check('zero Berry curvature still allows a scalar force', eq(-s.diff(scalar, coord), -coord))
    # Actual CP observation can erase phase geometry; a dilation preserves the whole state.
    theta_state = s.Matrix([1, s.exp(s.I*coord)])/s.sqrt(2)
    rho = theta_state*theta_state.H
    dephased = s.diag(rho[0, 0], rho[1, 1])
    check('unread Z measurement erases local phase distinguishability', eq(dephased.diff(coord), s.zeros(2)))
    joint = s.Matrix([1, 0, 0, s.exp(s.I*coord)])/s.sqrt(2)
    jq = s.eye(4)-joint*joint.H
    check('coherent record dilation preserves global phase metric', eq((joint.diff(coord).H*jq*joint.diff(coord))[0], s.Rational(1, 4)))

    ident = s.eye(2)
    square = [[s.kronecker_product(x, ident), s.kronecker_product(ident, x), s.kronecker_product(x, x)],
              [s.kronecker_product(ident, y), s.kronecker_product(y, ident), s.kronecker_product(y, y)],
              [s.kronecker_product(x, y), s.kronecker_product(y, x), s.kronecker_product(z, z)]]
    contexts = square+[[square[i][j] for i in range(3)] for j in range(3)]
    for index, context in enumerate(contexts):
        check(f'Mermin context {index} commutes', all(eq(a*b, b*a) for a in context for b in context))
        sign = -1 if index == 5 else 1
        check(f'Mermin context {index} product sign', eq(context[0]*context[1]*context[2], sign*s.eye(4)))

    # Schur complement and eliminated-memory source remain distinct from P-only state.
    energy = s.Rational(1, 3)
    h = s.Matrix([[0, 1, 0], [1, 2, 1], [0, 1, 3]])
    eff = h[:1, :1]+h[:1, 1:]*(energy*s.eye(2)-h[1:, 1:]).inv()*h[1:, :1]
    check('projected resolvent equals Schur effective Hamiltonian', eq((energy*s.eye(3)-h).inv()[:1, :1], (energy*s.eye(1)-eff).inv()))

    # Fixed numerical protocol: hbar=gap=1, t in [0,2], 41 common sample times.
    # Solve in lab frame; compare with the independent constant rotating-frame solution.
    xn, yn, zn = [np.array(a, dtype=complex) for a in [x, y, z]]
    initial_bare = np.array([1, 0], dtype=complex)
    samples = np.linspace(0, 2, 41)
    rows, max_error = [], 0.0
    for speed in [1/4, 1/8, 1/16, 1/32]:
        omega = np.sqrt(1+speed**2)
        heff = -(zn+speed*yn)/2
        _, eig = np.linalg.eigh(heff)
        for label, initial in [('dressed', eig[:, 0]), ('abrupt', initial_bare)]:
            def rhs(t, state):
                hlab = -(np.sin(speed*t)*xn+np.cos(speed*t)*zn)/2
                return -1j*hlab@state
            numerical = solve_ivp(rhs, (0, 2), initial, t_eval=samples, method='DOP853', rtol=1e-11, atol=1e-12)
            check(f'ODE completed {label} speed={speed}', numerical.success and numerical.y.shape == (2, len(samples)))
            exact = []
            for time in samples:
                rotation = np.cos(speed*time/2)*np.eye(2)-1j*np.sin(speed*time/2)*yn
                effective = np.cos(omega*time/2)*np.eye(2)-2j*np.sin(omega*time/2)*heff/omega
                exact.append(rotation@effective@initial)
            exact = np.asarray(exact).T
            error = float(np.max(np.abs(numerical.y-exact)))
            max_error = max(max_error, error)
            check(f'lab ODE versus rotating exact {label} speed={speed}', error < 1e-8)
            force = float(np.real(np.vdot(numerical.y[:, -1], yn@numerical.y[:, -1]))/2)
            predicted = speed/(2*omega) if label == 'dressed' else speed/(2*(1+speed**2))*(1-np.cos(2*omega))
            check(f'force at fixed final time {label} speed={speed}', abs(force-predicted) < 1e-8)
            rows.append({'speed': speed, 'preparation': label, 'force': force,
                         'force_over_speed': force/speed, 'exact_force': float(predicted),
                         'adiabatic_linear_coefficient': 0.5, 'state_error': error})
    errors = [abs(r['force_over_speed']-.5) for r in rows if r['preparation'] == 'dressed']
    check('dressed response approaches fixed curvature coefficient', all(b<a for a, b in zip(errors, errors[1:])))
    check('abrupt preparation retains leading transient', abs(rows[-1]['force_over_speed']-.5) > .1)

    report = root/'paper/01_측정과_접힘/15_관계기하와_힘의_연결_문헌과_검증.md'
    sources = [Path(__file__).resolve(), root/'verify/ce_dimension_correspondence.py', root/'verify/ce_state_dimensions.py', ledger]
    if report.exists():
        sources.append(report)
    receipt = {'schema_version': 1, 'checks_passed': len(checks), 'checks': checks,
               'preregistration_sha256': prereg_sha,
               'additional_preregistration_sha256': extra_prereg,
               'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'environment': {'python': platform.python_version(), 'sympy': s.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
               'arithmetic': 'exact rational/symbolic geometry; separate complex ODE check',
               'numerical_tolerance': 1e-8, 'maximum_state_error': max_error,
               'dynamics_protocol': {'gap': 1, 'hbar': 1, 'final_time': 2, 'sample_count': 41, 'fit_parameters': 0},
               'response_rows': rows, 'sphere': {'area': str(area), 'chern': str(chern), 'sigma_energy_alpha1': str(sigma_energy)},
               'product4': {'signature': 'positive Euclidean', 'ell': 1, 'scalar_curvature': 16,
                            'supplied_cosmological_constant': 4, 'self_dual': True, 'maxwell_stress': 0,
                            'integral_F_wedge_F': str(wedge_integral), 'c1_square_integral': 2,
                            'second_chern_character_integral': 1, 'line_bundle_second_chern_class': 0,
                            'yang_mills_action_coupling1': str(ym_action),
                            'qualification': 'same state supplies a conditional Euclidean geometry; action and constants supplied, no Lorentzian derivation'},
               'scientific_success': False, 'full_joint_rmse': None,
               'scope': 'conditional bridge identities and fixed examples, not a completed CE common action',
               'independence': 'same author, alternative symbolic derivations and lab-frame ODE against rotating-frame exact solution'}
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks), 'maximum_state_error': max_error, 'receipt': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    run_audit()

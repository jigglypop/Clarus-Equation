"""CE-RFD1: dimension-specific probe forces, Lorentz stress and dark-sector gates.

All actions and scales are explicit inputs. No observational fit or completed
dark matter / dark energy prediction is claimed. Preregistered scope is hashed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
import sympy as s


def clean(value):
    if isinstance(value, s.MatrixBase):
        return value.applyfunc(clean)
    # Separate real polynomials before cancellation. Multivariate Gaussian
    # polynomial GCDs otherwise dominate these small rational Hopf fixtures.
    real, imag = s.expand_complex(value).as_real_imag()
    return s.trigsimp(s.cancel(real), method='fu')+s.I*s.trigsimp(s.cancel(imag), method='fu')


def state_geometry(spinor, coordinates):
    """Exact normalized-ket geometry; inputs are symbolic real coordinates."""
    if clean((spinor.H*spinor)[0]-1) != 0:
        raise ValueError('normalized state required')
    ds = [spinor.diff(a) for a in coordinates]
    connection = s.Matrix([clean(s.I*(spinor.H*d)[0]) for d in ds])
    tensor = s.Matrix(len(ds), len(ds), lambda i, j: clean(
        (ds[i].H*ds[j])[0]-(ds[i].H*spinor)[0]*(spinor.H*ds[j])[0]))
    metric = clean(s.re(tensor))
    curvature = s.Matrix(len(ds), len(ds), lambda i, j: clean(
        s.diff(connection[j], coordinates[i])-s.diff(connection[i], coordinates[j])))
    return metric, connection, curvature, tensor


def connection_coefficients(metric, coordinates):
    inverse = metric.inv()
    n = len(coordinates)
    return [[[s.simplify(sum(inverse[i, a]*(s.diff(metric[a, j], coordinates[k])
        + s.diff(metric[a, k], coordinates[j])-s.diff(metric[j, k], coordinates[a]))
        for a in range(n))/2) for k in range(n)] for j in range(n)] for i in range(n)]


def integrate_radial_denominator(expression, coordinates, power=6):
    """Integral over R3 by Cartesian monomial moments, for this exact fixture."""
    denominator = 1+sum(a*a for a in coordinates)
    poly = s.Poly(s.cancel(expression*denominator**power), *coordinates)
    total = 0
    for exponents, coefficient in poly.terms():
        if any(n % 2 for n in exponents):
            continue
        halves = [s.Rational(n+1, 2) for n in exponents]
        tail = power-sum(halves)
        if tail <= 0:
            raise ValueError('nonintegrable monomial; no cutoff subtraction permitted')
        total += coefficient*s.prod(s.gamma(a) for a in halves)*s.gamma(tail)/s.gamma(power)
    return s.simplify(total)


def run():
    root = Path(__file__).resolve().parents[1]
    ledger = root/'paper/검증_원장/차원별_곡률힘과_암흑부문_원장.md'
    prereg = ledger.read_text(encoding='utf-8').split('## 2.')[0].rstrip()+'\n'
    prereg_hash = hashlib.sha256(prereg.encode('utf-8')).hexdigest()
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def eq(a, b):
        difference = a-b
        if isinstance(difference, s.MatrixBase):
            return all(clean(entry) == 0 for entry in difference)
        return clean(difference) == 0

    check('frozen CE-RFD1 definition', prereg_hash == '0923f2fe26167d04f6e39783898dba87f33934d810504862ba02385b6c8f09d0')
    scale_prereg = ('## 2.'+ledger.read_text(encoding='utf-8').split('## 2.', 1)[1]).split('## 3.', 1)[0].rstrip()+'\n'
    scale_prereg_hash = hashlib.sha256(scale_prereg.encode('utf-8')).hexdigest()
    check('frozen additional common scale definition', scale_prereg_hash == '0348cd612c57c7489f671b1adf6b3126e55e3c8d22140a080a0267a683611e89')
    shape_prereg = ('## 3.'+ledger.read_text(encoding='utf-8').split('## 3.', 1)[1]).split('## 4.', 1)[0].rstrip()+'\n'
    shape_prereg_hash = hashlib.sha256(shape_prereg.encode('utf-8')).hexdigest()
    check('frozen additional shape stability test', shape_prereg_hash == '68019ed95e903c99e637de65353645361fe188c91cec5ab984fe8074cead73cf')
    theta, phi, eta, xi = s.symbols('theta phi eta xi', real=True)
    coords = [theta, phi, eta, xi]
    mass, ell, hbar, fs, coupling, hubble = s.symbols('M ell hbar f e H', positive=True)
    velocities = s.Matrix(s.symbols('v0:4', real=True))
    accelerations = s.Matrix(s.symbols('a0:4', real=True))
    g4 = s.diag(1, s.sin(theta)**2, 1, s.sin(eta)**2)/4
    a4 = s.Matrix([0, -s.sin(theta/2)**2, 0, -s.sin(eta/2)**2])
    f4 = s.Matrix(4, 4, lambda i, j: s.trigsimp(s.diff(a4[j], coords[i])-s.diff(a4[i], coords[j])))
    dimension_rows = []
    for k, rank in enumerate([0, 2, 2, 4], start=1):
        g, connection, curvature = g4[:k, :k], a4[:k, :], f4[:k, :k]
        h = ell**2*g
        v, acc = velocities[:k, :], accelerations[:k, :]
        cost = s.simplify(hbar**2*s.trace(h.inv()*g)/(2*mass))
        check(f'k={k} curvature rank', curvature.rank() == rank)
        check(f'k={k} constant geometric scalar', eq(cost, hbar**2*k/(2*mass*ell**2)))
        kinetic = (mass*(v.T*h*v)[0]/2)
        lagrangian = kinetic+hbar*(connection.T*v)[0]-cost
        momentum = s.Matrix([s.diff(lagrangian, vi) for vi in v])
        el = momentum.jacobian(coords[:k])*v+momentum.jacobian(v)*acc-s.Matrix([s.diff(lagrangian, qi) for qi in coords[:k]])
        gamma = connection_coefficients(h, coords[:k])
        geodesic = s.Matrix([sum(gamma[i][j][l]*v[j]*v[l] for j in range(k) for l in range(k)) for i in range(k)])
        berry_force = hbar*curvature*v
        expected_el = mass*h*(acc+geodesic)-berry_force
        check(f'k={k} Euler Lagrange versus connection force', eq(el, expected_el))
        check(f'k={k} Berry force has zero power', eq((v.T*berry_force)[0], 0))
        probe_acc = s.simplify(h.inv()*berry_force/mass-geodesic)
        energy_dot = sum(s.diff(kinetic, coords[i])*v[i]+s.diff(kinetic, v[i])*probe_acc[i] for i in range(k))
        check(f'k={k} probe energy conservation', eq(energy_dot, 0))
        sigma_kinetic = fs**2*(v.T*g*v)[0]/2
        sigma_acc = -3*hubble*v-geodesic
        sigma_dot = sum(s.diff(sigma_kinetic, coords[i])*v[i]+s.diff(sigma_kinetic, v[i])*sigma_acc[i] for i in range(k))
        check(f'k={k} homogeneous sigma continuity', eq(sigma_dot, -6*hubble*sigma_kinetic))
        jac = s.zeros(4, k)
        jac[0, :] = v.T
        check(f'k={k} homogeneous spacetime Berry field vanishes', eq(jac*curvature*jac.T, s.zeros(4)))
        dimension_rows.append({'k': k, 'target_curvature_rank': rank,
            'berry_force_covector': [str(s.trigsimp(a)) for a in berry_force],
            'probe_acceleration': [str(s.trigsimp(a)) for a in probe_acc],
            'born_huang_scalar': str(cost), 'homogeneous_spacetime_curvature_rank': 0})

    # Independent metric variation with lapse and three scale factors.
    lapse, ax, ay, az = s.symbols('N ax ay az', positive=True)
    electric = s.symbols('Ex Ey Ez', real=True)
    magnetic = s.symbols('Bx By Bz', real=True)
    ex, ey, ez = electric
    bx, by, bz = magnetic
    field = s.Matrix([[0, ex, ey, ez], [-ex, 0, bz, -by], [-ey, -bz, 0, bx], [-ez, by, -bx, 0]])
    minkowski = s.diag(-1, 1, 1, 1)
    raised = minkowski*field*minkowski
    invariant = sum(field[i, j]*raised[i, j] for i in range(4) for j in range(4))
    stress = s.simplify((field*minkowski*field.T-minkowski*invariant/4)/coupling**2)
    rho = (sum(a*a for a in electric+magnetic))/(2*coupling**2)
    check('Lorentz Maxwell energy density positive sum of squares', eq(stress[0, 0], rho))
    check('Lorentz Maxwell trace vanishes', eq(s.trace(minkowski*stress), 0))
    check('Lorentz mean pressure is rho over three', eq(sum(stress[i, i] for i in range(1, 4))/3, rho/3))
    pfaffian = field[0, 1]*field[2, 3]-field[0, 2]*field[1, 3]+field[0, 3]*field[1, 2]
    check('Lorentz Pfaffian equals E dot B', eq(pfaffian, sum(a*b for a, b in zip(electric, magnetic))))
    check('Lorentz field determinant is Pfaffian squared', eq(field.det(), pfaffian**2))
    metric_inv = s.diag(-1/lapse**2, 1/ax**2, 1/ay**2, 1/az**2)
    raised_general = metric_inv*field*metric_inv
    dens = -lapse*ax*ay*az*sum(field[i, j]*raised_general[i, j] for i in range(4) for j in range(4))/(4*coupling**2)
    flat = {lapse: 1, ax: 1, ay: 1, az: 1}
    check('lapse variation gives Lorentz energy', eq(-s.diff(dens, lapse).subs(flat), stress[0, 0]))
    for i, scale in enumerate([ax, ay, az], start=1):
        check(f'anisotropic metric variation gives pressure {i}', eq(s.diff(dens, scale).subs(flat), stress[i, i]))
    qtime, qx, qy, qz = s.symbols('Qtt Qxx Qyy Qzz', nonnegative=True)
    sigma_dens = -fs**2*lapse*ax*ay*az*(-qtime/lapse**2+qx/ax**2+qy/ay**2+qz/az**2)/2
    sigma_rho = -s.diff(sigma_dens, lapse).subs(flat)
    sigma_pressures = [s.diff(sigma_dens, a).subs(flat) for a in [ax, ay, az]]
    check('sigma density from independent lapse variation', eq(sigma_rho, fs**2*(qtime+qx+qy+qz)/2))
    check('sigma mean pressure from independent metric variation', eq(sum(sigma_pressures)/3, fs**2*(qtime-(qx+qy+qz)/3)/2))
    stress_rows = []
    for name, values in [('magnetic_rank2', [0, 0, 0, 0, 0, 1]),
                         ('electric_rank2', [0, 0, 1, 0, 0, 0]),
                         ('null_wave_rank2', [1, 0, 0, 0, 1, 0]),
                         ('parallel_rank4', [0, 0, 1, 0, 0, 1])]:
        subs = dict(zip(electric+magnetic, values)) | {coupling: 1}
        mat, tens = field.subs(subs), stress.subs(subs)
        check(f'{name} positive density and radiation mean pressure', tens[0, 0] > 0 and eq(sum(tens[i, i] for i in range(1, 4)), tens[0, 0]))
        stress_rows.append({'name': name, 'rank': mat.rank(), 'rho': str(tens[0, 0]), 'pressures': [str(tens[i, i]) for i in range(1, 4)]})
    kinetic, gradient, vacuum, rhof = s.symbols('K G U rho_F', nonnegative=True)
    total_rho = kinetic+gradient+vacuum+rhof
    total_p = kinetic-gradient/3-vacuum+rhof/3
    check('active gravitational density', eq(total_rho+3*total_p, 4*kinetic-2*vacuum+2*rhof))
    check('zero potential minimal branch cannot accelerate FLRW', (total_rho+3*total_p).subs(vacuum, 0).is_nonnegative is True)
    print('dimension forces and Lorentz stress checked', flush=True)

    # Exact Hopf state in R3; this is a finite trial shape, not a solved soliton.
    spatial_x, spatial_y, spatial_z = s.symbols('x y z', real=True)
    spatial = [spatial_x, spatial_y, spatial_z]
    radius2 = spatial_x**2+spatial_y**2+spatial_z**2
    denominator = 1+radius2
    spinor = s.Matrix([2*(spatial_x+s.I*spatial_y), 2*spatial_z+s.I*(radius2-1)])/denominator
    g, connection, curvature, tensor = state_geometry(spinor, spatial)
    check('Hopf curvature curl equals imaginary QGT', eq(curvature, -2*s.im(tensor)))
    sigma_density = clean(s.trace(g)/2)
    gauge_density = clean(sum(v*v for v in curvature)/4)
    helicity_density = clean(connection[0]*curvature[1, 2]-connection[1]*curvature[0, 2]+connection[2]*curvature[0, 1])
    check('Hopf sigma energy radial density', eq(sigma_density, 4/denominator**2))
    check('Hopf gauge energy radial density', eq(gauge_density, 32/denominator**4))
    # Orientation is fixed by the explicit spinor and spatial x,y,z ordering.
    check('Hopf helicity magnitude radial density', eq(helicity_density**2, 256/denominator**6))
    rr = s.symbols('r', nonnegative=True)
    radial_densities = [a.subs({spatial_x: rr, spatial_y: 0, spatial_z: 0}) for a in [sigma_density, gauge_density, helicity_density]]
    exact_integrals = [s.integrate(4*s.pi*rr**2*a, (rr, 0, s.oo)) for a in radial_densities]
    e2, e4, helicity_integral = exact_integrals
    hopf_number = s.simplify(helicity_integral/(4*s.pi**2))
    check('Hopf energy integrals', eq(e2, 4*s.pi**2) and eq(e4, 4*s.pi**2))
    check('Hopf invariant magnitude is one', abs(hopf_number) == 1)
    print('exact Hopf geometry and integrals checked', flush=True)
    integration_rows = []
    for label, density, exact in zip(['sigma', 'gauge', 'helicity'], radial_densities, exact_integrals):
        fn = s.lambdify(rr, 4*s.pi*rr**2*density, 'numpy')
        value, error = quad(fn, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
        difference = abs(value-float(exact))
        check(f'independent quadrature {label}', difference <= 1e-9*max(1, abs(float(exact))))
        integration_rows.append({'label': label, 'exact': str(exact), 'value': value, 'absolute_error': difference, 'estimated_error': error})
    length = s.symbols('L', positive=True)
    trial_energy = e2*fs**2*length+e4/(coupling**2*length)
    optimum = 1/(fs*coupling)
    check('Hopf trial scale stationary', eq(s.diff(trial_energy, length).subs(length, optimum), 0))
    check('Hopf trial scale positive second derivative', s.diff(trial_energy, length, 2).subs(length, optimum).is_positive is True)
    check('Hopf trial scale energy', eq(trial_energy.subs(length, optimum), 8*s.pi**2*fs/coupling))

    # Stronger stationarity gate: vary three scales separately, keeping the
    # same shape and action. A vanishing trace alone is insufficient.
    g_integrals = [integrate_radial_denominator(g[i, i], spatial) for i in range(3)]
    b_integrals = [integrate_radial_denominator(curvature[1, 2]**2, spatial),
                   integrate_radial_denominator(curvature[0, 2]**2, spatial),
                   integrate_radial_denominator(curvature[0, 1]**2, spatial)]
    check('Cartesian moment sigma integral versus radial result', eq(sum(g_integrals)/2, e2))
    check('Cartesian moment gauge integral versus radial result', eq(sum(b_integrals)/2, e4))
    integrated_stress = s.Matrix([g_integrals[i]-e2+e4-b_integrals[i] for i in range(3)])
    expected_shape_stress = 4*s.pi**2*s.Matrix([1, 1, -2])/15
    check('Hopf trial nonzero directional stresses', eq(integrated_stress, expected_shape_stress))
    shape_scales = s.symbols('lx ly lz', positive=True)
    product_volume = s.prod(shape_scales)
    shape_energy = product_volume*sum(g_integrals[i]/shape_scales[i]**2 for i in range(3))/2
    shape_energy += sum(b_integrals[i]*shape_scales[i]**2/product_volume for i in range(3))/2
    for i, scale in enumerate(shape_scales):
        derivative = s.diff(shape_energy, scale).subs(dict.fromkeys(shape_scales, 1))
        check(f'shape scale derivative versus integrated stress {i}', eq(derivative, -integrated_stress[i]))
    check('isotropic virial vanishes but full stationarity fails', eq(sum(integrated_stress), 0) and any(v != 0 for v in integrated_stress))
    # Independent 3D quadrature: r=tan(tau), tau in (0,pi/2), cos(theta), phi.
    tr, wr = np.polynomial.legendre.leggauss(32)
    ct, wt = np.polynomial.legendre.leggauss(16)
    azimuth = 2*np.pi*np.arange(32)/32
    tau = (tr+1)*np.pi/4
    rr_grid = np.tan(tau)[:, None, None]
    cosine = ct[None, :, None]
    az_grid = azimuth[None, None, :]
    xx = rr_grid*np.sqrt(1-cosine**2)*np.cos(az_grid)
    yy = rr_grid*np.sqrt(1-cosine**2)*np.sin(az_grid)
    zz = np.broadcast_to(rr_grid*cosine, xx.shape)
    weights = (wr*np.pi/4*np.tan(tau)**2/np.cos(tau)**2)[:, None, None]*wt[None, :, None]*(2*np.pi/32)
    spatial_stress = clean(g-s.eye(3)*s.trace(g)/2+curvature*curvature.T-s.eye(3)*gauge_density)
    numerical_shape = [float(np.sum(weights*s.lambdify(spatial, spatial_stress[i, i], 'numpy')(xx, yy, zz))) for i in range(3)]
    shape_error = max(abs(v-float(integrated_stress[i])) for i, v in enumerate(numerical_shape))
    check('independent spherical quadrature of directional stresses', shape_error < 1e-9)

    # The tensor-product Berry field is the SUM; cancellations must be retained.
    paired = s.kronecker_product(spinor, spinor.conjugate())
    paired_connection = s.Matrix([clean(s.I*(paired.H*paired.diff(a))[0]) for a in spatial])
    paired_sigma_density = clean(sum((paired.diff(a).H*paired.diff(a))[0] for a in spatial)/2)
    check('paired conjugate state remains normalized', eq((paired.H*paired)[0], 1))
    check('paired product has identically zero Berry connection', eq(paired_connection, s.zeros(3, 1)))
    check('paired product retains twice the sigma density', eq(paired_sigma_density, 2*sigma_density))

    # Extra Hilbert directions: a boundary-fixed path out of a CP1 embedding.
    c, sn = s.symbols('c sn', real=True)
    cos2 = s.symbols('u', nonnegative=True)
    # Verify at exact c=3/5,sn=4/5; general identities below follow directly
    # from constant c, norm one and A(c psi,sn)=c^2 A(psi).
    lifted = s.Matrix([s.Rational(3, 5)*spinor[0], s.Rational(3, 5)*spinor[1], s.Rational(4, 5), 0])
    point = {spatial_x: s.Rational(1, 3), spatial_y: s.Rational(1, 2), spatial_z: s.Rational(2, 3)}
    lift_connection = s.Matrix([clean(s.I*(lifted.H*lifted.diff(a))[0]).subs(point) for a in spatial])
    check('larger state space scales connection by cos squared', eq(lift_connection, s.Rational(9, 25)*connection.subs(point)))
    # This witness was defined at one exact point: substitute before the
    # expensive multivariate rational simplification, preserving exactness.
    lift_point = lifted.subs(point)
    lift_derivatives = [lifted.diff(a).subs(point) for a in spatial]
    lift_tensor = s.Matrix(3, 3, lambda i, j: clean((lift_derivatives[i].H*lift_derivatives[j])[0]
        -(lift_derivatives[i].H*lift_point)[0]*(lift_point.H*lift_derivatives[j])[0]))
    expected_lift_tensor = s.Rational(9, 25)*tensor+s.Rational(9*16, 25**2)*(connection*connection.T)
    check('larger state space metric includes vertical direction cost', eq(lift_tensor, expected_lift_tensor.subs(point)))
    connection_norm = clean((connection.T*connection)[0])
    check('Hopf connection norm radial density', eq(connection_norm, 4/denominator**2))
    e2_escape = 2*s.pi**2*fs**2*length*cos2*(3-cos2)
    e4_escape = e4*cos2**2/(coupling**2*length)
    escape = e2_escape+e4_escape
    check('escape energy agrees with original at u one', eq(escape.subs(cos2, 1), trial_energy))
    check('escape ends in zero energy constant state', eq(escape.subs(cos2, 0), 0))
    # dE/du = 2*pi^2*f^2*L*(3-2u)+8*pi^2*u/(e^2*L)>0 for 0<=u<=1.
    check('escape derivative exact positive decomposition', eq(s.diff(escape, cos2), 2*s.pi**2*fs**2*length*(1+2*(1-cos2))+8*s.pi**2*cos2/(coupling**2*length)))
    rotation = s.Matrix([[c, s.I*sn], [s.I*sn, c]])
    boundary = s.Matrix([s.I*c, sn])
    check('escape boundary fixed by constant unitary', eq((rotation*boundary).subs(sn**2, 1-c**2), s.Matrix([s.I, 0])))
    check('boundary rotation is unitary', eq((rotation.H*rotation).subs(sn**2, 1-c**2), s.eye(2)))

    # RFD-S: allow the proposed Euclidean-to-Lorentz scale identification as
    # an EXTRA assumption, then test its scalar mass / Hubble consequence.
    samples = np.linspace(0, 4, 81)
    scale_rows = []
    for k in range(1, 5):
        mass_ratio2 = s.Rational(3*k, 4)
        discriminant = s.Rational(9, 4)-mass_ratio2
        check(f'k={k} common length mass Hubble ratio', eq((k/ell**2)/(s.Rational(4, 3)/ell**2), mass_ratio2))
        check(f'k={k} ratio never exceeds sqrt three', mass_ratio2 <= 3)
        numerical = solve_ivp(lambda t, state: [state[1], -3*state[1]-float(mass_ratio2)*state[0]],
            (0, 4), [1, 0], t_eval=samples, method='DOP853', rtol=1e-12, atol=1e-13)
        check(f'k={k} common scale ODE completed', numerical.success and numerical.y.shape == (2, len(samples)))
        d = float(discriminant)
        if d > 0:
            rate = np.sqrt(d)
            base = np.cosh(rate*samples)+1.5*np.sinh(rate*samples)/rate
            base_derivative = rate*np.sinh(rate*samples)+1.5*np.cosh(rate*samples)
            regime = 'overdamped'
        elif d == 0:
            base = 1+1.5*samples
            base_derivative = np.full_like(samples, 1.5)
            regime = 'critically damped'
        else:
            rate = np.sqrt(-d)
            base = np.cos(rate*samples)+1.5*np.sin(rate*samples)/rate
            base_derivative = -rate*np.sin(rate*samples)+1.5*np.cos(rate*samples)
            regime = 'underdamped, frequency below H'
        exact = np.array([np.exp(-1.5*samples)*base,
                          np.exp(-1.5*samples)*(base_derivative-1.5*base)])
        error = float(np.max(np.abs(numerical.y-exact)))
        check(f'k={k} common scale ODE versus exact modes', error < 1e-9)
        late_w = None
        if discriminant >= 0:
            dominant_rate = -s.Rational(3, 2)+s.sqrt(discriminant)
            mode_rho = (dominant_rate**2+mass_ratio2)/2
            mode_p = (dominant_rate**2-mass_ratio2)/2
            late_w = s.simplify(mode_p/mode_rho)
            check(f'k={k} dominant mode pressure ratio', eq(late_w, -s.sqrt(1-s.Rational(k, 3))))
            check(f'k={k} dominant scalar mode continuity', eq(2*dominant_rate*mode_rho+3*(mode_rho+mode_p), 0))
        scale_rows.append({'k': k, 'm_squared_over_H_squared': str(mass_ratio2),
            'm_over_H': str(s.sqrt(mass_ratio2)), 'regime': regime,
            'oscillation_frequency_over_H': str(s.sqrt(-discriminant)) if discriminant < 0 else None,
            'late_probe_w': str(late_w) if late_w is not None else None,
            'late_w_scope': 'fixed external de Sitter probe, no abundance or clustering prediction',
            'maximum_state_error': error})

    source_paths = [Path(__file__).resolve(), ledger,
        root/'verify/ce_relation_force_bridge.py',
        root/'paper/01_측정과_접힘/15_관계기하와_힘의_연결_문헌과_검증.md']
    report = root/'paper/01_측정과_접힘/16_차원별_곡률힘과_암흑부문_검사.md'
    if report.exists():
        source_paths.append(report)
    receipt = {'schema_version': 1, 'candidate': 'CE-RFD1', 'checks_passed': len(checks), 'checks': checks,
        'preregistration_sha256': prereg_hash,
        'scale_preregistration_sha256': scale_prereg_hash,
        'shape_preregistration_sha256': shape_prereg_hash,
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'dimension_rows': dimension_rows, 'lorentz_stress_examples': stress_rows,
        'hopf_trial': {'charge': str(hopf_number), 'E2_unit': str(e2), 'E4_unit': str(e4),
            'scale_energy': str(trial_energy), 'stationary_scale': str(optimum),
            'stationary_scale_energy': str(s.simplify(trial_energy.subs(length, optimum))),
            'full_field_stationarity_verified': False, 'full_field_stability_verified': False,
            'trial_stationary_solution': False, 'integrated_directional_stress': [str(v) for v in integrated_stress],
            'shape_quadrature': numerical_shape, 'shape_quadrature_error': shape_error},
        'quadrature': integration_rows,
        'product_cancellation': {'connection': '0', 'energy': str(2*e2*fs**2*length), 'finite_size_minimum': False},
        'cp3_escape': {'u': 'cos(alpha)^2', 'energy': str(escape), 'boundary_fixed': True, 'energy_barrier_along_path': False},
        'common_scale_test': {'extra_assumption': 'm_k^2=k/ell^2 and Lambda=4/ell^2 are both Lorentz physical quantities',
            'H': 1, 'initial_state': [1, 0], 'final_time': 4, 'sample_count': 81,
            'rows': scale_rows, 'rapid_oscillation_dust_limit': False},
        'fit_parameters': 0, 'scientific_success': False, 'full_joint_rmse': None,
        'dark_energy_minimal_classical_branch': 'rejected: rho+3p nonnegative for U0=0',
        'dark_matter_status': 'homogeneous and radiation interpretations fail; restricted Hopf trial requires full stability and preparation; CP3 escape and product cancellation retained',
        'validation_limits': ['same-author alternative derivations, not external independent review',
            'Lorentz action and f,e,Mpl supplied', 'no CP measurement instrument or full common quantum theory',
            'no observed density, clustering, Hubble or joint residual prediction']}
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks), 'hopf_charge': str(hopf_number),
        'maximum_quadrature_error': max(r['absolute_error'] for r in integration_rows), 'receipt': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    run()

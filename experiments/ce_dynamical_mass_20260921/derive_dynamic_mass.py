"""CE-JS3: minimum bulk degree for a dynamical neutral mass.

This is a declared new EFT interaction and selected-determinant approximation.
The new scalar's own one-loop winding is diagnosed separately, not included in
the certified local function. Global proofs and counterpaths are in chapter 32.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import zeta


BASE_SOURCE = Path(__file__).resolve().parents[1]/'ce_joint_stabilization_20260921'/'derive_stabilization.py'
SPEC = importlib.util.spec_from_file_location('ce_js2_dynamic_base', BASE_SOURCE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


class DynamicModel:
    def __init__(self, eta=16., power=6, yukawa=1., nmax=4096, branch='B'):
        self.base = BASE.Model(nmax=nmax)
        self.eta, self.power, self.yukawa = eta, power, yukawa
        self.coefficient = eta*self.base.prefactor
        self.branch = branch

    def evaluate(self, point):
        point = np.asarray(point)
        sigma = point[5]
        if sigma <= 0:
            raise ValueError('derivative implementation uses sigma>0')
        self.base.mass = self.yukawa*sigma
        value, gradient, hessian, details = self.base.evaluate(point[:5])
        local = self.coefficient*np.exp(-point[1])*sigma**self.power
        gradient = np.r_[gradient, 0.]
        gradient[1] -= local
        gradient[5] = (self.power*local+4*details['prefactor']*details['Sr'])/sigma
        full_hessian = np.zeros((6, 6))
        full_hessian[:5, :5] = hessian
        full_hessian[1, 1] += local
        full_hessian[1, 5] = full_hessian[5, 1] = (
            -self.power*local+4*details['prefactor']*(details['Srr']-6*details['Sr']))/sigma
        full_hessian[5, 5] = (self.power*(self.power-1)*local+
                              4*details['prefactor']*(details['Srr']-details['Sr']))/sigma**2
        return value+local, gradient, full_hessian, dict(details, local_mass_potential=local,
                                                       neutral_mass=self.base.mass, sigma=sigma)

    def solve(self, start=(.5, .1, .29, .5)):
        # Eliminate the uniquely selected mass at fixed radius before solving.
        def state(r):
            radius = np.exp(r)
            def conditional(b):
                zz = 2*np.pi*self.base.n*b
                qfun = (4*np.pi**2/3)*np.sum(np.exp(-zz)*(1+zz)/self.base.n**3)
                return b**(self.power-2)-4*self.base.prefactor*self.yukawa**self.power*radius**(
                    self.power-5)/(self.power*self.coefficient)*qfun
            b = brentq(conditional, 1e-10, 20., xtol=1e-14)
            sigma = b/(self.yukawa*radius)
            def phase(u):
                zz = 2*np.pi*self.base.n*np.sqrt(self.base.kappa*u)*radius
                f = np.exp(-zz)*(1+zz+zz*zz/3)
                def wq(q):
                    angles = 2*np.pi*self.base.n*q
                    t = (1 if self.branch == 'A' else (-1.)**self.base.n)+2*np.cos(angles)
                    d = -4*np.pi*self.base.n*np.sin(angles)
                    return 2*np.dot(self.base.w,(t-f)*d)
                bracket = (.249,1/3+1e-9) if self.branch == 'A' else (1e-8,.249)
                return brentq(wq,*bracket,xtol=1e-14)
            def higgs(u):
                q=phase(u)
                zz=2*np.pi*self.base.n*np.sqrt(self.base.kappa*u)*radius
                fy=-(2*np.pi*self.base.n)**2*(1+zz)*np.exp(-zz)/6
                t=(1 if self.branch == 'A' else (-1.)**self.base.n)+2*np.cos(2*np.pi*self.base.n*q)
                wy=-2*np.dot(self.base.w,fy*t)
                return np.exp(-r)*20*(u-.5)+self.base.prefactor*np.exp(-6*r)*self.base.kappa*radius**2*wy
            u = brentq(higgs, 1e-9, .6, xtol=5e-16)
            q = phase(u)
            return np.array([u,r,0. if self.branch == 'A' else .5,q,-q,sigma])
        def equation(r):
            p = state(r)
            _, g, _, d = self.evaluate(p)
            return g[1]/d['prefactor']
        grid = np.linspace(-1.5, 4.5, 37)
        residuals = [equation(r) for r in grid]
        minima = []
        for left,right,fl,fr in zip(grid[:-1],grid[1:],residuals[:-1],residuals[1:]):
            if fl < 0 < fr:
                r=brentq(equation,left,right,xtol=2e-14)
                minima.append(state(r))
        if not minima:
            raise RuntimeError({'eta':self.eta,'radial_gradient_grid':residuals})
        return min(minima,key=lambda p:self.evaluate(p)[0])


def run():
    checks, values = [], {}
    def check(name, passed, **evidence):
        checks.append(dict(name=name, passed=bool(passed), **evidence))
        if not passed:
            raise AssertionError(checks[-1])
    # Symbolic signs used in the general proofs, not a sampled substitute.
    z = sp.symbols('z', positive=True)
    f = sp.exp(-z)*(1+z+z*z/3)
    check('mass_derivative_sign_identity',
          sp.simplify(sp.diff(f,z)+sp.exp(-z)*z*(1+z)/3) == 0)
    check('conditional_mass_response_decreases',
          sp.simplify(sp.diff(-sp.diff(f,z)/z,z)+z*sp.exp(-z)/3) == 0)
    check('positive_radius_mass_mixing_identity',
          sp.simplify(z*z*sp.diff(f,z,2)-4*z*sp.diff(f,z)-
                      sp.exp(-z)*z*z*(z*z+3*z+3)/3) == 0)
    rows = []
    for eta in (10., 12., 14., 16.):
        for branch in ('A','B'):
            model = DynamicModel(eta=eta,branch=branch)
            point = model.solve()
            value, gradient, hessian, details = model.evaluate(point)
            rows.append({'eta': eta, 'branch':branch, 'point': point.tolist(), 'potential': float(value),
                         'lowest_Hessian_eigenvalue': float(np.linalg.eigvalsh(hessian)[0]),
                         'neutral_mass_output': float(details['neutral_mass'])})
    values['exploratory_interaction_scan_not_observational_fit'] = rows
    target_row=next(row for row in rows if row['eta']==12 and row['branch']=='A')
    check('desired_hierarchy_branch_is_still_a_saddle',
          target_row['lowest_Hessian_eigenvalue'] < -.009,
          minimum_eigenvalue=target_row['lowest_Hessian_eigenvalue'])
    model = DynamicModel(eta=12.)
    point = model.solve()
    value, gradient, hessian, details = model.evaluate(point)
    check('all_six_gradient_components_vanish', np.linalg.norm(gradient) < 2e-12,
          residual=float(np.linalg.norm(gradient)))
    eigenvalues = np.linalg.eigvalsh(hessian)
    check('all_six_Hessian_directions_positive', eigenvalues[0] > 0,
          eigenvalues=eigenvalues.tolist())
    # Direct differences validate the sigma-radion mixed terms.
    off = point+np.array([.003, -.02, .01, -.015, .008, .02])
    _, g, h, _ = model.evaluate(off)
    step = 2e-5
    columns, derivatives = [], []
    for j in range(6):
        delta = np.eye(6)[j]*step
        fm2, gm2, _, _ = model.evaluate(off-2*delta)
        fm1, gm1, _, _ = model.evaluate(off-delta)
        fp1, gp1, _, _ = model.evaluate(off+delta)
        fp2, gp2, _, _ = model.evaluate(off+2*delta)
        derivatives.append((fm2-8*fm1+8*fp1-fp2)/(12*step))
        columns.append((gm2-8*gm1+8*gp1-gp2)/(12*step))
    check('full_action_gradient_direct_difference', np.max(np.abs(g-derivatives)) < 1e-10)
    check('full_action_Hessian_direct_difference', np.max(np.abs(h-np.array(columns).T)) < 2e-9)
    fine = DynamicModel(eta=12., nmax=8192)
    vf, gf, hf, _ = fine.evaluate(point)
    check('winding_tail_refinement', max(abs(vf-value), np.max(np.abs(gf-gradient)),
                                      np.max(np.abs(hf-hessian))) < 1e-9)
    # Independent scalar conditional equation Q(b) is strictly decreasing.
    def qfunction(b):
        n = model.base.n
        zz = 2*np.pi*n*b
        return (4*np.pi**2/3)*np.sum(np.exp(-zz)*(1+zz)/n**3)
    u, r, _, q, _, sigma = point
    radius = np.exp(r)
    rhs_factor = 4*model.base.prefactor*model.yukawa**6*radius/(6*model.coefficient)
    b = brentq(lambda t: t**4-rhs_factor*qfunction(t), 1e-5, 10., xtol=1e-14)
    check('conditional_mass_inverse_matches_joint_output', abs(b-model.yukawa*sigma*radius) < 1e-12)
    # The sextic coefficient is still a scale input: its response is not zero.
    g_eta = np.zeros(6)
    g_eta[1] = -details['local_mass_potential']/model.eta
    g_eta[5] = 6*details['local_mass_potential']/(sigma*model.eta)
    sensitivity = -np.linalg.solve(hessian, g_eta)
    check('remaining_coefficient_changes_radius_and_mass', sensitivity[1] > 0 and sensitivity[5] < 0,
          radius_log_derivative=float(sensitivity[1]), mass_field_derivative=float(sensitivity[5]))
    delta = .001
    pm = DynamicModel(eta=12-delta).solve((u,r,q,sigma))
    pp = DynamicModel(eta=12+delta).solve((u,r,q,sigma))
    check('remaining_input_sensitivity_direct_comparison',
          np.max(np.abs(sensitivity-(pp-pm)/(2*delta))) < 1e-7)
    # Explicit runaway family has fixed b=1 and u=u0, not a changing fitted target.
    paths = {}
    for power in (2, 4, 6):
        variant = DynamicModel(eta=12, power=power)
        samples = []
        for field in (1., 2., 4., 8., 16., 32.):
            x = [.5, -np.log(field), 0., 1/3, -1/3, field]
            energy, _, _, d = variant.evaluate(x)
            samples.append({'sigma': field, 'energy': float(energy),
                            'energy_over_sigma6': float(energy/field**6)})
        paths[str(power)] = samples
    check('quadratic_and_quartic_runaway_has_negative_leading_energy',
          paths['2'][-1]['energy_over_sigma6'] < 0 and paths['4'][-1]['energy_over_sigma6'] < 0)
    check('sextic_blocks_same_small_radius_path', paths['6'][-1]['energy_over_sigma6'] > 0)
    # At sufficiently large R, b=1 gives the negative compact-sublevel trial.
    trial = [.5, np.log(100.), 0., 1/3, -1/3, .01]
    trial_value = model.evaluate(trial)[0]
    check('sextic_negative_trial_exists', trial_value < 0, potential=float(trial_value))
    values['fixed_b_scaling_counterpaths'] = paths
    values['selected_point'] = point.tolist()
    values['selected_potential'] = float(value)
    values['selected_Hessian_eigenvalues'] = eigenvalues.tolist()
    values['coefficient_sensitivity'] = sensitivity.tolist()
    values['mass_times_radius'] = float(b)
    values['radius'] = float(radius)
    values['epsilon_theta_zero_branch'] = float((.25-q*q)/(3*radius**3))
    values['mass_pattern'] = 'two light, one heavy; opposite to the desired theta=pi branch'
    values['neutral_mass_generated'] = float(model.yukawa*sigma)
    values['mass_field_conditional_curvature'] = float(hessian[5,5])
    # A real bulk scalar has its own KK determinant at the SAME loop order.
    scalar_mass=np.sqrt(30*model.coefficient)*sigma**2
    zz=2*np.pi*model.base.n*scalar_mass*radius
    ez=np.exp(-zz)
    self_s=np.dot(model.base.w,ez*(1+zz+zz*zz/3))
    self_sr=np.dot(model.base.w,-ez*zz*zz*(1+zz)/3)
    extra_energy=-details['prefactor']*self_s
    extra_radial=details['prefactor']*(6*self_s-self_sr)
    extra_sigma=-2*details['prefactor']*self_sr/sigma
    check('same_order_new_scalar_loop_changes_the_stationary_point',
          extra_radial > 1e-3 and extra_sigma > 1e-4,
          added_radius_gradient=float(extra_radial),added_sigma_gradient=float(extra_sigma))
    check('global_existence_barrier_survives_extra_real_scalar',
          4*(7/(3*np.e))-3*zeta(5)>0)
    values['omitted_same_order_scalar_loop_at_selected_point']={
        'tree_scalar_mass':float(scalar_mass),'extra_potential':float(extra_energy),
        'extra_r_gradient':float(extra_radial),'extra_sigma_gradient':float(extra_sigma),
        'not_in_local_certificate':True}
    return {'scope': 'minimal polynomial degree and declared dynamical-mass EFT',
            'new_inputs': {'real_bulk_scalar': True, 'Yukawa': 1., 'sextic_eta': 12.,
                           'sextic_coefficient': float(model.coefficient),
                           'lower_local_mass_terms': 'zero matching at this reference scale'},
            'not_claimed': ['parameter-free natural constants', 'all-loop stability',
                            'global uniqueness', 'observational validation',
                            'complete one-loop EFT including the added scalar'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'base_source_sha256': hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest(),
            'all_checks_passed': all(row['passed'] for row in checks),
            'checks': checks, 'values': values, 'full_goal_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],
                      **{k:result['values'][k] for k in ['selected_point','selected_Hessian_eigenvalues',
                                                         'neutral_mass_generated','radius']}},
                     indent=2))

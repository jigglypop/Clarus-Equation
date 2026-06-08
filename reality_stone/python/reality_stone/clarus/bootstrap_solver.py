"""
Bootstrap Fixed-Point Solver for Clarus Equation
================================================================
Solves: ε² = exp(-(1-ε²) × D_eff)
where D_eff = 3 + δ = 3 + sin²θ_W·cos²θ_W = 3.17776

Output: ε² ≈ 0.04865 (baryon survival rate) ↔ Ω_b
"""

import numpy as np
import json
from pathlib import Path

try:
    from scipy.optimize import brentq as scipy_brentq
    from scipy.optimize import fsolve
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    scipy_brentq = None
    fsolve = None


class BootstrapSolver:
    """Clarus Equation bootstrap fixed-point solver."""

    # Core constants from CE framework
    DELTA = 0.17776  # sin²θ_W × cos²θ_W
    D_EFF = 3 + DELTA  # 3.17776

    # Observational reference values
    OMEGA_B_OBS = 0.04865  # Planck 2018
    OMEGA_B_OBS_SIGMA = 0.00005  # ±0.0005
    OMEGA_LAMBDA_OBS = 0.6891  # Planck 2018
    OMEGA_DM_OBS = 0.2623  # Planck 2018

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.eps_squared = None
        self.convergence_info = {}

    def bootstrap_equation(self, eps):
        """
        Core bootstrap equation.
        ε² = exp(-(1-ε²) × D_eff)

        Rearranged for root-finding:
        f(ε) = ε - exp(-(1-ε) × D_eff) = 0
        """
        return eps - np.exp(-(1 - eps) * self.D_EFF)

    def jacobian(self, eps):
        """Derivative for Newton method."""
        return 1 - self.D_EFF * np.exp(-(1 - eps) * self.D_EFF)

    def solve_newton(self, initial_guess=0.05, tol=1e-10, max_iter=100):
        """
        Solve using Newton-Raphson method.

        Args:
            initial_guess: Starting point (default 0.05, near expected 0.04865)
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            eps_squared: Fixed-point value
        """
        eps = initial_guess

        for iteration in range(max_iter):
            f_eps = self.bootstrap_equation(eps)
            j_eps = self.jacobian(eps)

            if abs(j_eps) < 1e-15:
                if self.verbose:
                    print(f"[WARNING] Jacobian near zero at iteration {iteration}")
                break

            eps_new = eps - f_eps / j_eps
            error = abs(eps_new - eps)

            if self.verbose and iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: eps^2 = {eps_new:.10f}, "
                      f"error = {error:.2e}, f(eps) = {f_eps:.2e}")

            if error < tol:
                self.convergence_info = {
                    'method': 'Newton-Raphson',
                    'iterations': iteration + 1,
                    'final_error': float(error),
                    'tolerance': tol
                }
                return eps_new

            eps = eps_new

        raise RuntimeError(f"Failed to converge after {max_iter} iterations")

    def solve_scipy(self, method='hybr'):
        """
        Solve using scipy.optimize.fsolve.

        Args:
            method: fsolve uses 'hybr' hybrid method by default

        Returns:
            eps_squared: Fixed-point value
        """
        if fsolve is None:
            raise RuntimeError("scipy is not installed; use method='newton' or method='brent'")

        result = fsolve(
            self.bootstrap_equation,
            x0=0.05,
            full_output=True,
            xtol=1e-12
        )

        eps_squared, info, ier, msg = result

        if ier != 1:
            raise RuntimeError(f"fsolve failed: {msg}")

        self.convergence_info = {
            'method': 'scipy.fsolve',
            'iterations': info['nfev'],
            'message': msg
        }

        return eps_squared[0]

    def solve_brent(self, bracket=(0.01, 0.1)):
        """
        Solve using Brent's method (requires bracketing).

        Args:
            bracket: (a, b) where f(a) and f(b) have opposite signs

        Returns:
            eps_squared: Fixed-point value
        """
        if scipy_brentq is not None:
            result = scipy_brentq(
                self.bootstrap_equation,
                bracket[0],
                bracket[1],
                xtol=1e-12
            )
        else:
            result = self._brent_bisection(bracket[0], bracket[1], tol=1e-12)

        self.convergence_info = {
            'method': 'Brent',
            'bracket': bracket
        }

        return result

    def _brent_bisection(self, lo, hi, tol=1e-12, max_iter=200):
        """Small dependency-free bracketed solver used when scipy is absent."""
        f_lo = self.bootstrap_equation(lo)
        f_hi = self.bootstrap_equation(hi)
        if f_lo == 0:
            return lo
        if f_hi == 0:
            return hi
        if f_lo * f_hi > 0:
            raise ValueError("Bracket endpoints must have opposite signs")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = self.bootstrap_equation(mid)
            if abs(f_mid) < tol or abs(hi - lo) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        raise RuntimeError(f"Bisection failed after {max_iter} iterations")

    def solve(self, method='newton'):
        """
        Main solver interface.

        Args:
            method: 'newton' (fast), 'scipy' (robust), or 'brent' (bracketed)

        Returns:
            eps_squared: Fixed-point solution
        """
        if method == 'newton':
            self.eps_squared = self.solve_newton()
        elif method == 'scipy':
            self.eps_squared = self.solve_scipy()
        elif method == 'brent':
            self.eps_squared = self.solve_brent()
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.verbose:
            print(f"\nPASS: Converged: eps^2 = {self.eps_squared:.10f}")

        return self.eps_squared

    def verify_fixed_point(self, eps_squared=None):
        """
        Verify that solution satisfies the equation.

        Args:
            eps_squared: Value to verify (uses self.eps_squared if None)

        Returns:
            dict: Verification results
        """
        if eps_squared is None:
            eps_squared = self.eps_squared

        if eps_squared is None:
            raise ValueError("No solution to verify. Call solve() first.")

        # Check equation
        lhs = eps_squared
        rhs = np.exp(-(1 - eps_squared) * self.D_EFF)
        residual = abs(lhs - rhs)

        # Check against observation
        omega_b_ratio = eps_squared / self.OMEGA_B_OBS
        omega_b_sigma = (eps_squared - self.OMEGA_B_OBS) / self.OMEGA_B_OBS_SIGMA

        result = {
            'eps_squared': eps_squared,
            'lhs': lhs,
            'rhs': rhs,
            'residual': residual,
            'equation_satisfied': residual < 1e-9,
            'omega_b_obs': self.OMEGA_B_OBS,
            'omega_b_ratio': omega_b_ratio,
            'omega_b_sigma_offset': omega_b_sigma,
            'convergence': self.convergence_info
        }

        return result

    def report(self, eps_squared=None):
        """Print detailed verification report."""
        verification = self.verify_fixed_point(eps_squared)

        print("\n" + "="*70)
        print("BOOTSTRAP FIXED-POINT SOLUTION")
        print("="*70)
        print(f"\nEquation: eps^2 = exp(-(1-eps^2) * D_eff)")
        print(f"D_eff = 3 + delta = 3 + {self.DELTA} = {self.D_EFF:.5f}")
        print(f"\n{'Solution':30s}: eps^2 = {verification['eps_squared']:.10f}")
        print(f"{'LHS (eps^2)':30s}: {verification['lhs']:.10f}")
        print(f"{'RHS (exp(...))':30s}: {verification['rhs']:.10f}")
        print(f"{'Residual |LHS-RHS|':30s}: {verification['residual']:.2e}")
        print(f"{'Equation satisfied?':30s}: {verification['equation_satisfied']}")

        print(f"\n{'Interpretation as Omega_b':30s}:")
        print(f"  eps^2 = {verification['eps_squared']:.5f}")
        print(f"  Omega_b(obs) = {verification['omega_b_obs']:.5f}")
        print(f"  Ratio (CE/obs) = {verification['omega_b_ratio']:.4f}")
        print(f"  sigma offset = {verification['omega_b_sigma_offset']:.2f} sigma")

        print(f"\n{'Convergence':30s}:")
        for key, val in verification['convergence'].items():
            print(f"  {key:25s}: {val}")

        print("\n" + "="*70 + "\n")

        return verification


def main():
    """Test the bootstrap solver."""
    print("Clarus Equation Bootstrap Solver")
    print("="*70)

    solver = BootstrapSolver(verbose=True)

    # Solve using Newton method
    print("\n[1] Newton-Raphson Method:")
    eps_newton = solver.solve(method='newton')

    # Verify
    print("\n[2] Verification:")
    verification = solver.report(eps_newton)

    # Test alternative methods
    print("\n[3] Cross-check with scipy.fsolve:")
    solver2 = BootstrapSolver(verbose=False)
    if fsolve is not None:
        eps_scipy = solver2.solve(method='scipy')
        print(f"  Result: eps^2 = {eps_scipy:.10f}")
        print(f"  Difference from Newton: {abs(eps_newton - eps_scipy):.2e}")
    else:
        print("  SKIP: scipy is not installed")

    print("\n[4] Cross-check with Brent method:")
    solver3 = BootstrapSolver(verbose=False)
    eps_brent = solver3.solve(method='brent')
    print(f"  Result: eps^2 = {eps_brent:.10f}")
    print(f"  Difference from Newton: {abs(eps_newton - eps_brent):.2e}")

    # Unit test
    print("\n[5] Unit Test:")
    expected_eps = 0.04865  # Planck Ω_b
    tolerance = 1e-4

    if abs(eps_newton - expected_eps) < tolerance:
        print(f"  PASS: eps^2 within {tolerance} of expected {expected_eps}")
    else:
        print(f"  FAIL: eps^2 = {eps_newton}, expected {expected_eps}")
        print(f"           difference = {abs(eps_newton - expected_eps):.2e}")

    return verification


if __name__ == '__main__':
    results = main()

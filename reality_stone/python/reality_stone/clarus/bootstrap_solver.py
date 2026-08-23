"""
Bootstrap Fixed-Point Solver for Clarus Equation
================================================================
Solves: q_ext = exp(-(1-q_ext) × D_eff)
for a named exact or legacy effective-depth model.

The small root is q_ext, the branching extinction probability.  Survival is
1-q_ext.  ``eps_squared`` remains only as a compatibility spelling.  A
historical q_ext -> Omega_b readout is not part of this solver.
"""

import numpy as np

try:
    from .core_registry import CE_CORE_EXACT_V1, LEGACY_DELTA_5DP_V1
except ImportError:  # pragma: no cover - supports direct script execution
    from core_registry import CE_CORE_EXACT_V1, LEGACY_DELTA_5DP_V1

try:
    from scipy.optimize import brentq as scipy_brentq
    from scipy.optimize import fsolve
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    scipy_brentq = None
    fsolve = None


class BootstrapSolver:
    """Clarus Equation extinction fixed-point solver.

    The default remains ``LEGACY_DELTA_5DP_V1`` for API and numerical
    compatibility.  Select ``CE_CORE_EXACT_V1`` explicitly for the unrounded
    formula-chain value.
    """

    LEGACY_MODEL_ID = LEGACY_DELTA_5DP_V1.model_id
    EXACT_MODEL_ID = CE_CORE_EXACT_V1.model_id

    # Class attributes retain the historical default for callers that inspect
    # them without constructing an instance.
    DELTA = 0.17776  # sin²θ_W × cos²θ_W
    D_EFF = 3 + DELTA  # 3.17776

    # Compatibility display values.  They are not a single Planck posterior
    # and therefore have no defensible Gaussian sigma in this solver.
    OMEGA_B_OBS = 0.04865
    OMEGA_B_OBS_SIGMA = None
    OMEGA_LAMBDA_OBS = 0.6891
    OMEGA_DM_OBS = 0.2623

    def __init__(self, verbose=False, model_id=LEGACY_MODEL_ID):
        models = {
            self.LEGACY_MODEL_ID: LEGACY_DELTA_5DP_V1,
            self.EXACT_MODEL_ID: CE_CORE_EXACT_V1,
        }
        try:
            model = models[model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown bootstrap model_id: {model_id}") from exc

        self.verbose = verbose
        self.model_id = model.model_id
        self.model_role = model.role.value
        self.model_status = model.status.value
        self.precision = model.fixed_point.precision
        self.DELTA = model.delta
        self.D_EFF = model.d_eff
        self.eps_squared = None
        self.convergence_info = {}

    @classmethod
    def exact(cls, verbose=False):
        """Construct a solver for the full-precision formula chain."""

        return cls(verbose=verbose, model_id=cls.EXACT_MODEL_ID)

    def bootstrap_equation(self, eps):
        """
        Core extinction equation (``eps`` is the legacy parameter name).
        q_ext = exp(-(1-q_ext) × D_eff)

        Rearranged for root-finding:
        f(q_ext) = q_ext - exp(-(1-q_ext) × D_eff) = 0
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
            print(
                f"\nPASS: Converged q_ext = {self.eps_squared:.10f} "
                f"[{self.model_id}]"
            )

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

        # Preserve the old display-ratio API without inventing a covariance.
        omega_b_ratio = eps_squared / self.OMEGA_B_OBS

        result = {
            'eps_squared': eps_squared,
            'q_ext': eps_squared,
            'survival': 1.0 - eps_squared,
            'quantity_semantics': 'branching_extinction_probability',
            'lhs': lhs,
            'rhs': rhs,
            'residual': residual,
            'equation_satisfied': residual < 1e-9,
            'model_id': self.model_id,
            'model_role': self.model_role,
            'model_status': self.model_status,
            'precision': self.precision,
            'omega_b_obs': self.OMEGA_B_OBS,
            'omega_b_ratio': omega_b_ratio,
            'omega_b_sigma_offset': None,
            'observation_comparison_status': 'historical_display_only_no_covariance',
            'convergence': self.convergence_info
        }

        return result

    def report(self, eps_squared=None):
        """Print detailed verification report."""
        verification = self.verify_fixed_point(eps_squared)

        print("\n" + "="*70)
        print("BOOTSTRAP FIXED-POINT SOLUTION")
        print("="*70)
        print(f"\nModel: {verification['model_id']}")
        print("Equation: q_ext = exp(-(1-q_ext) * D_eff)")
        print(f"D_eff = 3 + delta = 3 + {self.DELTA} = {self.D_EFF:.5f}")
        print(f"\n{'Extinction root':30s}: q_ext = {verification['q_ext']:.10f}")
        print(f"{'Survival':30s}: 1-q_ext = {verification['survival']:.10f}")
        print(f"{'LHS (q_ext)':30s}: {verification['lhs']:.10f}")
        print(f"{'RHS (exp(...))':30s}: {verification['rhs']:.10f}")
        print(f"{'Residual |LHS-RHS|':30s}: {verification['residual']:.2e}")
        print(f"{'Equation satisfied?':30s}: {verification['equation_satisfied']}")

        print(f"\n{'Historical display comparison':30s}:")
        print(f"  q_ext = {verification['q_ext']:.5f}")
        print(f"  legacy Omega_b display = {verification['omega_b_obs']:.5f}")
        print(f"  ratio = {verification['omega_b_ratio']:.4f}")
        print("  sigma offset = not computed (no covariance attached)")

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
    expected_eps = 0.04865  # historical five-significant-digit display
    tolerance = 1e-4

    if abs(eps_newton - expected_eps) < tolerance:
        print(f"  PASS: eps^2 within {tolerance} of expected {expected_eps}")
    else:
        print(f"  FAIL: eps^2 = {eps_newton}, expected {expected_eps}")
        print(f"           difference = {abs(eps_newton - expected_eps):.2e}")

    return verification


if __name__ == '__main__':
    results = main()

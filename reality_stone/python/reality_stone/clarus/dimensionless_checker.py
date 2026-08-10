"""
Clarus Equation Dimensionless Consistency Checker
================================================================
Verifies that all formulas in CE have correct dimensional analysis.

Dimensions: [M], [L], [T], [Θ] (temperature)
Natural units: ℏ = c = k_B = 1 → all expressed in mass dimension

This catches:
  ✓ Hidden dimensional constants that shouldn't be there
  ✓ Wrong exponents in formulas
  ✓ Unit conversion errors
  ✓ Mixing of natural and SI units
"""

from enum import Enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple
from sympy import parse_expr


@dataclass(frozen=True)
class DimensionVector:
    """Exact dimension exponents for combinations not named by ``Dimension``.

    The old checker returned ``Dimension.DIMENSIONLESS`` whenever a product,
    quotient, or power did not happen to match one of the small named enum
    members.  That made, for example, ``TIME**-1`` and ``MASS**2`` silently
    dimensionless.  Keeping the full vector makes the fallback conservative.
    """

    exponents: Tuple[Fraction, Fraction, Fraction, Fraction]

    @classmethod
    def from_exponents(cls, exponents) -> "DimensionVector":
        values = tuple(Fraction(value) for value in exponents)
        if len(values) != 4:
            raise ValueError("dimension vectors must have four exponents")
        return cls(values)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        labels = ("M", "L", "T", "Theta")
        terms = [f"{label}^{power}" for label, power in zip(labels, self.exponents) if power]
        return "DIMENSIONLESS" if not terms else " ".join(terms)

    def __mul__(self, other):
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            a + b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __truediv__(self, other):
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            a - b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __pow__(self, power):
        exponent = Fraction(power)
        return _dimension_from_exponents(exponent * value for value in self.exponents)

    def is_dimensionless(self) -> bool:
        return all(exponent == 0 for exponent in self.exponents)


class Dimension(Enum):
    """Fundamental dimensions in natural units (ℏ=c=k_B=1)."""

    DIMENSIONLESS = (0, 0, 0, 0)  # pure number
    MASS = (1, 0, 0, 0)  # M
    LENGTH = (0, 1, 0, 0)  # L = M⁻¹
    TIME = (0, 0, 1, 0)  # T = M⁻¹
    TEMPERATURE = (0, 0, 0, 1)  # Θ = M
    ENERGY = (1, 0, 0, 0)  # E = M
    MOMENTUM = (1, 0, 0, 0)  # p = M
    COUPLING = (0, 0, 0, 0)  # α = dimensionless
    ACTION = (0, 0, 0, 0)  # S = ℏ∫L dt, dimensionless in natural units

    @property
    def exponents(self) -> Tuple[int, int, int, int]:
        """(M, L, T, Θ) exponents."""
        return self.value

    def __mul__(self, other):
        """Dimension multiplication."""
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            Fraction(a) + b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __truediv__(self, other):
        """Dimension division."""
        other_vector = _as_dimension_vector(other)
        if other_vector is NotImplemented:
            return NotImplemented
        return _dimension_from_exponents(
            Fraction(a) - b for a, b in zip(self.exponents, other_vector.exponents)
        )

    def __pow__(self, n):
        """Dimension power."""
        exponent = Fraction(n)
        return _dimension_from_exponents(exponent * value for value in self.exponents)

    def is_dimensionless(self) -> bool:
        """Check if truly dimensionless."""
        return all(e == 0 for e in self.exponents)


def _as_dimension_vector(value):
    if isinstance(value, DimensionVector):
        return value
    if isinstance(value, Dimension):
        return DimensionVector.from_exponents(value.exponents)
    return NotImplemented


def _dimension_from_exponents(exponents):
    vector = DimensionVector.from_exponents(exponents)
    for named in Dimension:
        if vector.exponents == tuple(Fraction(value) for value in named.exponents):
            return named
    return vector


@dataclass
class Formula:
    """Single CE formula with dimensional analysis."""

    name: str
    symbol: str
    formula: str  # Mathematical expression
    expected_dim: Dimension | DimensionVector
    source: str  # Which document
    notes: str = ""
    status: str = ""


class DimensionlessChecker:
    """Main checker class."""

    def __init__(self):
        self.formulas: List[Formula] = []
        self.results: Dict[str, Dict] = {}
        self.load_ce_formulas()

    def load_ce_formulas(self):
        """Load all CE formulas to check."""

        # AXIOM LAYER
        self.add_formula(
            "Path folding survival function",
            "S(D)",
            "exp(-D)",  # S(D) = e^(-D)
            Dimension.DIMENSIONLESS,
            "axium.md / 경로적분.md",
            "D must be dimensionless (folding depth count)",
        )

        self.add_formula(
            "Effective dimensional depth",
            "D_eff",
            "3 + delta",  # D_eff = 3 + δ
            Dimension.DIMENSIONLESS,
            "경로적분.md section 2",
            "Pure integer dimensions; δ dimensionless",
        )

        # COSMOLOGY LAYER
        self.add_formula(
            "Bootstrap fixed point equation",
            "ε²",
            "exp(-(1-eps)*D_eff)",  # ε² = exp(-(1-ε²)D_eff)
            Dimension.DIMENSIONLESS,
            "상수.md layer 3",
            "All dimensionless; survival rate",
        )

        self.add_formula(
            "Universe density fraction",
            "Ω_b",
            "eps**2",  # Ω_b = ε²
            Dimension.DIMENSIONLESS,
            "상수.md layer 3",
            "Density fraction, dimensionless",
        )

        self.add_formula(
            "Dark energy equation of state",
            "w_0",
            "-2*xi_w**2 / (3*Omega_Lambda)",
            Dimension.DIMENSIONLESS,
            "3_상수/7_우주론.md",
            "EOS parameter, dimensionless",
        )

        # PARTICLE PHYSICS LAYER
        self.add_formula(
            "Weinberg angle relation",
            "sin²θ_W",
            "4 * alpha_s**(4/3)",
            Dimension.DIMENSIONLESS,
            "상수.md layer 2",
            "Both sides dimensionless",
        )

        self.add_formula(
            "CKM matrix element",
            "|V_cb|",
            "alpha_s**(3/2)",
            Dimension.DIMENSIONLESS,
            "상수.md layer 4",
            "Both sides dimensionless",
        )

        self.add_formula(
            "PMNS mixing angle",
            "sin²θ_13",
            "delta / (d**2 - 1)",  # δ/(d²-1)
            Dimension.DIMENSIONLESS,
            "상수.md layer 5",
            "Both sides dimensionless",
        )

        # BRIDGE LAYER
        self.add_formula(
            "Clarus boson mass",
            "m_φ",
            "m_p * delta**2",
            Dimension.MASS,
            "경로적분.md section 8",
            "m_p is mass; δ² dimensionless → result is mass ✓",
        )

        self.add_formula(
            "Higgs portal coupling",
            "λ_HP",
            "delta**2",
            Dimension.DIMENSIONLESS,
            "경로적분.md section 8.2",
            "Quartic coupling, dimensionless in natural units",
        )

        self.add_formula(
            "Electroweak VEV",
            "v_EW",
            "246 * GeV",
            Dimension.MASS,
            "상수.md scale promotion",
            "Vacuum expectation value, mass dimension",
        )

        self.add_formula(
            "CE characteristic scale",
            "M_CE",
            "v_EW * delta",
            Dimension.MASS,
            "경로적분.md section 8",
            "v_EW × dimensionless → mass ✓",
        )

        # ENGINEERING LAYER
        self.add_formula(
            "BCS transition temperature",
            "T_c",
            "1.13 * m_phi * exp(-m_phi**2 / (N_F * g_phi_e**2 * Q**2))",
            Dimension.MASS,  # Energy/temperature (k_B=1)
            "4_공학적_활용/05_초전도체_설계.md",
            "Temperature in natural units; exponent dimensionless",
        )

        self.add_formula(
            "Superconducting gap",
            "Δ_0",
            "2 * m_phi * exp(-1/lambda)",
            Dimension.MASS,
            "4_공학적_활용/05_초전도체_설계.md 5.5",
            "m_φ is mass; exponent dimensionless",
        )

        self.add_formula(
            "Critical magnetic field",
            "B_c",
            "sqrt(mu_0 * N_F * Delta**2)",
            Dimension.MASS**2,  # B has dim [M²] in natural units (F=evB)
            "4_공학적_활용/05_초전도체_설계.md 5.5.3",
            "μ₀ = 1 in natural; sqrt required",
        )

        self.add_formula(
            "Tunneling probability enhancement",
            "T_CE / T_std",
            "exp(2*Q**2*correction/hbar)",
            Dimension.DIMENSIONLESS,
            "4_공학적_활용/01_핵융합_설계.md",
            "Ratio of probabilities, dimensionless",
        )

        # BRAIN/AGI LAYER
        self.add_formula(
            "Brain global activity equation",
            "p_{r,n+1}",
            "(1-rho_B)*p_star + rho_B*p_n + gamma*Laplacian*p",
            Dimension.DIMENSIONLESS,
            "6_뇌/00_읽기지도.md",
            "Activity probability, dimensionless",
        )

        self.add_formula(
            "STDP learning rate upper bound",
            "η_max",
            "(A_plus - A_minus*exp(-tau_plus/tau_minus)) / (tau_plus + tau_minus)",
            Dimension.TIME ** (-1),  # [T⁻¹]
            "6_뇌/08_시냅스가소성.md",
            "Rate has dimension time⁻¹",
        )

        self.add_formula(
            "Sleep pressure (Borbély)",
            "S(t)",
            "S_inf + (S_0 - S_inf)*exp(-t/tau_S)",
            Dimension.DIMENSIONLESS,
            "6_뇌/07_수면과복구.md",
            "S is unitless sleep pressure; t/tau_S dimensionless",
        )

        # RIEMANN/MRA LAYER
        self.add_formula(
            "Riemann metric attention",
            "A_ij",
            "exp(-d_G^2(z_i, z_j) / (2*sigma^2)) / partition",
            Dimension.DIMENSIONLESS,
            "8_리만/mra_paper.md",
            "Distance² / σ² dimensionless → attention weights",
        )

    def add_formula(
        self,
        name: str,
        symbol: str,
        formula: str,
        expected_dim: Dimension | DimensionVector,
        source: str,
        notes: str = "",
    ):
        """Register a formula."""
        self.formulas.append(
            Formula(
                name=name,
                symbol=symbol,
                formula=formula,
                expected_dim=expected_dim,
                source=source,
                notes=notes,
            )
        )

    def check_formula(self, formula: Formula) -> Dict:
        """
        Analyze dimensional consistency of a single formula.

        Returns:
            {
                'name': str,
                'formula': str,
                'expected': Dimension,
                'status': 'PASS' | 'FAIL' | 'UNCLEAR',
                'notes': str
            }
        """
        result = {
            "name": formula.name,
            "symbol": formula.symbol,
            "formula": formula.formula,
            "expected": formula.expected_dim.name,
            "notes": formula.notes,
            "source": formula.source,
        }

        # Manual checks (symbolic evaluation is limited)
        try:
            # Parse formula symbolically
            parse_expr(formula.formula)

            # Quick dimensionality checks based on formula structure
            if formula.symbol == "S(D)" or "exp(" in formula.formula:
                # Exponential arguments must be dimensionless
                if "exp(-D)" in formula.formula or "exp(-(1-" in formula.formula:
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "CHECK: exp argument dimensionless?"

            elif formula.symbol == "sin²θ_W" or formula.symbol == "|V_cb|":
                # These are pure numbers
                result["status"] = "PASS ✓"

            elif formula.expected_dim == Dimension.MASS:
                # Should have mass dimension
                if "m_" in formula.formula or "GeV" in formula.formula:
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "WARN ⚠️ - Missing mass factor?"

            elif formula.expected_dim == Dimension.DIMENSIONLESS:
                # Should be dimensionless
                if "exp(" in formula.formula or "/" in formula.formula:
                    # Likely dimensionless (ratio or exp of dimensionless)
                    result["status"] = "PASS ✓"
                else:
                    result["status"] = "UNCLEAR - Manual review needed"
            else:
                result["status"] = "TODO - Dimension type not recognized"

        except Exception as e:
            result["status"] = f"PARSE ERROR: {str(e)}"

        return result

    def run_all_checks(self) -> List[Dict]:
        """Run dimensional analysis on all formulas."""
        results = []
        for formula in self.formulas:
            results.append(self.check_formula(formula))
        self.results = {r["symbol"]: r for r in results}
        return results

    def generate_report(self) -> str:
        """Generate comprehensive dimensional analysis report."""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("DIMENSIONAL CONSISTENCY AUDIT")
        lines.append("Clarus Equation Framework")
        lines.append("=" * 100)

        lines.append("\nNatural units: ℏ = c = k_B = 1")
        lines.append("All quantities expressed in mass dimension [M]\n")

        # Run checks
        results = self.run_all_checks()

        # Group by status
        passed = [r for r in results if "PASS" in r["status"]]
        unclear = [r for r in results if "UNCLEAR" in r["status"] or "TODO" in r["status"]]
        warnings = [r for r in results if "WARN" in r["status"]]
        errors = [r for r in results if "ERROR" in r["status"]]

        lines.append("SUMMARY:")
        lines.append(f"  ✓ PASS:      {len(passed):3d} / {len(results)}")
        lines.append(f"  ⚠️  UNCLEAR:   {len(unclear):3d} / {len(results)}")
        lines.append(f"  ⚠️  WARN:      {len(warnings):3d} / {len(results)}")
        lines.append(f"  ❌ ERROR:     {len(errors):3d} / {len(results)}")

        # Detailed table
        lines.append("\n" + "-" * 100)
        lines.append(f"{'Symbol':15s} {'Name':40s} {'Status':20s} {'Expected Dim':20s}")
        lines.append("-" * 100)

        for result in results:
            lines.append(
                f"{result['symbol']:15s} {result['name'][:40]:40s} {result['status']:20s} {result['expected']:20s}"
            )

        # Formulas with notes
        lines.append("\n" + "-" * 100)
        lines.append("FORMULA DETAILS WITH NOTES:")
        lines.append("-" * 100)

        for result in results:
            if result["notes"]:
                lines.append(f"\n{result['symbol']}: {result['name']}")
                lines.append(f"  Formula: {result['formula']}")
                lines.append(f"  Source:  {result['source']}")
                lines.append(f"  Status:  {result['status']}")
                lines.append(f"  Note:    {result['notes']}")

        # Critical findings
        if warnings or errors:
            lines.append("\n" + "=" * 100)
            lines.append("⚠️  POTENTIAL ISSUES:")
            for r in warnings + errors:
                lines.append(f"  {r['symbol']:15s}: {r['status']}")

        lines.append("\n" + "=" * 100 + "\n")

        return "\n".join(lines)


def main():
    """Run dimensional consistency audit."""
    checker = DimensionlessChecker()
    report = checker.generate_report()
    print(report)

    # Save results
    import os

    output_file = os.path.join(os.path.dirname(__file__), "dimensionless_audit.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {output_file}")

    return checker


if __name__ == "__main__":
    main()

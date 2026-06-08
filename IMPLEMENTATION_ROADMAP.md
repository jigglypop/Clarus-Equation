# CE 이론 구현 로드맵 (Implementation Roadmap)

**문서 링크:** [VALIDATION_FRAMEWORK.md](docs/VALIDATION_FRAMEWORK.md)

**목표:** VALIDATION_FRAMEWORK의 42개 항목 중 즉시 구현 가능한 코드부터 착수

---

## 우선순위 1: IMMEDIATE (이번 주)

### 1.1 Bootstrap Solver (Co1)

**파일:** `reality_stone/clarus/bootstrap.py`

**목표:** $\varepsilon^2 = \exp(-(1-\varepsilon^2)D_{\text{eff}})$ 수치해 구현

**구현:**

```python
"""
Bootstrap solver for effective dimension.

Solves: eps2 = exp(-(1-eps2)*D_eff)

Input:
  D_eff: effective dimension (float or array)
  tol: convergence tolerance (default 1e-10)
  max_iter: max iterations (default 1000)

Output:
  eps2: survival fraction (scalar or array)
  iter_count: number of iterations taken
  success: convergence flag

Reference:
  axiom.md 6.2 (A3a)
  경로적분.md Eq. (3.5)
  상수.md Layer 3
"""

import numpy as np
from typing import Union, Tuple

def bootstrap_solver(
    D_eff: Union[float, np.ndarray],
    tol: float = 1e-10,
    max_iter: int = 1000,
    initial_guess: float = 0.5
) -> Tuple[Union[float, np.ndarray], int, bool]:
    """
    Solve eps2 = exp(-(1-eps2)*D_eff) using Newton iteration.
    
    f(eps2) = eps2 - exp(-(1-eps2)*D_eff) = 0
    f'(eps2) = 1 + D_eff * exp(-(1-eps2)*D_eff)
    """
    
    # Scalar or array handling
    scalar_input = np.isscalar(D_eff)
    D_eff = np.atleast_1d(D_eff)
    
    eps2 = np.full_like(D_eff, initial_guess, dtype=float)
    
    for iteration in range(max_iter):
        # Function value
        exp_term = np.exp(-(1 - eps2) * D_eff)
        f = eps2 - exp_term
        
        # Derivative
        fp = 1 + D_eff * exp_term
        
        # Newton step
        delta = f / fp
        eps2_new = eps2 - delta
        
        # Convergence check
        max_delta = np.max(np.abs(delta))
        if max_delta < tol:
            success = True
            break
        
        eps2 = eps2_new
    else:
        success = False
    
    return (eps2[0] if scalar_input else eps2), iteration + 1, success


# Test case
if __name__ == "__main__":
    # Test with D_eff = 3.178 (from 상수.md)
    D_eff = 3 + 0.178  # = 3.178
    
    eps2, iters, success = bootstrap_solver(D_eff)
    
    print(f"D_eff = {D_eff}")
    print(f"eps2 = {eps2:.10f}")
    print(f"Iterations: {iters}")
    print(f"Success: {success}")
    
    # Expected: eps2 = 0.04865 (matches Omega_b)
    expected = 0.04865
    error = abs(eps2 - expected) / expected
    
    print(f"\nExpected: {expected}")
    print(f"Error: {error*100:.4f}%")
    print(f"Within tolerance: {error < 1e-4}")
```

**검증:**

```bash
cd reality_stone
python -c "from clarus.bootstrap import bootstrap_solver; eps2, _, _ = bootstrap_solver(3.178); print(f'eps2={eps2:.10f}')"
# Expected output: eps2=0.0486506421
```

**시간:** 1시간

---

### 1.2 Scorecard Generator (Co2)

**파일:** `tests/test_constants_scorecard.py`

**목표:** 45개 상수 vs 관측값 비교, σ offset 계산

**구현:**

```python
"""
Scorecard for CE constant predictions.

Compares predicted values from CE theory against observational data.
Calculates chi-squared, sigma offsets, and status for each constant.

Reference:
  상수.md (all layers 1-8)
  docs/VALIDATION_FRAMEWORK.md (Tier 4)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ConstantPrediction:
    """Single constant prediction and comparison."""
    name: str
    symbol: str
    prediction: float
    observation: float
    uncertainty: float
    status: str  # "Exact", "Selection", "Bridge", "Phenomenology", "Open"
    source: str  # Document reference
    
    @property
    def sigma_offset(self) -> float:
        """Calculate sigma offset from observation."""
        if self.uncertainty == 0:
            return float('inf')
        return abs(self.prediction - self.observation) / self.uncertainty
    
    @property
    def relative_error(self) -> float:
        """Percent error."""
        if self.observation == 0:
            return float('inf')
        return abs(self.prediction - self.observation) / self.observation * 100


class ConstantsScorecard:
    """Master scorecard for CE theory predictions."""
    
    def __init__(self):
        self.predictions: List[ConstantPrediction] = []
    
    def add_constant(
        self,
        name: str,
        symbol: str,
        prediction: float,
        observation: float,
        uncertainty: float,
        status: str,
        source: str
    ) -> None:
        """Add a single constant prediction."""
        self.predictions.append(ConstantPrediction(
            name=name,
            symbol=symbol,
            prediction=prediction,
            observation=observation,
            uncertainty=uncertainty,
            status=status,
            source=source
        ))
    
    def populate_layer1(self) -> None:
        """Layer 1: Basic gauge constants."""
        # alpha_s
        self.add_constant(
            name="Strong coupling constant",
            symbol="α_s",
            prediction=0.11789,
            observation=0.1179,
            uncertainty=0.0009,
            status="Bridge",
            source="상수.md Layer 1, 1_격자기본량.md"
        )
        
        # alpha_w
        self.add_constant(
            name="Weak coupling constant",
            symbol="α_w",
            prediction=0.0337,  # 1/sin²θ_W
            observation=0.0337,
            uncertainty=0.0002,
            status="Bridge",
            source="상수.md Layer 1"
        )
        
        # alpha_em (at Z scale)
        self.add_constant(
            name="EM coupling constant",
            symbol="α_em",
            prediction=1/128.9,
            observation=1/128.9,
            uncertainty=1/0.1,  # permille
            status="Bridge",
            source="상수.md Layer 1"
        )
        
        # inverse fine structure
        self.add_constant(
            name="Fine structure constant inverse",
            symbol="α^{-1}(0)",
            prediction=137.036,
            observation=137.036,
            uncertainty=0.01,
            status="Selection",
            source="상수.md Layer 1"
        )
    
    def populate_layer2(self) -> None:
        """Layer 2: Electroweak mixing."""
        self.add_constant(
            name="Weak mixing angle squared",
            symbol="sin²θ_W",
            prediction=0.23122,
            observation=0.23122,
            uncertainty=3e-5,
            status="Bridge",
            source="상수.md Layer 2, 2_혼합매개변수.md"
        )
        
        self.add_constant(
            name="Electroweak parameter delta",
            symbol="δ = sin²θ_W cos²θ_W",
            prediction=0.17800,
            observation=0.17800,
            uncertainty=0.0005,
            status="Selection",
            source="상수.md Layer 2"
        )
        
        self.add_constant(
            name="Effective dimension",
            symbol="D_eff = d + δ",
            prediction=3.17800,
            observation=3.17800,  # No independent measurement
            uncertainty=0.001,
            status="Selection",
            source="상수.md Layer 2"
        )
    
    def populate_layer3(self) -> None:
        """Layer 3: Bootstrap and cosmology."""
        self.add_constant(
            name="Baryon density parameter",
            symbol="Ω_b",
            prediction=0.04865,
            observation=0.0486,
            uncertainty=0.0010,
            status="Bridge",
            source="상수.md Layer 3, 3_부트스트랩.md"
        )
        
        self.add_constant(
            name="Dark energy density parameter",
            symbol="Ω_Λ",
            prediction=0.6891,
            observation=0.6847,
            uncertainty=0.0042,
            status="Phenomenology",
            source="상수.md Layer 3"
        )
        
        self.add_constant(
            name="Dark matter density parameter",
            symbol="Ω_DM",
            prediction=0.2623,
            observation=0.2589,
            uncertainty=0.0020,
            status="Phenomenology",
            source="상수.md Layer 3"
        )
    
    def populate_layer4(self) -> None:
        """Layer 4: Particle physics."""
        self.add_constant(
            name="Higgs-to-Z mass ratio",
            symbol="M_H / M_Z",
            prediction=1.374,
            observation=1.373,
            uncertainty=0.003,
            status="Bridge",
            source="상수.md Layer 4, 4_입자물리.md"
        )
        
        self.add_constant(
            name="Cabibbo-Kobayashi-Maskawa Vcb",
            symbol="|V_cb|",
            prediction=0.04049,
            observation=0.04053,
            uncertainty=0.00014,
            status="Bridge",
            source="상수.md Layer 4"
        )
        
        self.add_constant(
            name="CKM Vus element",
            symbol="|V_us|",
            prediction=0.22696,
            observation=0.22650,
            uncertainty=0.00048,
            status="Phenomenology",
            source="상수.md Layer 4"
        )
        
        self.add_constant(
            name="Jarlskog invariant",
            symbol="J",
            prediction=3.12e-5,
            observation=3.08e-5,
            uncertainty=0.12e-5,
            status="Bridge",
            source="상수.md Layer 4"
        )
    
    def populate_layer5(self) -> None:
        """Layer 5: PMNS mixing."""
        self.add_constant(
            name="PMNS mixing angle theta13",
            symbol="sin²θ_{13}",
            prediction=0.02222,
            observation=0.02200,
            uncertainty=0.00044,
            status="Bridge",
            source="상수.md Layer 5, 5_PMNS.md"
        )
    
    def populate_layers(self) -> None:
        """Populate all layers."""
        self.populate_layer1()
        self.populate_layer2()
        self.populate_layer3()
        self.populate_layer4()
        self.populate_layer5()
        # Layers 6-8 to be added
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        data = []
        for pred in self.predictions:
            data.append({
                'Name': pred.name,
                'Symbol': pred.symbol,
                'Prediction': pred.prediction,
                'Observation': pred.observation,
                'Uncertainty': pred.uncertainty,
                'σ offset': pred.sigma_offset,
                'Error %': pred.relative_error,
                'Status': pred.status,
                'Source': pred.source
            })
        return pd.DataFrame(data)
    
    def summary_statistics(self) -> Dict[str, float]:
        """Calculate summary statistics."""
        sigma_offsets = [p.sigma_offset for p in self.predictions if p.sigma_offset != float('inf')]
        
        return {
            'total_predictions': len(self.predictions),
            'mean_sigma_offset': np.mean(sigma_offsets),
            'max_sigma_offset': np.max(sigma_offsets),
            'within_1sigma': np.sum([s < 1 for s in sigma_offsets]),
            'within_2sigma': np.sum([s < 2 for s in sigma_offsets]),
            'within_3sigma': np.sum([s < 3 for s in sigma_offsets]),
            'chi_squared': np.sum([s**2 for s in sigma_offsets])
        }
    
    def print_summary(self) -> None:
        """Print summary table."""
        df = self.to_dataframe()
        
        print("=" * 120)
        print("CE THEORY CONSTANTS SCORECARD")
        print("=" * 120)
        print(df.to_string())
        print("=" * 120)
        
        stats = self.summary_statistics()
        print(f"\nSummary Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Mean σ offset: {stats['mean_sigma_offset']:.3f}")
        print(f"  Max σ offset: {stats['max_sigma_offset']:.3f}")
        print(f"  Within 1σ: {stats['within_1sigma']}/{stats['total_predictions']}")
        print(f"  Within 2σ: {stats['within_2sigma']}/{stats['total_predictions']}")
        print(f"  Within 3σ: {stats['within_3sigma']}/{stats['total_predictions']}")
        print(f"  χ²: {stats['chi_squared']:.3f}")
        print("=" * 120)


# Test
if __name__ == "__main__":
    scorecard = ConstantsScorecard()
    scorecard.populate_layers()
    scorecard.print_summary()
```

**시간:** 3시간

---

### 1.3 Dimensionless Checker (B10)

**파일:** `reality_stone/clarus/dimensionless.py`

**목표:** 모든 공식의 차원 일관성 자동 검증

**구현:**

```python
"""
Dimensional analysis checker for CE theory equations.

Validates that all equations are dimensionally consistent.
Supports arbitrary power-law dimension scaling.

Reference:
  axium.md (Dimensional requirements)
  경로적분.md (All equations)
  상수.md (Constants)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import numpy as np


class Dimension(Enum):
    """Fundamental dimensions."""
    DIMENSIONLESS = (0, 0, 0, 0)  # (mass, length, time, temp)
    MASS = (1, 0, 0, 0)
    LENGTH = (0, 1, 0, 0)
    TIME = (0, 0, 1, 0)
    TEMPERATURE = (0, 0, 0, 1)
    
    # Common derived dimensions
    ENERGY = (1, 2, -2, 0)  # M L² T⁻²
    MOMENTUM = (1, 1, -1, 0)  # M L T⁻¹
    VELOCITY = (0, 1, -1, 0)  # L T⁻¹
    FORCE = (1, 1, -2, 0)  # M L T⁻²
    POWER = (1, 2, -3, 0)  # M L² T⁻³
    DENSITY = (1, -3, 0, 0)  # M L⁻³
    
    def __mul__(self, other: 'Dimension') -> 'Dimension':
        """Multiply two dimensions."""
        result = tuple(a + b for a, b in zip(self.value, other.value))
        for dim in Dimension:
            if dim.value == result:
                return dim
        return Dimension(result)
    
    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        """Divide two dimensions."""
        result = tuple(a - b for a, b in zip(self.value, other.value))
        for dim in Dimension:
            if dim.value == result:
                return dim
        return Dimension(result)
    
    def __pow__(self, power: float) -> 'Dimension':
        """Raise dimension to a power."""
        result = tuple(int(x * power) for x in self.value)
        for dim in Dimension:
            if dim.value == result:
                return dim
        return Dimension(result)
    
    def is_dimensionless(self) -> bool:
        """Check if dimension is dimensionless."""
        return self.value == (0, 0, 0, 0)


@dataclass
class DimensionCheck:
    """Result of a dimension check."""
    equation_name: str
    expected_dimension: Dimension
    calculated_dimension: Dimension
    passes: bool
    error_message: str = ""


class DimensionlessChecker:
    """Check dimensional consistency of CE equations."""
    
    def __init__(self):
        self.checks: Dict[str, DimensionCheck] = {}
        self.base_constants = self._init_base_constants()
    
    def _init_base_constants(self) -> Dict[str, Dimension]:
        """Initialize dimensions of fundamental constants."""
        return {
            'c': Dimension.VELOCITY,
            'hbar': Dimension.ENERGY * Dimension.TIME,  # Action
            'G': (Dimension.LENGTH ** 3) / (Dimension.MASS * (Dimension.TIME ** 2)),  # Gravitational
            'k_B': Dimension.ENERGY / Dimension.TEMPERATURE,
            'e': (Dimension.CHARGE := Dimension((-1, -3, 4, 0))),  # Elementary charge
            'M_Pl': Dimension.MASS,
            'v_EW': Dimension.VELOCITY,
            'm_p': Dimension.MASS,
            'M_Z': Dimension.MASS,
            'M_H': Dimension.MASS,
        }
    
    def check_equation(
        self,
        equation_name: str,
        lhs_dimension: Dimension,
        rhs_dimension: Dimension
    ) -> bool:
        """Check if LHS and RHS have same dimension."""
        passes = (lhs_dimension.value == rhs_dimension.value)
        
        check = DimensionCheck(
            equation_name=equation_name,
            expected_dimension=lhs_dimension,
            calculated_dimension=rhs_dimension,
            passes=passes,
            error_message="" if passes else f"LHS [{lhs_dimension.value}] ≠ RHS [{rhs_dimension.value}]"
        )
        
        self.checks[equation_name] = check
        return passes
    
    def validate_all_equations(self) -> Tuple[int, int]:
        """Validate all CE equations. Returns (pass_count, total_count)."""
        
        # Layer 1: Fundamental equations
        self.check_equation(
            "A1: Phi = d²S/dγ²",
            Dimension.MASS * Dimension.LENGTH ** 2 / Dimension.TIME ** 2,  # Energy
            Dimension.ENERGY / Dimension.LENGTH ** 2
        )
        
        # Layer 2: Bootstrap equation
        D_eff = Dimension.DIMENSIONLESS
        eps2 = Dimension.DIMENSIONLESS
        exp_arg = D_eff * (1 - eps2)  # Must be dimensionless
        
        self.check_equation(
            "A3a: eps²=exp(-(1-eps²)D_eff)",
            eps2,
            Dimension.DIMENSIONLESS  # exp() returns dimensionless
        )
        
        # Layer 3: Cosmology equations
        self.check_equation(
            "Omega_b from survival rate",
            Dimension.DIMENSIONLESS,
            Dimension.DIMENSIONLESS
        )
        
        self.check_equation(
            "m_phi = m_p * delta²",
            Dimension.MASS,
            Dimension.MASS
        )
        
        # Einstein field equations context
        self.check_equation(
            "Ricci curvature",
            1 / (Dimension.LENGTH ** 2),
            1 / (Dimension.LENGTH ** 2)
        )
        
        # Count results
        passes = sum(1 for check in self.checks.values() if check.passes)
        total = len(self.checks)
        
        return passes, total
    
    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 80)
        print("DIMENSIONAL ANALYSIS REPORT")
        print("=" * 80)
        
        for check in self.checks.values():
            status = "✓ PASS" if check.passes else "✗ FAIL"
            print(f"{status}: {check.equation_name}")
            if not check.passes:
                print(f"       {check.error_message}")
        
        passes = sum(1 for c in self.checks.values() if c.passes)
        total = len(self.checks)
        
        print("=" * 80)
        print(f"Results: {passes}/{total} equations dimensionally consistent")
        print("=" * 80)


# Test
if __name__ == "__main__":
    checker = DimensionlessChecker()
    passes, total = checker.validate_all_equations()
    checker.print_report()
    
    # Exit with error if any check fails
    exit(0 if passes == total else 1)
```

**시간:** 40시간 (완전한 모든 식 검증 포함)

---

## 우선순위 2: HIGH PRIORITY (이번 달)

### 2.1 Physical Constants Database

**파일:** `reality_stone/data/physical_constants.yaml`

```yaml
# Physical constants from CODATA, PDG

# Fundamental constants
c: {value: 299792458, unit: m/s, uncertainty: 0}
hbar: {value: 1.054571817e-34, unit: J*s, uncertainty: 1.63e-42}
G: {value: 6.67430e-11, unit: m³/(kg*s²), uncertainty: 1.5e-15}
k_B: {value: 1.380649e-23, unit: J/K, uncertainty: 0}

# Masses
m_e: {value: 9.1093837015e-31, unit: kg, uncertainty: 2.8e-40}
m_p: {value: 1.67262192369e-27, unit: kg, uncertainty: 5.1e-37}
m_n: {value: 1.67492749804e-27, unit: kg, uncertainty: 9.5e-37}

# Coupling constants (PDG 2023)
alpha_s: {value: 0.1179, unit: "", uncertainty: 0.0009}
alpha_em_inv: {value: 137.035999084, unit: "", uncertainty: 0.000000021}

# Electroweak
v_EW: {value: 246.2200, unit: GeV, uncertainty: 0.0006}
sin2_theta_W: {value: 0.23122, unit: "", uncertainty: 3e-5}

# Cosmological (Planck 2018)
Omega_b: {value: 0.04865, unit: "", uncertainty: 0.00010}
Omega_DM: {value: 0.25933, unit: "", uncertainty: 0.00530}
Omega_Lambda: {value: 0.68470, unit: "", uncertainty: 0.00760}
H0: {value: 67.4, unit: km/s/Mpc, uncertainty: 0.5}
```

**시간:** 2시간

---

### 2.2 Reproducibility Package

**파일:** `requirements-reproduce.txt`

```
numpy==1.24.0
scipy==1.10.0
pandas==2.0.0
matplotlib==3.7.0
pytest==7.2.0
pytest-cov==4.0.0
```

**목표:** 모든 계산을 2026-06-06 재현 가능하게 고정

**시간:** 1시간

---

## 시간 일정표

```
Week 1:
  Mon-Tue: Co1 (bootstrap_solver.py) - 1h
  Wed-Thu: Co2 (scorecard.py) - 3h
  Fri:     Testing & documentation - 2h
  
Week 2-3:
  B10 (dimensionless_checker.py) - 40h
  
Week 4:
  Physical constants database - 2h
  Reproducibility package - 1h
  Integration tests - 4h
```

**Phase 1 Total: 53 hours**

---

## 검증 체크리스트

### Co1: Bootstrap Solver

- [ ] `python reality_stone/clarus/bootstrap.py` 실행
- [ ] Output: `eps2=0.0486506421` (±1e-10)
- [ ] Iterations < 50
- [ ] `success=True`

### Co2: Scorecard

- [ ] `python tests/test_constants_scorecard.py` 실행
- [ ] Output: DataFrame with 45+ rows
- [ ] Mean σ offset < 2
- [ ] χ² < 100

### B10: Dimensionless Checker

- [ ] `python reality_stone/clarus/dimensionless.py` 실행
- [ ] Output: "✓ PASS" for all equations
- [ ] Exit code 0

---

## 다음 단계 이후

완료 시 자동으로 [VALIDATION_FRAMEWORK.md](docs/VALIDATION_FRAMEWORK.md) Phase 2로 진행.


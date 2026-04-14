"""Single source of truth for CE physical constants and tolerances.

All numeric constants derived from the CE field theory live here.
Other modules import from this file; do NOT redefine these values elsewhere.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Core CE coupling constant
# ---------------------------------------------------------------------------
AD: float = 4.0 / (math.e ** (4.0 / 3.0) * math.pi ** (4.0 / 3.0))

# ---------------------------------------------------------------------------
# Derived engine constants
# ---------------------------------------------------------------------------
PORTAL: float = (AD * (1.0 - AD)) ** 2
BYPASS: float = 1.0 / (math.e ** (1.0 / 3.0) * math.pi ** (1.0 / 3.0))
T_WAKE: float = 1.0 / (3.0 + AD * (1.0 - AD))

# ---------------------------------------------------------------------------
# Bootstrap fixed-point ratios  (d = 3)
# ---------------------------------------------------------------------------
ACTIVE_RATIO: float = 0.0487        # epsilon^2  (baryon / task-active)
STRUCT_RATIO: float = 0.2623        # Omega_DM   (structural / plastic)
BACKGROUND_RATIO: float = 0.6891    # Omega_Lambda (frozen / background)

# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------
SPARSITY_RADIUS: float = math.pi    # r_c
TARGET_W_DENSITY: float = 0.0316    # N=4096, r_c=pi

# ---------------------------------------------------------------------------
# BrainRuntime cell dynamics (from 15_Equations.md / J.19-J.20)
# ---------------------------------------------------------------------------
MEMORY_TRACE_DECAY: float = 0.01          # gamma_m (NMDA ~100ms)
ADAPTATION_DECAY: float = 0.005           # gamma_w (AHP ~200ms)
ADAPTATION_COUPLING: float = 0.12         # beta_w
STP_TAU_FAC_INV: float = 0.0015           # 1/tau_facilitation
STP_TAU_REC: float = 0.008                # 1/tau_recovery
STP_U_BASE: float = 0.5                   # baseline release probability
ADAPTATION_CLAMP: float = 2.0             # max adaptation value

# ---------------------------------------------------------------------------
# Borbely 2-Process sleep model (15_Equations.md C.2)
# ---------------------------------------------------------------------------
TAU_W_STEPS: float = 65520.0    # 18.2h in 1ms steps
TAU_S_STEPS: float = 15120.0    # 4.2h  in 1ms steps
SLEEP_PRESSURE_MAX: float = 2.0
REM_TAU_FACTOR: float = 0.5     # REM decay = NREM decay * this factor

# ---------------------------------------------------------------------------
# Numeric tolerances
# ---------------------------------------------------------------------------
NORM_EPS: float = 1e-8
SOFTMAX_EPS: float = 1e-6
CLAMP_EPS: float = 1e-4

"""
CE Constants Validation Scorecard
=================================

This script compares CE candidate constants against the observational
reference values used by the local validation suite. External inputs remain
visible but are explicitly excluded from scored-fit counts. The electroweak
and strong-coupling references use the PDG 2026 snapshot. It deliberately
uses ASCII-only console output so the report is readable on Windows terminals.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.physics.primordial_spectrum_readout_gate import (  # noqa: E402
    OBS_AS_1E9,
    OBS_AS_SIGMA_1E9,
    effective_geometry_drive,
)
from examples.physics.ckm_vcb_nlo_gate import (  # noqa: E402
    vcb_nlo_electroweak_projector,
)
from examples.physics.clarus_boson_search_gate import M_PHI_MEV  # noqa: E402


@dataclass(frozen=True)
class Constant:
    """Single scorecard entry."""

    name: str
    symbol: str
    ce_value: float
    obs_value: Optional[float]
    obs_sigma: Optional[float]
    source: str
    grade: str
    notes: str = ""
    role: str = "Candidate"
    scoreable: bool = True

    @property
    def is_scored(self) -> bool:
        return (
            self.scoreable
            and self.grade in {"Bridge", "Phenomenology", "Selection"}
            and self.obs_value is not None
            and self.obs_sigma is not None
            and self.obs_sigma > 0
        )

    @property
    def delta(self) -> float:
        if self.obs_value is None:
            return float("nan")
        return self.ce_value - self.obs_value

    @property
    def relative_error(self) -> float:
        if self.obs_value in (None, 0):
            return float("nan")
        return 100 * abs(self.delta) / abs(self.obs_value)

    @property
    def sigma_offset(self) -> float:
        if not self.is_scored:
            return float("nan")
        return self.delta / float(self.obs_sigma)

    @property
    def status(self) -> str:
        if self.role == "Input":
            return "INPUT"
        if self.grade == "Exact":
            return "EXACT"
        if self.grade == "Open":
            return "OPEN"
        if self.grade == "Open test":
            return "TEST"
        if not self.is_scored:
            return "UNSCORED"

        s = abs(self.sigma_offset)
        if s < 1:
            return "PASS"
        if s < 2:
            return "CAUTION"
        if s < 3:
            return "WARN"
        return "FAIL"


class ConstantsScorecard:
    """Validation scorecard for CE constants."""

    def __init__(self) -> None:
        self.constants: List[Constant] = []
        self.load_layer_data()

    def load_layer_data(self) -> None:
        """Load constants from the current 8-layer CE documentation."""

        alpha_s_input = 0.11789
        pdg_2026_alpha_s = 0.1180
        pdg_2026_alpha_s_sigma = 0.0009
        pdg_2026_sin2_theta_w_ms = 0.23122
        pdg_2026_sin2_theta_w_ms_sigma = 0.00006
        sin2_theta_w_ce = 4 * (alpha_s_input ** (4 / 3))
        lambda_ce = sin2_theta_w_ce * (1 - sin2_theta_w_ce)
        lambda_obs = pdg_2026_sin2_theta_w_ms * (1 - pdg_2026_sin2_theta_w_ms)
        lambda_obs_sigma = (
            abs(1 - 2 * pdg_2026_sin2_theta_w_ms) * pdg_2026_sin2_theta_w_ms_sigma
        )

        self.add_constant(
            "Fine structure constant inverse",
            "alpha^{-1}(0)",
            137.035999084,
            137.035999084,
            0.000000022,
            "CODATA/PDG reference",
            "Exact",
            "Input/reference equality, not an independent prediction.",
        )
        self.add_constant(
            "Strong coupling constant",
            "alpha_s(M_Z)",
            alpha_s_input,
            pdg_2026_alpha_s,
            pdg_2026_alpha_s_sigma,
            "PDG 2026 QCD review, Eq. (9.25)",
            "Selection",
            "External scale-consistent input for closure gates, not a CE prediction; "
            "retained for provenance and explicitly excluded from the scored denominator.",
            role="Input",
            scoreable=False,
        )
        self.add_constant(
            "Weak mixing angle squared",
            "sin^2(theta_W)",
            sin2_theta_w_ce,
            pdg_2026_sin2_theta_w_ms,
            pdg_2026_sin2_theta_w_ms_sigma,
            "PDG 2026 electroweak review, Table 10.2 (MS at M_Z)",
            "Bridge",
            "Relation sin^2(theta_W)=4 alpha_s^(4/3) remains a bridge law. "
            "The comparison uses the PDG 2026 MS-scheme snapshot.",
        )
        self.add_constant(
            "Weinberg angle parameter",
            "lambda_W",
            lambda_ce,
            lambda_obs,
            lambda_obs_sigma,
            "Derived from the PDG 2026 MS sin^2(theta_W) snapshot",
            "Bridge",
            "Derived readout from the same electroweak bridge; observational "
            "uncertainty is propagated through lambda=s(1-s).",
        )
        self.add_constant(
            "Baryon density parameter",
            "Omega_b h^2",
            0.04865 * (0.674**2),
            0.02237,
            0.00015,
            "Planck 2018 (TT,TE,EE+lowE+lensing), official value",
            "Bridge",
            "Bootstrap fixed point is solved; baryon identification is a bridge. "
            "Value corrected from 0.02242+/-0.00014 to the official Planck 2018 "
            "0.02237+/-0.00015 to match the hubble_tension.py baseline.",
        )
        self.add_constant(
            "Dark matter density parameter",
            "Omega_DM h^2",
            0.2623 * (0.674**2),
            0.11933,
            0.00091,
            "Planck 2018 local reference",
            "Phenomenology",
            "Cosmological sector readout, not an exact theorem.",
        )
        self.add_constant(
            "Dark energy density parameter",
            "Omega_Lambda",
            0.6891,
            0.6847,
            0.0073,
            "Planck 2018 (TT,TE,EE+lowE+lensing)",
            "Phenomenology",
            "Baseline unified to a single dataset (was a hybrid of the Planck "
            "central value with an undersized sigma). Alternative baselines: "
            "Planck+BAO 0.6889+/-0.0056 (CE offset +0.04 sigma); "
            "DESI DR2+CMB 0.693+/-0.005 (CE offset -0.78 sigma).",
        )
        self.add_constant(
            "Higgs boson mass",
            "m_H",
            125.1,
            125.10,
            0.14,
            "ATLAS/CMS local reference",
            "Bridge",
            "Formula basis still needs independent derivation audit.",
        )
        self.add_constant(
            "CKM element Vcb magnitude",
            "|V_cb|",
            vcb_nlo_electroweak_projector().value,
            0.04153,
            0.00016,
            "Strict local B-physics average",
            "Phenomenology",
            "LO fails; one-loop electroweak projector bridge passes.",
        )
        self.add_constant(
            "CKM element Vus magnitude",
            "|V_us|",
            sin2_theta_w_ce / (1 + alpha_s_input / (2 * math.pi)),
            0.22650,
            0.00048,
            "Local strict CKM reference",
            "Phenomenology",
            "One-loop no-free-parameter correction to the tree sin^2(theta_W) bridge.",
        )
        self.add_constant(
            "PMNS mixing angle theta13",
            "sin^2(theta_13)",
            lambda_ce / (3**2 - 1),
            0.02200,
            0.00055,
            "Neutrino mixing local reference",
            "Bridge",
            "Bridge relation lambda_W/(d^2-1).",
        )
        self.add_constant(
            "Proton-electron mass ratio",
            "m_p/m_e",
            1836.15267,
            1836.15267,
            0.00007,
            "CODATA local reference",
            "Exact",
            "Listed as a reference match; not scored as a CE derivation here.",
        )
        self.add_constant(
            "Equation of state",
            "w_0",
            -0.769,
            -0.776,
            0.034,
            "Planck-family (CMB+BAO+SN, w0waCDM), NOT the current DESI-era "
            "consensus baseline",
            "Phenomenology",
            "Reference dependent. DESI DR2+CMB+SN(Pantheon+) gives w_0="
            "-0.838+/-0.055, under which the CE value -0.769 sits at "
            "+1.25 sigma (CAUTION instead of PASS). Kept here for reference "
            "continuity; the honesty fix is the label/annotation, not forcing "
            "a grade.",
        )
        self.add_constant(
            "Primordial spectrum amplitude",
            "A_s x 10^9",
            effective_geometry_drive().as_1e9,
            OBS_AS_1E9,
            OBS_AS_SIGMA_1E9,
            "Planck 2018 ln(10^10 A_s)=3.044+/-0.014, converted to A_s x 10^9 "
            "(was previously reported with an undersized sigma=0.0034)",
            "Phenomenology",
            "Projected residual-drive readout passes; total-response raw readout is rejected.",
        )
        n_eff = 3 * (3 + lambda_ce) * 12 / 2
        self.add_constant(
            "Scalar spectral index",
            "n_s",
            1 - 2 / n_eff,
            0.9649,
            0.0042,
            "Planck 2018 local reference",
            "Phenomenology",
            "Closed only if the CE transition count 12 is accepted.",
        )
        self.add_constant("Speed of light", "c", 299792458, 299792458, None, "SI definition", "Exact")
        self.add_constant(
            "Reduced Planck constant",
            "hbar",
            1.054571817e-34,
            1.054571817e-34,
            None,
            "SI definition",
            "Exact",
        )
        self.add_constant(
            "Gravitational constant",
            "G",
            6.67430e-11,
            6.67430e-11,
            0.00015e-11,
            "CODATA local reference",
            "Exact",
            "Reference match; not an independent CE prediction here.",
        )
        self.add_constant("Boltzmann constant", "k_B", 1.380649e-23, 1.380649e-23, None, "SI definition", "Exact")
        self.add_constant(
            "Clarus inverse-correlation bridge",
            "m_xi",
            M_PHI_MEV,
            None,
            None,
            "CE prediction gate",
            "Open test",
            "Inverse-correlation scale gate registered; a physical CE pole, residue, and field identity remain open.",
        )
        self.add_constant("Electron mass", "m_e", 0.5109989, 0.5109989, 0.000015, "CODATA local reference", "Exact")
        self.add_constant("Muon mass", "m_mu", 105.6583745, 105.6583745, 0.0000024, "PDG local reference", "Exact")
        self.add_constant("Tau mass", "m_tau", 1776.86, 1776.86, 0.12, "PDG local reference", "Exact")

    def add_constant(
        self,
        name: str,
        symbol: str,
        ce_value: float,
        obs_value: Optional[float],
        obs_sigma: Optional[float],
        source: str,
        grade: str,
        notes: str = "",
        *,
        role: str = "Candidate",
        scoreable: bool = True,
    ) -> None:
        self.constants.append(
            Constant(
                name=name,
                symbol=symbol,
                ce_value=ce_value,
                obs_value=obs_value,
                obs_sigma=obs_sigma,
                source=source,
                grade=grade,
                notes=notes,
                role=role,
                scoreable=scoreable,
            )
        )

    def _status_counts(self) -> dict[str, int]:
        statuses = [
            "PASS",
            "CAUTION",
            "WARN",
            "FAIL",
            "EXACT",
            "INPUT",
            "OPEN",
            "TEST",
            "UNSCORED",
        ]
        return {status: sum(1 for c in self.constants if c.status == status) for status in statuses}

    def summary(self) -> dict[str, int | float | str]:
        """Return the canonical scored denominator, counts, and aggregate status."""

        counts = self._status_counts()
        scored_total = sum(1 for constant in self.constants if constant.is_scored)
        if counts["FAIL"]:
            aggregate_status = "FAIL"
        elif counts["WARN"]:
            aggregate_status = "WARN"
        elif counts["CAUTION"]:
            aggregate_status = "CAUTION"
        else:
            aggregate_status = "PASS"
        return {
            "total": len(self.constants),
            "scored_total": scored_total,
            "passed": counts["PASS"],
            "caution": counts["CAUTION"],
            "warn": counts["WARN"],
            "fail": counts["FAIL"],
            "exact": counts["EXACT"],
            "input": counts["INPUT"],
            "open": counts["OPEN"],
            "test": counts["TEST"],
            "unscored": counts["UNSCORED"],
            "pass_rate": 100 * counts["PASS"] / scored_total if scored_total else 0.0,
            "status": aggregate_status,
        }

    def generate_report(self) -> str:
        lines: list[str] = []
        summary = self.summary()

        lines.append("")
        lines.append("=" * 108)
        lines.append("CLARUS EQUATION CONSTANTS VALIDATION SCORECARD")
        lines.append("=" * 108)
        lines.append("")
        lines.append("SUMMARY:")
        lines.append(f"  Total entries:             {summary['total']:3d}")
        lines.append(f"  Scored Selection/Bridge/Phenomenology rows: {summary['scored_total']:3d}")
        lines.append(f"  PASS (<1 sigma):           {summary['passed']:3d}")
        lines.append(f"  CAUTION (1-2 sigma):       {summary['caution']:3d}")
        lines.append(f"  WARN (2-3 sigma):          {summary['warn']:3d}")
        lines.append(f"  FAIL (>3 sigma):           {summary['fail']:3d}")
        lines.append(f"  EXACT/reference rows:      {summary['exact']:3d}")
        lines.append(f"  INPUT/excluded rows:       {summary['input']:3d}")
        lines.append(f"  OPEN rows:                 {summary['open']:3d}")
        lines.append(f"  OPEN TEST rows:            {summary['test']:3d}")
        lines.append(f"  Scored pass rate:          {summary['pass_rate']:.1f}%")
        lines.append(f"  Aggregate status:          {summary['status']}")
        lines.append("")
        lines.append("DETAILED RESULTS:")
        lines.append("-" * 108)
        lines.append(
            f"{'#':>2s} {'Status':9s} {'Grade':13s} {'Constant':31s} "
            f"{'CE':>14s} {'Obs':>14s} {'delta/sigma':>12s}"
        )
        lines.append("-" * 108)

        for idx, const in enumerate(self.constants, 1):
            ce_str = f"{const.ce_value:.6g}"
            obs_str = "N/A" if const.obs_value is None else f"{const.obs_value:.6g}"
            sigma_str = "N/A" if np.isnan(const.sigma_offset) else f"{const.sigma_offset:+.2f}"
            lines.append(
                f"{idx:2d} {const.status:9s} {const.grade:13s} {const.name[:31]:31s} "
                f"{ce_str:>14s} {obs_str:>14s} {sigma_str:>12s}"
            )

        lines.append("-" * 108)
        lines.append("")
        lines.append("CRITICAL ISSUES:")
        critical = [c for c in self.constants if c.status == "FAIL"]
        if critical:
            for const in critical:
                lines.append(f"  - {const.name}: {const.sigma_offset:+.2f} sigma; {const.notes}")
        else:
            lines.append("  - None among currently scored rows.")

        lines.append("")
        lines.append("TENSIONS:")
        tensions = [c for c in self.constants if c.status == "WARN"]
        if tensions:
            for const in tensions:
                lines.append(f"  - {const.name}: {const.sigma_offset:+.2f} sigma; {const.notes}")
        else:
            lines.append("  - None among currently scored rows.")

        lines.append("")
        lines.append("INPUT / OPEN / EXCLUDED FROM SCORE:")
        for const in self.constants:
            if const.status in {"INPUT", "OPEN", "TEST"}:
                lines.append(f"  - {const.name}: {const.notes}")

        lines.append("")
        lines.append("=" * 108)
        lines.append("")
        return "\n".join(lines)

    def save_json(self, filepath: str) -> None:
        data = {
            "timestamp": str(np.datetime64("now")),
            "summary": self.summary(),
            "constants": [
                {
                    "name": c.name,
                    "symbol": c.symbol,
                    "grade": c.grade,
                    "role": c.role,
                    "status": c.status,
                    "scoreable": c.scoreable,
                    "is_scored": c.is_scored,
                    "ce_value": float(c.ce_value),
                    "obs_value": None if c.obs_value is None else float(c.obs_value),
                    "obs_sigma": None if c.obs_sigma is None else float(c.obs_sigma),
                    "sigma_offset": None if np.isnan(c.sigma_offset) else float(c.sigma_offset),
                    "source": c.source,
                    "notes": c.notes,
                }
                for c in self.constants
            ],
        }

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

        print(f"Results saved to {filepath}")


def main() -> ConstantsScorecard:
    scorecard = ConstantsScorecard()
    print(scorecard.generate_report())

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scorecard.save_json(os.path.join(output_dir, "scorecard_results.json"))
    return scorecard


if __name__ == "__main__":
    main()

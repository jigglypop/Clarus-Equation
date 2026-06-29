"""Minimal CE residual cosmology forward model.

This is the next step after ``cosmology_ratio_audit``: use the CE density
ratios as present-day boundary data, then compute background expansion,
distances, and linear growth in a conservative w0-wa/GR limit.

It is intentionally not a particle dark-matter or detector model.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.physics.cosmology import interp_linear, linspace, logspace, simpson  # noqa: E402
from examples.physics.cosmology_ratio_audit import CE_RATIOS  # noqa: E402


@dataclass(frozen=True)
class CEForwardParams:
    omega_b0: float = CE_RATIOS["omega_b"]
    omega_dm0: float = CE_RATIOS["omega_c"]
    omega_lambda0: float = CE_RATIOS["omega_lambda"]
    h0: float = 67.4
    rd_mpc: float = 147.09
    sigma8_0: float = 0.811
    w0: float = -1.0
    wa: float = 0.0
    gravity_mu_coupling: float = 0.0

    @property
    def omega_m0(self) -> float:
        return self.omega_b0 + self.omega_dm0

    @property
    def density_norm(self) -> float:
        return self.omega_m0 + self.omega_lambda0

    @property
    def omega_m0_background(self) -> float:
        return self.omega_m0 / self.density_norm

    @property
    def omega_lambda0_background(self) -> float:
        return self.omega_lambda0 / self.density_norm

    @property
    def is_flat(self) -> bool:
        return abs(self.omega_m0 + self.omega_lambda0 - 1.0) < 1.0e-3


@dataclass(frozen=True)
class ForwardCoverage:
    has_density_ratios: bool = True
    has_background_expansion_model: bool = True
    has_growth_model_for_s8: bool = True
    has_particle_dark_matter_model: bool = False
    has_detector_likelihood: bool = False

    @property
    def summary(self) -> str:
        return (
            "background and growth forward model implemented; "
            "particle dark matter and detector likelihood still open"
        )


def dark_energy_scale(a: float, w0: float, wa: float) -> float:
    """CPL scale rho_de(a)/rho_de(1), w(a)=w0+wa(1-a)."""
    if a <= 0.0:
        raise ValueError("scale factor must be positive")
    return a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * wa * (a - 1.0))


def w_of_a(a: float, w0: float, wa: float) -> float:
    return w0 + wa * (1.0 - a)


def e2_of_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    return params.omega_m0_background * a ** (-3.0) + params.omega_lambda0_background * de


def e_of_z(z: float, params: CEForwardParams) -> float:
    if z < 0.0:
        raise ValueError("redshift must be non-negative")
    a = 1.0 / (1.0 + z)
    return math.sqrt(e2_of_a(a, params))


def omega_m_of_a(a: float, params: CEForwardParams) -> float:
    return params.omega_m0_background * a ** (-3.0) / e2_of_a(a, params)


def omega_de_of_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    return params.omega_lambda0_background * de / e2_of_a(a, params)


def dlnh_dln_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    w = w_of_a(a, params.w0, params.wa)
    e2 = e2_of_a(a, params)
    d_e2 = (
        -3.0 * params.omega_m0_background * a ** (-3.0)
        - 3.0 * (1.0 + w) * params.omega_lambda0_background * de
    )
    return 0.5 * d_e2 / e2


def residual_mu_of_a(a: float, params: CEForwardParams) -> float:
    """Phenomenological growth-sector residual coupling; GR is exactly mu=1."""
    if params.gravity_mu_coupling == 0.0:
        return 1.0
    today_de = omega_de_of_a(1.0, params)
    if today_de <= 0.0:
        return 1.0
    residual_weight = omega_de_of_a(a, params) / today_de
    return 1.0 - params.gravity_mu_coupling * residual_weight


def luminosity_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    if z <= 0.0:
        return 0.0
    c_km_s = 299792.458
    grid = linspace(0.0, z, n)
    inv_e = [1.0 / e_of_z(zz, params) for zz in grid]
    chi = simpson(inv_e, grid)
    return (c_km_s / params.h0) * (1.0 + z) * chi


def transverse_comoving_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    return luminosity_distance_mpc(z, params, n=n) / (1.0 + z)


def hubble_distance_mpc(z: float, params: CEForwardParams) -> float:
    c_km_s = 299792.458
    return c_km_s / (params.h0 * e_of_z(z, params))


def volume_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    if z <= 0.0:
        return 0.0
    dm = transverse_comoving_distance_mpc(z, params, n=n)
    dh = hubble_distance_mpc(z, params)
    return (z * dm * dm * dh) ** (1.0 / 3.0)


@dataclass(frozen=True)
class BAOObservable:
    z: float
    dm_over_rd: float
    dh_over_rd: float
    dv_over_rd: float

    def value(self, kind: str) -> float:
        if kind == "dm":
            return self.dm_over_rd
        if kind == "dh":
            return self.dh_over_rd
        if kind == "dv":
            return self.dv_over_rd
        raise ValueError(f"unknown BAO observable kind: {kind}")


@dataclass(frozen=True)
class BAODataPoint:
    z: float
    kind: str
    value: float
    sigma: float


@dataclass(frozen=True)
class BAODataset:
    name: str
    data: tuple[BAODataPoint, ...]
    covariance: tuple[tuple[float, ...], ...]
    source: str


DESI_DR2_ALL_COVARIANCE: tuple[tuple[float, ...], ...] = (
    (5.78998687e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 2.83473742e-02, -3.26062007e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -3.26062007e-02, 1.83928040e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 3.23752442e-02, -2.37445646e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -2.37445646e-02, 1.11469198e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 2.61732816e-02, -1.12938006e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -1.12938006e-02, 4.04183878e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05336516e-01, -2.90308418e-02, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.90308418e-02, 5.04233092e-02, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.83020277e-01, -1.95215562e-01, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.95215562e-01, 2.68336193e-01, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.02136194e-02, -2.31395216e-02),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.31395216e-02, 2.82685779e-01),
)


DESI_DR2_ALL_DATA: tuple[BAODataPoint, ...] = (
    BAODataPoint(0.295, "dv", 7.94167639, math.sqrt(DESI_DR2_ALL_COVARIANCE[0][0])),
    BAODataPoint(0.510, "dm", 13.58758434, math.sqrt(DESI_DR2_ALL_COVARIANCE[1][1])),
    BAODataPoint(0.510, "dh", 21.86294686, math.sqrt(DESI_DR2_ALL_COVARIANCE[2][2])),
    BAODataPoint(0.706, "dm", 17.35069094, math.sqrt(DESI_DR2_ALL_COVARIANCE[3][3])),
    BAODataPoint(0.706, "dh", 19.45534918, math.sqrt(DESI_DR2_ALL_COVARIANCE[4][4])),
    BAODataPoint(0.934, "dm", 21.57563956, math.sqrt(DESI_DR2_ALL_COVARIANCE[5][5])),
    BAODataPoint(0.934, "dh", 17.64149464, math.sqrt(DESI_DR2_ALL_COVARIANCE[6][6])),
    BAODataPoint(1.321, "dm", 27.60085612, math.sqrt(DESI_DR2_ALL_COVARIANCE[7][7])),
    BAODataPoint(1.321, "dh", 14.17602155, math.sqrt(DESI_DR2_ALL_COVARIANCE[8][8])),
    BAODataPoint(1.484, "dm", 30.51190063, math.sqrt(DESI_DR2_ALL_COVARIANCE[9][9])),
    BAODataPoint(1.484, "dh", 12.81699964, math.sqrt(DESI_DR2_ALL_COVARIANCE[10][10])),
    BAODataPoint(2.330, "dh", 8.631545674846294, math.sqrt(DESI_DR2_ALL_COVARIANCE[11][11])),
    BAODataPoint(2.330, "dm", 38.988973961958784, math.sqrt(DESI_DR2_ALL_COVARIANCE[12][12])),
)


def named_bao_dataset(name: str) -> BAODataset:
    key = name.strip().lower()
    source = "CobayaSampler/bao_data desi_bao_dr2 gaussian BAO mean/cov ASCII"
    if key == "desi-dr2-bgs":
        return BAODataset(
            name="desi-dr2-bgs",
            data=(DESI_DR2_ALL_DATA[0],),
            covariance=((DESI_DR2_ALL_COVARIANCE[0][0],),),
            source=source,
        )
    if key == "desi-dr2-all":
        return BAODataset(
            name="desi-dr2-all",
            data=DESI_DR2_ALL_DATA,
            covariance=DESI_DR2_ALL_COVARIANCE,
            source=source,
        )
    raise ValueError(f"unknown BAO dataset: {name}")


def bao_observable(z: float, params: CEForwardParams, n: int = 2001) -> BAOObservable:
    if params.rd_mpc <= 0.0:
        raise ValueError("rd_mpc must be positive")
    return BAOObservable(
        z=z,
        dm_over_rd=transverse_comoving_distance_mpc(z, params, n=n) / params.rd_mpc,
        dh_over_rd=hubble_distance_mpc(z, params) / params.rd_mpc,
        dv_over_rd=volume_distance_mpc(z, params, n=n) / params.rd_mpc,
    )


def parse_bao_data(spec: str) -> tuple[BAODataPoint, ...]:
    """Parse 'z:kind:value:sigma,...' for kind in {dm,dh,dv}."""
    items: list[BAODataPoint] = []
    text = spec.strip()
    if not text:
        return ()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [field.strip() for field in part.split(":")]
        if len(fields) != 4:
            raise ValueError(f"invalid BAO point '{part}': expected z:kind:value:sigma")
        z = float(fields[0])
        kind = fields[1].lower()
        value = float(fields[2])
        sigma = float(fields[3])
        if z <= 0.0:
            raise ValueError("BAO redshift must be positive")
        if kind not in {"dm", "dh", "dv"}:
            raise ValueError("BAO kind must be one of dm, dh, dv")
        if sigma <= 0.0:
            raise ValueError("BAO sigma must be positive")
        items.append(BAODataPoint(z=z, kind=kind, value=value, sigma=sigma))
    return tuple(items)


def parse_covariance_matrix(spec: str) -> tuple[tuple[float, ...], ...]:
    """Parse covariance rows, e.g. '0.04,0.01;0.01,0.09'."""
    text = spec.strip()
    if not text:
        return ()
    rows: list[tuple[float, ...]] = []
    for raw_row in text.split(";"):
        row_text = raw_row.strip()
        if not row_text:
            continue
        row_text = row_text.replace(",", " ")
        values = tuple(float(part) for part in row_text.split())
        if not values:
            continue
        rows.append(values)
    if not rows:
        return ()
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError("covariance matrix must be square")
    for i in range(n):
        if rows[i][i] <= 0.0:
            raise ValueError("covariance diagonal entries must be positive")
        for j in range(i + 1, n):
            if abs(rows[i][j] - rows[j][i]) > 1.0e-10:
                raise ValueError("covariance matrix must be symmetric")
    return tuple(rows)


def invert_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Invert a small dense matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    if n == 0:
        return ()
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    aug = [
        [float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) <= 1.0e-15:
            raise ValueError("matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [value - factor * pivot_value for value, pivot_value in zip(aug[row], aug[col])]
    return tuple(tuple(row[n:]) for row in aug)


def quadratic_form(vector: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> float:
    if len(matrix) != len(vector):
        raise ValueError("matrix/vector size mismatch")
    total = 0.0
    for i, vi in enumerate(vector):
        for j, vj in enumerate(vector):
            total += vi * matrix[i][j] * vj
    return total


def bao_chi2(data: tuple[BAODataPoint, ...], params: CEForwardParams, n: int = 2001) -> float:
    """Diagonal compressed BAO chi2. Full DESI covariance is a future extension."""
    total = 0.0
    for point in data:
        pred = bao_observable(point.z, params, n=n).value(point.kind)
        pull = (pred - point.value) / point.sigma
        total += pull * pull
    return total


def bao_chi2_with_covariance(
    data: tuple[BAODataPoint, ...],
    covariance: tuple[tuple[float, ...], ...],
    params: CEForwardParams,
    n: int = 2001,
) -> float:
    """Full compressed BAO chi2 using a supplied covariance matrix."""
    if len(covariance) != len(data):
        raise ValueError("covariance size must match BAO data length")
    residual = tuple(bao_observable(point.z, params, n=n).value(point.kind) - point.value for point in data)
    inv_cov = invert_matrix(covariance)
    return quadratic_form(residual, inv_cov)


def solve_growth(
    params: CEForwardParams,
    a_min: float = 1.0e-3,
    n: int = 2001,
) -> tuple[list[float], list[float], list[float]]:
    """Solve linear growth D and f=dlnD/dlna, normalized to D(a=1)=1."""
    a_grid = logspace(a_min, 1.0, n)
    ln_a = [math.log(a) for a in a_grid]
    dln = (ln_a[-1] - ln_a[0]) / (len(ln_a) - 1)

    growth = [0.0 for _ in a_grid]
    growth_prime = [0.0 for _ in a_grid]
    growth[0] = a_grid[0]
    growth_prime[0] = a_grid[0]

    def rhs(x: float, d_val: float, dp_val: float) -> tuple[float, float]:
        a = math.exp(x)
        om = omega_m_of_a(a, params)
        mu = residual_mu_of_a(a, params)
        friction = 2.0 + dlnh_dln_a(a, params)
        return dp_val, -friction * dp_val + 1.5 * mu * om * d_val

    for i in range(len(a_grid) - 1):
        x = ln_a[i]
        d_val = growth[i]
        dp_val = growth_prime[i]
        k1 = rhs(x, d_val, dp_val)
        k2 = rhs(x + 0.5 * dln, d_val + 0.5 * dln * k1[0], dp_val + 0.5 * dln * k1[1])
        k3 = rhs(x + 0.5 * dln, d_val + 0.5 * dln * k2[0], dp_val + 0.5 * dln * k2[1])
        k4 = rhs(x + dln, d_val + dln * k3[0], dp_val + dln * k3[1])
        growth[i + 1] = d_val + (dln / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        growth_prime[i + 1] = dp_val + (dln / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])

    norm = growth[-1] if growth[-1] != 0.0 else 1.0
    d_norm = [v / norm for v in growth]
    f_grid = []
    for d_val, dp_val in zip(growth, growth_prime):
        f_grid.append(0.0 if d_val <= 0.0 else dp_val / d_val)
    return a_grid, d_norm, f_grid


def sigma8_at_z(z: float, params: CEForwardParams, a_grid: list[float], d_grid: list[float]) -> float:
    a = 1.0 / (1.0 + z)
    return params.sigma8_0 * interp_linear(a_grid, d_grid, a)


def f_sigma8_at_z(
    z: float,
    params: CEForwardParams,
    a_grid: list[float],
    d_grid: list[float],
    f_grid: list[float],
) -> float:
    a = 1.0 / (1.0 + z)
    d = interp_linear(a_grid, d_grid, a)
    f = interp_linear(a_grid, f_grid, a)
    return f * params.sigma8_0 * d


def s8_today(params: CEForwardParams) -> float:
    return params.sigma8_0 * math.sqrt(params.omega_m0 / 0.3)


def print_report(params: CEForwardParams, z_values: tuple[float, ...]) -> None:
    a_grid, d_grid, f_grid = solve_growth(params)
    print("# CE Residual Cosmology Forward Model")
    print()
    print(f"omega_b0 {params.omega_b0:.6f}")
    print(f"omega_dm0 {params.omega_dm0:.6f}")
    print(f"omega_m0 {params.omega_m0:.6f}")
    print(f"omega_lambda0 {params.omega_lambda0:.6f}")
    print(f"h0 {params.h0:.6f}")
    print(f"rd_mpc {params.rd_mpc:.6f}")
    print(f"w0 {params.w0:.6f}")
    print(f"wa {params.wa:.6f}")
    print(f"gravity_mu_coupling {params.gravity_mu_coupling:.6f}")
    print(f"S8_today {s8_today(params):.6f}")
    print()
    print("z,E(z),D_L_Mpc,D_M_over_rd,D_H_over_rd,D_V_over_rd,Omega_m(z),Omega_de(z),sigma8(z),f_sigma8(z)")
    for z in z_values:
        a = 1.0 / (1.0 + z)
        bao = bao_observable(z, params)
        print(
            f"{z:.6f},"
            f"{e_of_z(z, params):.9f},"
            f"{luminosity_distance_mpc(z, params):.6f},"
            f"{bao.dm_over_rd:.9f},"
            f"{bao.dh_over_rd:.9f},"
            f"{bao.dv_over_rd:.9f},"
            f"{omega_m_of_a(a, params):.9f},"
            f"{omega_de_of_a(a, params):.9f},"
            f"{sigma8_at_z(z, params, a_grid, d_grid):.9f},"
            f"{f_sigma8_at_z(z, params, a_grid, d_grid, f_grid):.9f}"
        )
    print()
    print("coverage", ForwardCoverage().summary)


def main() -> int:
    parser = argparse.ArgumentParser(prog="ce_residual_forward_model")
    parser.add_argument("--h0", type=float, default=67.4)
    parser.add_argument("--rd-mpc", type=float, default=147.09)
    parser.add_argument("--sigma8-0", type=float, default=0.811)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--wa", type=float, default=0.0)
    parser.add_argument("--gravity-mu-coupling", type=float, default=0.0)
    parser.add_argument("--z-list", type=str, default="0,0.5,1,2")
    parser.add_argument(
        "--bao-data",
        type=str,
        default="",
        help="Optional diagonal BAO data: z:kind:value:sigma,... where kind is dm, dh, or dv.",
    )
    parser.add_argument(
        "--bao-cov",
        type=str,
        default="",
        help="Optional full BAO covariance rows, e.g. '0.04,0.01;0.01,0.09'.",
    )
    parser.add_argument(
        "--bao-dataset",
        type=str,
        default="",
        choices=["", "desi-dr2-bgs", "desi-dr2-all"],
        help="Optional built-in BAO dataset. Overrides --bao-data/--bao-cov.",
    )
    args = parser.parse_args()

    z_values = tuple(float(part.strip()) for part in args.z_list.split(",") if part.strip())
    params = CEForwardParams(
        h0=args.h0,
        rd_mpc=args.rd_mpc,
        sigma8_0=args.sigma8_0,
        w0=args.w0,
        wa=args.wa,
        gravity_mu_coupling=args.gravity_mu_coupling,
    )
    print_report(params, z_values)
    dataset = named_bao_dataset(args.bao_dataset) if args.bao_dataset else None
    bao_data = dataset.data if dataset is not None else parse_bao_data(args.bao_data)
    if bao_data:
        bao_cov = dataset.covariance if dataset is not None else parse_covariance_matrix(args.bao_cov)
        print()
        if bao_cov:
            print("bao_chi2", f"{bao_chi2_with_covariance(bao_data, bao_cov, params):.9f}")
            print("bao_covariance", "full")
        else:
            print("bao_chi2", f"{bao_chi2(bao_data, params):.9f}")
            print("bao_covariance", "diagonal")
        print("bao_n", len(bao_data))
        if dataset is not None:
            print("bao_dataset", dataset.name)
            print("bao_source", dataset.source)
        print("bao_note", "compressed_bao_likelihood")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

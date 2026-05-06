from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SHORT_WINDOW = 21
MED_WINDOW = 63
LONG_WINDOW = 126
HORIZON = 21

CE = {
    "epsilon2": 0.04864672,
    "rho_contract": 0.15458752,
    "sigma": 0.95135328,
    "d_eff": 3.177359,
    "alpha_s": 0.116820,
}

MODEL_TYPES = {
    "equation_residual": "direct equation",
    "multiscale_bridge": "multi-scale",
    "self_recursive": "self-recursion",
    "entropy_selection": "entropy/selection",
    "downside_cascade": "downside cascade",
    "hybrid_ce": "combined",
}


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Features:
    shock_z: float
    residual_cluster: float
    vol_accel: float
    anchor_gap: float
    anchor_accel: float
    drawdown: float
    downside_ratio: float
    downside_cascade: float
    liq_stress: float
    entropy_shift: float
    recursion_memory: float
    recovery_failure: float
    vol_ratio: float


@dataclass
class AssetResult:
    symbol: str
    model: str
    model_type: str
    rows: int
    period: str
    current_stress: float
    current_selection: float
    current_regime: str
    stress_to_future_vol_spearman: float
    base_vol_to_future_vol_spearman: float
    stress_to_future_dd_spearman: float
    base_vol_to_future_dd_spearman: float
    top_decile_future_vol: float
    bottom_decile_future_vol: float
    top_decile_future_dd: float
    bottom_decile_future_dd: float
    advantage_score: float


def load_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                close = pick_float(row, "Close", "close", "Adj Close")
                high = pick_float(row, "High", "high")
                low = pick_float(row, "Low", "low")
                open_ = pick_float(row, "Open", "open")
                volume = pick_float(row, "Volume", "volume")
                date = str(row.get("Date") or row.get("date") or row.get("datetime") or "")
            except (TypeError, ValueError, KeyError):
                continue
            if close > 0 and high > 0 and low > 0:
                bars.append(Bar(date, open_, high, low, close, volume))
    return bars


def pick_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(names[0])


def synthetic_suite() -> list[tuple[str, list[Bar]]]:
    return [
        ("SYN_REGIME_SHIFT", synthetic_bars("regime_shift", 42)),
        ("SYN_SLOW_BLEED", synthetic_bars("slow_bleed", 43)),
        ("SYN_FLASH_CRASH", synthetic_bars("flash_crash", 44)),
        ("SYN_VOL_CYCLE", synthetic_bars("vol_cycle", 45)),
        ("SYN_TREND_BREAK", synthetic_bars("trend_break", 46)),
        ("SYN_LIQUIDITY_GAP", synthetic_bars("liquidity_gap", 47)),
        ("SYN_MEAN_REVERT", synthetic_bars("mean_revert", 48)),
        ("SYN_SMOOTH_BULL", synthetic_bars("smooth_bull", 49)),
    ]


def synthetic_bars(kind: str, seed: int, n: int = 1500) -> list[Bar]:
    price = 100.0
    bars: list[Bar] = []
    prev_ret = 0.0
    for i in range(n):
        u1, seed = next_unit(seed)
        u2, seed = next_unit(seed)
        u3, seed = next_unit(seed)
        z1 = u1 - 0.5
        z2 = u2 - 0.5
        drift = 0.00025
        vol = 0.010
        jump = 0.0
        volume_boost = 1.0

        if kind == "regime_shift" and (330 <= i < 405 or 930 <= i < 995):
            drift, vol, volume_boost = -0.0016, 0.034, 5.5
        elif kind == "slow_bleed" and 420 <= i < 780:
            drift, vol, volume_boost = -0.0009, 0.017, 2.3
        elif kind == "flash_crash" and i in (360, 361, 362, 920):
            jump, vol, volume_boost = -0.075, 0.050, 8.0
        elif kind == "vol_cycle":
            vol = 0.008 + 0.020 * (0.5 + 0.5 * math.sin(i / 36.0))
            volume_boost = 1.0 + 2.5 * max(0.0, math.sin(i / 36.0))
        elif kind == "trend_break":
            drift = 0.0010 if i < 700 else -0.0012
            vol = 0.011 if i < 700 else 0.020
            volume_boost = 1.0 if i < 700 else 2.4
        elif kind == "liquidity_gap" and 520 <= i < 620:
            drift, vol, volume_boost = -0.0004, 0.024, 0.28
        elif kind == "mean_revert":
            drift = -0.10 * prev_ret
            vol = 0.018
        elif kind == "smooth_bull":
            drift, vol, volume_boost = 0.00065, 0.007, 1.0

        ret = drift + vol * z1 + 0.45 * vol * z2 + jump
        prev_ret = ret
        prev = price
        price = max(1.0, price * math.exp(ret))
        spread = 0.004 + abs(ret) * (0.8 + 0.4 * u3)
        high = max(prev, price) * (1.0 + spread)
        low = min(prev, price) * max(0.1, 1.0 - spread)
        volume = 1_000_000.0 * max(0.05, volume_boost) * (1.0 + abs(z2))
        bars.append(Bar(f"{kind.upper()}-{i + 1:04d}", prev, high, low, price, volume))
    return bars


def next_unit(seed: int) -> tuple[float, int]:
    seed = (6364136223846793005 * seed + 1442695040888963407) & ((1 << 64) - 1)
    return (seed >> 11) / float(1 << 53), seed


def analyze_asset(symbol: str, bars: list[Bar]) -> list[AssetResult]:
    if len(bars) < LONG_WINDOW + HORIZON + 5:
        return []

    closes = [b.close for b in bars]
    volumes = [max(1.0, b.volume) for b in bars]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

    features, base_vol, future_vol, future_dd = build_feature_series(closes, volumes, returns)
    if not features:
        return []

    results = []
    for model in MODEL_TYPES:
        stress, selection = score_model(model, features)
        results.append(
            summarize_model(
                symbol=symbol,
                model=model,
                bars=bars,
                stress=stress,
                selection=selection,
                base_vol=base_vol,
                future_vol=future_vol,
                future_dd=future_dd,
            )
        )
    return results


def build_feature_series(
    closes: list[float], volumes: list[float], returns: list[float]
) -> tuple[list[Features], list[float], list[float], list[float]]:
    features: list[Features] = []
    base_vol: list[float] = []
    future_vol: list[float] = []
    future_dd: list[float] = []
    memory_phi = 0.0
    prev_anchor_gap = 0.0
    prev_downside_ratio = 0.0

    for i in range(LONG_WINDOW, len(returns) - HORIZON):
        long_rets = returns[i - LONG_WINDOW + 1 : i + 1]
        med_rets = returns[i - MED_WINDOW + 1 : i + 1]
        short_rets = returns[i - SHORT_WINDOW + 1 : i + 1]
        sigma_long = stdev(long_rets) + 1e-12
        sigma_med = stdev(med_rets) + 1e-12
        sigma_short = stdev(short_rets) + 1e-12
        mu_long = mean(long_rets)
        shock_z = abs(returns[i] - mu_long) / sigma_long
        vol_ratio = sigma_short / sigma_long
        vol_accel = max(0.0, sigma_short / sigma_med - 1.0) + max(0.0, sigma_med / sigma_long - 1.0)
        cur_close = closes[i + 1]
        recent_peak = max(closes[max(0, i + 1 - LONG_WINDOW) : i + 2])
        drawdown = max(0.0, 1.0 - cur_close / max(recent_peak, 1e-12))
        volume_ratio = volumes[i + 1] / (mean(volumes[max(0, i + 1 - LONG_WINDOW) : i + 2]) + 1e-12)
        liq_stress = max(0.0, 1.0 / max(volume_ratio, 1e-6) - 1.0)
        short_energy = mean([min(9.0, ((r - mu_long) / sigma_long) ** 2) for r in short_rets])
        long_energy = mean([min(9.0, ((r - mu_long) / sigma_long) ** 2) for r in long_rets])
        residual_cluster = max(0.0, short_energy / (long_energy + 1e-12) - 1.0)
        downside = stdev([min(0.0, r - mu_long) for r in short_rets])
        downside_ratio = max(0.0, downside / sigma_short - 0.55)
        downside_cascade = max(0.0, downside_ratio - prev_downside_ratio) + max(0.0, drawdown - 0.04)
        prev_downside_ratio = downside_ratio
        anchor = mean(closes[max(0, i + 1 - MED_WINDOW) : i + 2])
        anchor_gap = abs(math.log(cur_close / max(anchor, 1e-12))) / (sigma_long * math.sqrt(MED_WINDOW))
        anchor_accel = max(0.0, anchor_gap - prev_anchor_gap)
        prev_anchor_gap = anchor_gap
        entropy_shift = abs(sign_entropy(short_rets) - sign_entropy(long_rets))

        instant_phi = (
            0.18 * saturate(shock_z, 2.5)
            + 0.22 * saturate(residual_cluster, 1.0)
            + 0.20 * saturate(vol_accel, 0.75)
            + 0.15 * saturate(anchor_gap, 1.0)
            + 0.10 * saturate(anchor_accel, 0.35)
            + 0.08 * saturate(drawdown, 0.18)
            + 0.05 * saturate(downside_ratio, 0.35)
            + 0.02 * saturate(liq_stress, 1.0)
        )
        memory_phi = 0.90 * memory_phi + 0.10 * instant_phi
        recovery_failure = max(0.0, memory_phi - 0.70 * instant_phi)

        fwd_rets = returns[i + 1 : i + 1 + HORIZON]
        fwd_closes = closes[i + 1 : i + 2 + HORIZON]
        features.append(
            Features(
                shock_z=shock_z,
                residual_cluster=residual_cluster,
                vol_accel=vol_accel,
                anchor_gap=anchor_gap,
                anchor_accel=anchor_accel,
                drawdown=drawdown,
                downside_ratio=downside_ratio,
                downside_cascade=downside_cascade,
                liq_stress=liq_stress,
                entropy_shift=entropy_shift,
                recursion_memory=memory_phi,
                recovery_failure=recovery_failure,
                vol_ratio=vol_ratio,
            )
        )
        base_vol.append(sigma_short * math.sqrt(252.0))
        future_vol.append(stdev(fwd_rets) * math.sqrt(252.0))
        future_dd.append(max_drawdown(fwd_closes))
    return features, base_vol, future_vol, future_dd


def score_model(model: str, features: list[Features]) -> tuple[list[float], list[float]]:
    stress: list[float] = []
    selection: list[float] = []
    recursive_state = 0.0
    for f in features:
        if model == "equation_residual":
            phi = (
                0.45 * saturate(f.shock_z, 2.5)
                + 0.25 * saturate(f.residual_cluster, 1.0)
                + 0.20 * saturate(f.vol_ratio - 1.0, 1.25)
                + 0.10 * saturate(f.anchor_gap, 1.0)
            )
        elif model == "multiscale_bridge":
            phi = (
                0.24 * saturate(f.residual_cluster, 1.0)
                + 0.26 * saturate(f.vol_accel, 0.75)
                + 0.24 * saturate(f.anchor_gap, 1.0)
                + 0.16 * saturate(f.anchor_accel, 0.35)
                + 0.10 * saturate(f.drawdown, 0.18)
            )
        elif model == "self_recursive":
            raw = (
                0.32 * saturate(f.recursion_memory, 0.22)
                + 0.25 * saturate(f.recovery_failure, 0.10)
                + 0.20 * saturate(f.residual_cluster, 1.0)
                + 0.15 * saturate(f.anchor_gap, 1.0)
                + 0.08 * saturate(f.vol_accel, 0.75)
            )
            recursive_state = 0.82 * recursive_state + 0.18 * raw
            phi = 0.55 * raw + 0.45 * recursive_state
        elif model == "entropy_selection":
            phi = (
                0.36 * saturate(f.entropy_shift, 0.25)
                + 0.24 * saturate(f.liq_stress, 1.0)
                + 0.20 * saturate(f.vol_accel, 0.75)
                + 0.12 * saturate(f.residual_cluster, 1.0)
                + 0.08 * saturate(f.shock_z, 2.5)
            )
        elif model == "downside_cascade":
            phi = (
                0.34 * saturate(f.downside_ratio, 0.35)
                + 0.28 * saturate(f.downside_cascade, 0.16)
                + 0.20 * saturate(f.drawdown, 0.18)
                + 0.12 * saturate(f.anchor_accel, 0.35)
                + 0.06 * saturate(f.liq_stress, 1.0)
            )
        elif model == "hybrid_ce":
            raw = (
                0.16 * saturate(f.shock_z, 2.5)
                + 0.18 * saturate(f.residual_cluster, 1.0)
                + 0.18 * saturate(f.vol_accel, 0.75)
                + 0.14 * saturate(f.anchor_gap, 1.0)
                + 0.12 * saturate(f.recovery_failure, 0.10)
                + 0.10 * saturate(f.downside_cascade, 0.16)
                + 0.08 * saturate(f.entropy_shift, 0.25)
                + 0.04 * saturate(f.liq_stress, 1.0)
            )
            recursive_state = 0.88 * recursive_state + 0.12 * raw
            phi = 0.72 * raw + 0.28 * recursive_state
        else:
            raise ValueError(model)

        sel = min(1.0, max(CE["epsilon2"], math.exp(-CE["d_eff"] * phi)))
        selection.append(sel)
        stress.append(100.0 * (1.0 - sel) / (1.0 - CE["epsilon2"]))
    return stress, selection


def summarize_model(
    symbol: str,
    model: str,
    bars: list[Bar],
    stress: list[float],
    selection: list[float],
    base_vol: list[float],
    future_vol: list[float],
    future_dd: list[float],
) -> AssetResult:
    sv = spearman(stress, future_vol)
    bv = spearman(base_vol, future_vol)
    sd = spearman(stress, future_dd)
    bd = spearman(base_vol, future_dd)
    top_vol, bottom_vol = top_bottom_means(stress, future_vol)
    top_dd, bottom_dd = top_bottom_means(stress, future_dd)
    advantage = 0.5 * ((sv - bv) + (sd - bd))
    return AssetResult(
        symbol=symbol,
        model=model,
        model_type=MODEL_TYPES[model],
        rows=len(bars),
        period=f"{bars[0].date} ~ {bars[-1].date}",
        current_stress=stress[-1],
        current_selection=selection[-1],
        current_regime=regime_name(stress[-1], selection[-1]),
        stress_to_future_vol_spearman=sv,
        base_vol_to_future_vol_spearman=bv,
        stress_to_future_dd_spearman=sd,
        base_vol_to_future_dd_spearman=bd,
        top_decile_future_vol=top_vol,
        bottom_decile_future_vol=bottom_vol,
        top_decile_future_dd=top_dd,
        bottom_decile_future_dd=bottom_dd,
        advantage_score=advantage,
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def saturate(x: float, scale: float) -> float:
    x = max(0.0, x)
    return x / (x + scale + 1e-12)


def sign_entropy(xs: list[float]) -> float:
    if not xs:
        return 0.0
    p_pos = sum(1 for x in xs if x >= 0.0) / len(xs)
    p_neg = 1.0 - p_pos
    h = 0.0
    for p in (p_pos, p_neg):
        if p > 1e-12:
            h -= p * math.log(p, 2)
    return h


def max_drawdown(prices: list[float]) -> float:
    peak = prices[0]
    dd = 0.0
    for price in prices:
        peak = max(peak, price)
        dd = max(dd, 1.0 - price / max(peak, 1e-12))
    return dd


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def top_bottom_means(keys: list[float], values: list[float]) -> tuple[float, float]:
    n = max(1, len(keys) // 10)
    pairs = sorted(zip(keys, values), key=lambda x: x[0])
    bottom = mean([v for _, v in pairs[:n]])
    top = mean([v for _, v in pairs[-n:]])
    return top, bottom


def regime_name(stress: float, selection: float) -> str:
    if selection < CE["rho_contract"] or stress >= 75.0:
        return "transition-risk"
    if stress >= 55.0:
        return "unstable"
    if stress >= 35.0:
        return "watch"
    return "calm"


def discover_inputs() -> list[tuple[str, list[Bar]]]:
    out: list[tuple[str, list[Bar]]] = []
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.csv")):
            bars = load_csv(path)
            if bars:
                out.append((path.stem, bars))
    out.extend(synthetic_suite())
    return out


def write_outputs(results: list[AssetResult]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "windows": {
            "short": SHORT_WINDOW,
            "medium": MED_WINDOW,
            "long": LONG_WINDOW,
            "horizon": HORIZON,
        },
        "ce_constants": CE,
        "model_types": MODEL_TYPES,
        "results": [r.__dict__ for r in results],
        "interpretation": {
            "stress": "0 calm, 100 high residual/selection failure.",
            "advantage_score": "Positive means CE stress beats simple short realized volatility on rank correlation.",
        },
    }
    (RESULTS_DIR / "ce_market_residual_index.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# CE Market Residual Index Gate",
        "",
        "Purpose: compare CE residual models across market regimes.",
        "",
        f"Windows: short={SHORT_WINDOW}, medium={MED_WINDOW}, long={LONG_WINDOW}, forward={HORIZON} trading days.",
        "",
        "## Best Model By Case",
        "",
        "| Asset/case | Best model | Type | Advantage | rho(vol) CE/base | rho(dd) CE/base | top/bottom dd |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for symbol in sorted({r.symbol for r in results}):
        case_rows = [r for r in results if r.symbol == symbol]
        best = max(case_rows, key=lambda r: r.advantage_score)
        lines.append(
            f"| {best.symbol} | {best.model} | {best.model_type} | {best.advantage_score:+.3f} | "
            f"{best.stress_to_future_vol_spearman:.3f}/{best.base_vol_to_future_vol_spearman:.3f} | "
            f"{best.stress_to_future_dd_spearman:.3f}/{best.base_vol_to_future_dd_spearman:.3f} | "
            f"{best.top_decile_future_dd:.3f}/{best.bottom_decile_future_dd:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Full Model Matrix",
            "",
            "| Asset/case | Model | Type | Current stress | Regime | Sel. | rho(vol) CE/base | rho(dd) CE/base | top/bottom vol | top/bottom dd | Advantage |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in sorted(results, key=lambda x: (x.symbol, -x.advantage_score, x.model)):
        lines.append(
            f"| {r.symbol} | {r.model} | {r.model_type} | {r.current_stress:.2f} | {r.current_regime} | "
            f"{r.current_selection:.4f} | {r.stress_to_future_vol_spearman:.3f}/{r.base_vol_to_future_vol_spearman:.3f} | "
            f"{r.stress_to_future_dd_spearman:.3f}/{r.base_vol_to_future_dd_spearman:.3f} | "
            f"{r.top_decile_future_vol:.3f}/{r.bottom_decile_future_vol:.3f} | "
            f"{r.top_decile_future_dd:.3f}/{r.bottom_decile_future_dd:.3f} | {r.advantage_score:+.3f} |"
        )

    lines.extend(
        [
            "",
            "Reading:",
            "- equation_residual tests the direct residual equation.",
            "- self_recursive tests whether stress memory and failed recovery matter.",
            "- entropy_selection tests selection failure through sign entropy and liquidity stress.",
            "- downside_cascade tests drawdown-specific cascade risk.",
            "- hybrid_ce combines the above with a recursive memory state.",
            "- This is a risk/regime gate, not a trade recommendation system.",
        ]
    )
    (RESULTS_DIR / "ce_market_residual_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    results: list[AssetResult] = []
    for symbol, bars in discover_inputs():
        results.extend(analyze_asset(symbol, bars))
    if not results:
        raise SystemExit("No analyzable data. Add CSV files to quant/data.")
    write_outputs(results)
    for symbol in sorted({r.symbol for r in results}):
        case_rows = [r for r in results if r.symbol == symbol]
        best = max(case_rows, key=lambda r: r.advantage_score)
        print(
            f"{symbol}: best={best.model} ({best.model_type}), "
            f"adv={best.advantage_score:+.3f}, "
            f"rho_vol={best.stress_to_future_vol_spearman:.3f}/{best.base_vol_to_future_vol_spearman:.3f}, "
            f"rho_dd={best.stress_to_future_dd_spearman:.3f}/{best.base_vol_to_future_dd_spearman:.3f}"
        )
    print(f"wrote {RESULTS_DIR / 'ce_market_residual_report.md'}")


if __name__ == "__main__":
    main()

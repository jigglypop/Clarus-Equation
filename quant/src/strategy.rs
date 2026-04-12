use crate::data::Bar;
use crate::market::{ChaosParams, GeometryState, MarketGeometry};

const INITIAL_CAPITAL: f64 = 10_000.0;
const FEE_RATE: f64 = 0.0005;
const SLIPPAGE_RATE: f64 = 0.0005;
const WHIPSAW_DAYS: usize = 5;
const AGGRESSIVE_DD_CAP_PCT: f64 = 35.0;

#[derive(Clone, Debug)]
pub struct Trade {
    pub entry_date: String,
    pub exit_date: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_pct: f64,
    pub holding_days: usize,
    pub reason: &'static str,
}

#[derive(Clone, Debug)]
pub struct StrategyResult {
    pub name: String,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub cagr: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub total_return: f64,
    pub win_rate: f64,
    pub whipsaw_rate: f64,
    pub avg_holding_days: f64,
    pub turnover: f64,
    pub n_trades: usize,
    pub utility: f64,
    pub traded_notional: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WalkForwardConfig {
    pub train_bars: usize,
    pub test_bars: usize,
    pub step_bars: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            train_bars: 504,
            test_bars: 126,
            step_bars: 126,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WalkForwardWindow {
    pub train_start: String,
    pub train_end: String,
    pub test_start: String,
    pub test_end: String,
    pub selection: String,
    pub train_score: f64,
    pub train_utility: f64,
    pub test_return: f64,
    pub test_sharpe: f64,
    pub test_max_drawdown: f64,
    pub test_whipsaw_rate: f64,
    pub test_turnover: f64,
    pub test_trades: usize,
}

#[derive(Clone, Debug)]
pub struct WalkForwardReport {
    pub name: String,
    pub aggregate: StrategyResult,
    pub avg_train_score: f64,
    pub windows: Vec<WalkForwardWindow>,
}

#[derive(Clone, Debug)]
pub struct WalkForwardSuite {
    pub config: WalkForwardConfig,
    pub benchmark: StrategyResult,
    pub reports: Vec<WalkForwardReport>,
}

#[derive(Clone, Copy, Debug)]
struct RiskMetrics {
    sharpe: f64,
    sortino: f64,
    calmar: f64,
    max_drawdown: f64,
}

#[derive(Clone, Copy, Debug)]
struct Fill {
    price: f64,
    notional: f64,
    qty: f64,
}

#[derive(Clone, Debug)]
struct OpenTrade {
    entry_price: f64,
    entry_date: String,
    holding_days: usize,
}

#[derive(Clone, Copy, Debug)]
struct WindowBoundary {
    train_start: usize,
    train_end: usize,
    test_start: usize,
    test_end: usize,
}

#[derive(Clone, Debug)]
struct SelectedCandidate<P> {
    params: P,
    label: &'static str,
    score: f64,
    train_utility: f64,
}

#[derive(Clone, Copy, Debug)]
struct TrendParams {
    label: &'static str,
    window: usize,
    alpha: f64,
    theta: f64,
    smooth_beta: f64,
    anchor_beta: f64,
    trend_entry: f64,
    tau_entry: f64,
    phi_entry: f64,
    anchor_converge: f64,
    soft_cut: f64,
    hard_cut: f64,
    trend_exit: f64,
    tau_exit: f64,
    drawdown_cap: f64,
    stop_loss: f64,
    take_profit: f64,
    scale: f64,
    min_alloc: f64,
    max_alloc: f64,
}

impl TrendParams {
    fn base() -> Self {
        Self {
            label: "core",
            window: 64,
            alpha: 4.8,
            theta: 2.25,
            smooth_beta: 0.22,
            anchor_beta: 0.08,
            trend_entry: 0.06,
            tau_entry: 0.30,
            phi_entry: 0.04,
            anchor_converge: 0.15,
            soft_cut: 0.18,
            hard_cut: 0.07,
            trend_exit: -0.18,
            tau_exit: 0.12,
            drawdown_cap: 0.30,
            stop_loss: 0.85,
            take_profit: 1.20,
            scale: 1.55,
            min_alloc: 0.18,
            max_alloc: 0.95,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MeanParams {
    label: &'static str,
    window: usize,
    alpha: f64,
    theta: f64,
    smooth_beta: f64,
    anchor_beta: f64,
    dislocation_entry: f64,
    shock_entry: f64,
    curvature_floor: f64,
    selection_entry: f64,
    tau_entry: f64,
    phi_entry: f64,
    soft_cut: f64,
    hard_cut: f64,
    stop_loss: f64,
    take_profit: f64,
    mean_exit: f64,
    relapse_exit: f64,
    hold_cap: usize,
    cooldown: usize,
    scale: f64,
    min_alloc: f64,
    max_alloc: f64,
}

impl MeanParams {
    fn base() -> Self {
        Self {
            label: "core",
            window: 48,
            alpha: 5.2,
            theta: 2.10,
            smooth_beta: 0.26,
            anchor_beta: 0.10,
            dislocation_entry: -0.020,
            shock_entry: -0.70,
            curvature_floor: 0.00055,
            selection_entry: 0.005,
            tau_entry: 0.26,
            phi_entry: 0.05,
            soft_cut: 0.14,
            hard_cut: 0.05,
            stop_loss: 0.86,
            take_profit: 1.10,
            mean_exit: -0.002,
            relapse_exit: -0.04,
            hold_cap: 18,
            cooldown: 4,
            scale: 2.20,
            min_alloc: 0.10,
            max_alloc: 0.55,
        }
    }
}

pub fn stability_trend(bars: &[Bar]) -> StrategyResult {
    run_stability_trend("ClarusContinuation", bars, TrendParams::base())
}

pub fn stability_mean_revert(bars: &[Bar]) -> StrategyResult {
    run_stability_mean_revert("ClarusTransition", bars, MeanParams::base())
}

pub fn buy_and_hold(bars: &[Bar]) -> StrategyResult {
    buy_and_hold_named("BuyAndHold", bars)
}

pub fn walk_forward_suite(bars: &[Bar], config: WalkForwardConfig) -> WalkForwardSuite {
    let boundaries = walkforward_boundaries(bars.len(), config);
    let benchmark_segments: Vec<StrategyResult> = boundaries
        .iter()
        .map(|boundary| buy_and_hold_named("WF BuyAndHold", &bars[boundary.test_start..boundary.test_end]))
        .collect();
    let benchmark = aggregate_segments("WF BuyAndHold", &benchmark_segments);

    let trend_candidates = trend_candidates();
    let mean_candidates = mean_candidates();
    let reports = vec![
        build_walkforward_report(
            "WF ClarusContinuation",
            bars,
            &boundaries,
            &trend_candidates,
            run_trend_candidate,
            trend_label,
        ),
        build_walkforward_report(
            "WF ClarusTransition",
            bars,
            &boundaries,
            &mean_candidates,
            run_mean_candidate,
            mean_label,
        ),
    ];

    WalkForwardSuite {
        config,
        benchmark,
        reports,
    }
}

fn build_walkforward_report<P: Copy>(
    name: &str,
    bars: &[Bar],
    boundaries: &[WindowBoundary],
    candidates: &[P],
    runner: fn(&[Bar], P) -> StrategyResult,
    labeler: fn(P) -> &'static str,
) -> WalkForwardReport {
    let mut windows = Vec::new();
    let mut test_segments = Vec::new();
    let mut train_score_sum = 0.0;

    for boundary in boundaries {
        let train_bars = &bars[boundary.train_start..boundary.train_end];
        let test_bars = &bars[boundary.test_start..boundary.test_end];
        let selected = select_best_candidate(train_bars, candidates, runner, labeler);
        let test_result = runner(test_bars, selected.params);
        train_score_sum += selected.score;

        windows.push(WalkForwardWindow {
            train_start: bars[boundary.train_start].date.clone(),
            train_end: bars[boundary.train_end - 1].date.clone(),
            test_start: bars[boundary.test_start].date.clone(),
            test_end: bars[boundary.test_end - 1].date.clone(),
            selection: selected.label.to_string(),
            train_score: selected.score,
            train_utility: selected.train_utility,
            test_return: test_result.total_return,
            test_sharpe: test_result.sharpe,
            test_max_drawdown: test_result.max_drawdown,
            test_whipsaw_rate: test_result.whipsaw_rate,
            test_turnover: test_result.turnover,
            test_trades: test_result.n_trades,
        });
        test_segments.push(test_result);
    }

    let aggregate = aggregate_segments(name, &test_segments);
    let avg_train_score = if windows.is_empty() {
        0.0
    } else {
        train_score_sum / windows.len() as f64
    };

    WalkForwardReport {
        name: name.to_string(),
        aggregate,
        avg_train_score,
        windows,
    }
}

fn select_best_candidate<P: Copy>(
    train_bars: &[Bar],
    candidates: &[P],
    runner: fn(&[Bar], P) -> StrategyResult,
    labeler: fn(P) -> &'static str,
) -> SelectedCandidate<P> {
    let default = candidates[0];
    let mut best = SelectedCandidate {
        params: default,
        label: labeler(default),
        score: f64::NEG_INFINITY,
        train_utility: f64::NEG_INFINITY,
    };

    for &candidate in candidates {
        let train_result = runner(train_bars, candidate);
        let score = walkforward_score(&train_result);
        if score > best.score {
            best = SelectedCandidate {
                params: candidate,
                label: labeler(candidate),
                score,
                train_utility: train_result.utility,
            };
        }
    }

    best
}

fn walkforward_score(result: &StrategyResult) -> f64 {
    let inactivity_penalty = match result.n_trades {
        0 => 2.0,
        1 => 0.50,
        _ => 0.0,
    };
    let dd_penalty = if result.max_drawdown > AGGRESSIVE_DD_CAP_PCT {
        (result.max_drawdown - AGGRESSIVE_DD_CAP_PCT) / 8.0
    } else {
        0.0
    };
    let turnover_penalty = (result.turnover / 30.0).min(1.0) * 0.20;
    let whipsaw_penalty = (result.whipsaw_rate / 100.0) * 0.25;

    1.80 * result.cagr
        + result.total_return / 35.0
        + 0.35 * result.sharpe
        + 0.20 * result.calmar
        + 0.20 * result.utility
        - inactivity_penalty
        - dd_penalty
        - turnover_penalty
        - whipsaw_penalty
}

fn trend_candidates() -> [TrendParams; 7] {
    [
        TrendParams {
            label: "etfCarry",
            window: 72,
            alpha: 4.5,
            theta: 2.45,
            smooth_beta: 0.18,
            anchor_beta: 0.06,
            trend_entry: 0.05,
            tau_entry: 0.34,
            phi_entry: 0.05,
            anchor_converge: 0.18,
            soft_cut: 0.20,
            hard_cut: 0.08,
            trend_exit: -0.20,
            tau_exit: 0.14,
            drawdown_cap: 0.24,
            stop_loss: 0.88,
            take_profit: 1.18,
            scale: 1.40,
            min_alloc: 0.20,
            max_alloc: 0.85,
        },
        TrendParams {
            label: "etfBalance",
            window: 60,
            alpha: 4.8,
            theta: 2.30,
            smooth_beta: 0.20,
            anchor_beta: 0.08,
            trend_entry: 0.07,
            tau_entry: 0.28,
            phi_entry: 0.06,
            anchor_converge: 0.20,
            soft_cut: 0.18,
            hard_cut: 0.07,
            trend_exit: -0.16,
            tau_exit: 0.10,
            drawdown_cap: 0.28,
            stop_loss: 0.86,
            take_profit: 1.20,
            scale: 1.55,
            min_alloc: 0.15,
            max_alloc: 0.90,
        },
        TrendParams {
            label: "singlePulse",
            window: 56,
            alpha: 5.0,
            theta: 2.10,
            smooth_beta: 0.24,
            anchor_beta: 0.10,
            trend_entry: 0.05,
            tau_entry: 0.24,
            phi_entry: 0.08,
            anchor_converge: 0.24,
            soft_cut: 0.14,
            hard_cut: 0.05,
            trend_exit: -0.14,
            tau_exit: 0.08,
            drawdown_cap: 0.34,
            stop_loss: 0.82,
            take_profit: 1.25,
            scale: 1.80,
            min_alloc: 0.12,
            max_alloc: 0.95,
        },
        TrendParams {
            label: "singleAdaptive",
            window: 48,
            alpha: 5.2,
            theta: 2.00,
            smooth_beta: 0.28,
            anchor_beta: 0.10,
            trend_entry: 0.04,
            tau_entry: 0.20,
            phi_entry: 0.08,
            anchor_converge: 0.28,
            soft_cut: 0.12,
            hard_cut: 0.04,
            trend_exit: -0.12,
            tau_exit: 0.06,
            drawdown_cap: 0.38,
            stop_loss: 0.80,
            take_profit: 1.28,
            scale: 1.90,
            min_alloc: 0.10,
            max_alloc: 0.98,
        },
        TrendParams {
            label: "highVolBreak",
            window: 36,
            alpha: 4.0,
            theta: 1.75,
            smooth_beta: 0.34,
            anchor_beta: 0.14,
            trend_entry: 0.03,
            tau_entry: 0.16,
            phi_entry: 0.10,
            anchor_converge: 0.35,
            soft_cut: 0.10,
            hard_cut: 0.03,
            trend_exit: -0.10,
            tau_exit: 0.04,
            drawdown_cap: 0.45,
            stop_loss: 0.76,
            take_profit: 1.35,
            scale: 2.10,
            min_alloc: 0.10,
            max_alloc: 0.95,
        },
        TrendParams {
            label: "highVolCarry",
            window: 44,
            alpha: 4.2,
            theta: 1.90,
            smooth_beta: 0.30,
            anchor_beta: 0.12,
            trend_entry: 0.02,
            tau_entry: 0.18,
            phi_entry: 0.12,
            anchor_converge: 0.30,
            soft_cut: 0.10,
            hard_cut: 0.03,
            trend_exit: -0.08,
            tau_exit: 0.05,
            drawdown_cap: 0.50,
            stop_loss: 0.74,
            take_profit: 1.40,
            scale: 2.30,
            min_alloc: 0.08,
            max_alloc: 1.00,
        },
        TrendParams::base(),
    ]
}

fn mean_candidates() -> [MeanParams; 7] {
    [
        MeanParams {
            label: "etfTransition",
            window: 52,
            alpha: 4.8,
            theta: 2.30,
            smooth_beta: 0.20,
            anchor_beta: 0.08,
            dislocation_entry: -0.018,
            shock_entry: -0.55,
            curvature_floor: 0.00045,
            selection_entry: 0.0,
            tau_entry: 0.32,
            phi_entry: 0.05,
            soft_cut: 0.16,
            hard_cut: 0.07,
            stop_loss: 0.89,
            take_profit: 1.08,
            mean_exit: 0.002,
            relapse_exit: -0.05,
            hold_cap: 16,
            cooldown: 4,
            scale: 1.80,
            min_alloc: 0.08,
            max_alloc: 0.40,
        },
        MeanParams {
            label: "etfDeep",
            window: 60,
            alpha: 5.0,
            theta: 2.25,
            smooth_beta: 0.22,
            anchor_beta: 0.09,
            dislocation_entry: -0.025,
            shock_entry: -0.80,
            curvature_floor: 0.00055,
            selection_entry: 0.005,
            tau_entry: 0.30,
            phi_entry: 0.04,
            soft_cut: 0.14,
            hard_cut: 0.06,
            stop_loss: 0.87,
            take_profit: 1.10,
            mean_exit: 0.0,
            relapse_exit: -0.04,
            hold_cap: 18,
            cooldown: 4,
            scale: 2.00,
            min_alloc: 0.08,
            max_alloc: 0.50,
        },
        MeanParams {
            label: "singleSnap",
            window: 44,
            alpha: 5.4,
            theta: 2.00,
            smooth_beta: 0.28,
            anchor_beta: 0.12,
            dislocation_entry: -0.025,
            shock_entry: -0.85,
            curvature_floor: 0.00070,
            selection_entry: 0.0,
            tau_entry: 0.24,
            phi_entry: 0.06,
            soft_cut: 0.12,
            hard_cut: 0.05,
            stop_loss: 0.84,
            take_profit: 1.12,
            mean_exit: 0.002,
            relapse_exit: -0.06,
            hold_cap: 14,
            cooldown: 3,
            scale: 2.60,
            min_alloc: 0.10,
            max_alloc: 0.60,
        },
        MeanParams {
            label: "singleReflex",
            window: 36,
            alpha: 5.8,
            theta: 1.85,
            smooth_beta: 0.32,
            anchor_beta: 0.14,
            dislocation_entry: -0.030,
            shock_entry: -1.00,
            curvature_floor: 0.00090,
            selection_entry: -0.005,
            tau_entry: 0.18,
            phi_entry: 0.08,
            soft_cut: 0.10,
            hard_cut: 0.04,
            stop_loss: 0.82,
            take_profit: 1.15,
            mean_exit: 0.005,
            relapse_exit: -0.08,
            hold_cap: 12,
            cooldown: 2,
            scale: 3.00,
            min_alloc: 0.10,
            max_alloc: 0.70,
        },
        MeanParams {
            label: "highVolReentry",
            window: 32,
            alpha: 4.4,
            theta: 1.65,
            smooth_beta: 0.38,
            anchor_beta: 0.16,
            dislocation_entry: -0.035,
            shock_entry: -1.15,
            curvature_floor: 0.00100,
            selection_entry: -0.01,
            tau_entry: 0.14,
            phi_entry: 0.10,
            soft_cut: 0.08,
            hard_cut: 0.03,
            stop_loss: 0.78,
            take_profit: 1.18,
            mean_exit: 0.010,
            relapse_exit: -0.10,
            hold_cap: 10,
            cooldown: 2,
            scale: 3.50,
            min_alloc: 0.10,
            max_alloc: 0.75,
        },
        MeanParams {
            label: "highVolFlush",
            window: 28,
            alpha: 4.2,
            theta: 1.55,
            smooth_beta: 0.42,
            anchor_beta: 0.18,
            dislocation_entry: -0.045,
            shock_entry: -1.25,
            curvature_floor: 0.00120,
            selection_entry: -0.015,
            tau_entry: 0.10,
            phi_entry: 0.12,
            soft_cut: 0.08,
            hard_cut: 0.03,
            stop_loss: 0.76,
            take_profit: 1.22,
            mean_exit: 0.012,
            relapse_exit: -0.12,
            hold_cap: 8,
            cooldown: 1,
            scale: 4.00,
            min_alloc: 0.10,
            max_alloc: 0.85,
        },
        MeanParams::base(),
    ]
}

fn walkforward_boundaries(len: usize, config: WalkForwardConfig) -> Vec<WindowBoundary> {
    let mut boundaries = Vec::new();
    if config.train_bars == 0
        || config.test_bars == 0
        || config.step_bars == 0
        || len < config.train_bars + config.test_bars
    {
        return boundaries;
    }

    let mut test_start = config.train_bars;
    while test_start + config.test_bars <= len {
        boundaries.push(WindowBoundary {
            train_start: test_start - config.train_bars,
            train_end: test_start,
            test_start,
            test_end: test_start + config.test_bars,
        });
        test_start += config.step_bars;
    }
    boundaries
}

fn aggregate_segments(name: &str, segments: &[StrategyResult]) -> StrategyResult {
    if segments.is_empty() {
        return compute_result(name.to_string(), vec![], vec![INITIAL_CAPITAL], INITIAL_CAPITAL, 0.0);
    }

    let mut capital = INITIAL_CAPITAL;
    let mut equity = Vec::new();
    let mut trades = Vec::new();
    let mut traded_notional = 0.0;

    for segment in segments {
        let base = segment
            .equity_curve
            .first()
            .copied()
            .unwrap_or(INITIAL_CAPITAL)
            .max(1e-12);
        let scale = capital / base;

        for (idx, value) in segment.equity_curve.iter().enumerate() {
            let scaled = capital * (*value / base);
            if !equity.is_empty() && idx == 0 {
                continue;
            }
            equity.push(scaled);
        }

        traded_notional += segment.traded_notional * scale;
        trades.extend(segment.trades.clone());
        capital = equity.last().copied().unwrap_or(capital);
    }

    compute_result(name.to_string(), trades, equity, capital, traded_notional)
}

fn trend_label(params: TrendParams) -> &'static str {
    params.label
}

fn mean_label(params: MeanParams) -> &'static str {
    params.label
}

fn run_trend_candidate(bars: &[Bar], params: TrendParams) -> StrategyResult {
    run_stability_trend("ClarusContinuation", bars, params)
}

fn run_mean_candidate(bars: &[Bar], params: MeanParams) -> StrategyResult {
    run_stability_mean_revert("ClarusTransition", bars, params)
}

fn run_stability_trend(name: &str, bars: &[Bar], params: TrendParams) -> StrategyResult {
    let mut geo = configured_market(
        params.window,
        params.alpha,
        params.theta,
        params.smooth_beta,
        params.anchor_beta,
    );
    let mut cash = INITIAL_CAPITAL;
    let mut holdings = 0.0;
    let mut open: Option<OpenTrade> = None;
    let mut trades = Vec::new();
    let mut equity = Vec::with_capacity(bars.len());
    let mut traded_notional = 0.0;

    for bar in bars {
        let st = geo.update(bar.close, bar.volume, bar.high, bar.low);
        if let Some(position) = open.as_mut() {
            position.holding_days += 1;
        }

        let continuation_score = continuation_alpha(&st);
        let press_edge = st.trend_score > params.trend_entry
            && st.tau_proxy > params.tau_entry
            && st.d_phi < params.phi_entry
            && st.d_anchor_gap < params.anchor_converge;
        let can_hold = continuation_score > 0.18
            && st.trend_score > params.trend_exit
            && st.tau_proxy > params.tau_exit;

        let mut target_alloc = 0.0;
        if press_edge || (open.is_some() && can_hold) {
            let selection_scale = 0.25 + 0.75 * st.p_selected.max(params.hard_cut);
            target_alloc = (continuation_score * selection_scale * params.scale)
                .clamp(0.0, params.max_alloc);
            if target_alloc > 0.0 {
                target_alloc = target_alloc.max(params.min_alloc.min(params.max_alloc));
            }
            if st.drawdown > params.drawdown_cap {
                target_alloc *= 0.35;
            } else if st.p_selected < params.soft_cut || st.d_phi > params.phi_entry * 1.5 {
                target_alloc *= 0.45;
            }
        }

        let portfolio_val = portfolio_value(cash, holdings, bar.close);
        equity.push(portfolio_val);

        if let Some(position) = open.as_ref() {
            let pnl = bar.close / position.entry_price;
            let exit_reason = if st.p_selected < params.hard_cut {
                Some("HardChaos")
            } else if st.trend_score < params.trend_exit && st.d_phi > 0.0 {
                Some("TrendBreak")
            } else if st.tau_proxy < params.tau_exit && st.d_phi > 0.0 {
                Some("TransitionLost")
            } else if st.drawdown > params.drawdown_cap + 0.08 && pnl < 0.98 {
                Some("DDCap")
            } else if pnl < params.stop_loss {
                Some("StopLoss")
            } else if pnl > params.take_profit && st.d_phi > 0.0 {
                Some("TakeProfit")
            } else if target_alloc < params.min_alloc * 0.5 {
                Some("NoEdge")
            } else {
                None
            };

            if let Some(reason) = exit_reason {
                traded_notional += close_position(
                    &mut trades,
                    &mut cash,
                    &mut holdings,
                    bar,
                    &mut open,
                    reason,
                );
                continue;
            }
        }

        if target_alloc > 0.0 {
            traded_notional +=
                rebalance_toward(&mut cash, &mut holdings, bar, target_alloc, &mut open);
        }
    }

    if let Some(last) = bars.last() {
        if open.is_some() {
            traded_notional +=
                close_position(&mut trades, &mut cash, &mut holdings, last, &mut open, "EoP");
        }
    }

    compute_result(name.to_string(), trades, equity, cash, traded_notional)
}

fn run_stability_mean_revert(name: &str, bars: &[Bar], params: MeanParams) -> StrategyResult {
    let mut geo = configured_market(
        params.window,
        params.alpha,
        params.theta,
        params.smooth_beta,
        params.anchor_beta,
    );
    let mut cash = INITIAL_CAPITAL;
    let mut holdings = 0.0;
    let mut open: Option<OpenTrade> = None;
    let mut trades = Vec::new();
    let mut equity = Vec::with_capacity(bars.len());
    let mut traded_notional = 0.0;
    let mut cooldown = 0usize;

    for bar in bars {
        let st = geo.update(bar.close, bar.volume, bar.high, bar.low);
        if cooldown > 0 {
            cooldown -= 1;
        }
        if let Some(position) = open.as_mut() {
            position.holding_days += 1;
        }

        let transition_score = transition_alpha(&st);
        let shock_regime = st.price_dislocation < params.dislocation_entry
            && st.shock_z < params.shock_entry
            && st.base_curvature > params.curvature_floor;
        let stabilization = st.selection_recovery > params.selection_entry
            && st.tau_proxy > params.tau_entry
            && st.d_phi < params.phi_entry;

        let mut target_alloc = 0.0;
        if (cooldown == 0 && shock_regime && stabilization)
            || (open.is_some() && transition_score > 0.16 && st.tau_proxy > params.tau_entry * 0.65)
        {
            let selection_scale = 0.25 + 0.75 * st.p_selected.max(params.hard_cut);
            target_alloc = (transition_score * selection_scale * params.scale)
                .clamp(0.0, params.max_alloc);
            if target_alloc > 0.0 {
                target_alloc = target_alloc.max(params.min_alloc.min(params.max_alloc));
            }
            if st.p_selected < params.soft_cut || st.d_phi > params.phi_entry * 1.4 {
                target_alloc *= 0.40;
            }
        }

        let portfolio_val = portfolio_value(cash, holdings, bar.close);
        equity.push(portfolio_val);

        if let Some(position) = open.as_ref() {
            let pnl = bar.close / position.entry_price;
            let exit_reason = if st.p_selected < params.hard_cut {
                Some("HardChaos")
            } else if pnl < params.stop_loss {
                Some("StopLoss")
            } else if pnl > params.take_profit && st.selection_recovery < 0.02 {
                Some("TakeProfit")
            } else if position.holding_days >= params.hold_cap {
                Some("TimeExit")
            } else if position.holding_days >= 2 && st.price_dislocation > params.mean_exit {
                Some("MeanRevert")
            } else if position.holding_days >= 2 && st.selection_recovery < params.relapse_exit {
                Some("Relapse")
            } else if position.holding_days >= 2
                && st.tau_proxy < params.tau_entry * 0.45
                && st.d_phi > 0.0
            {
                Some("TransitionLost")
            } else if position.holding_days >= 2 && target_alloc < params.min_alloc * 0.5 {
                Some("NoEdge")
            } else {
                None
            };

            if let Some(reason) = exit_reason {
                traded_notional += close_position(
                    &mut trades,
                    &mut cash,
                    &mut holdings,
                    bar,
                    &mut open,
                    reason,
                );
                cooldown = params.cooldown;
                continue;
            }
        }

        if target_alloc > 0.0 {
            traded_notional +=
                rebalance_toward(&mut cash, &mut holdings, bar, target_alloc, &mut open);
        }
    }

    if let Some(last) = bars.last() {
        if open.is_some() {
            traded_notional +=
                close_position(&mut trades, &mut cash, &mut holdings, last, &mut open, "EoP");
        }
    }

    compute_result(name.to_string(), trades, equity, cash, traded_notional)
}

fn buy_and_hold_named(name: &str, bars: &[Bar]) -> StrategyResult {
    let initial = INITIAL_CAPITAL;
    if bars.is_empty() {
        return compute_result(name.to_string(), vec![], vec![initial], initial, 0.0);
    }

    let start_price = bars.first().map(|bar| bar.close).unwrap_or(1.0).max(1e-12);
    let holdings = initial / start_price;
    let equity: Vec<f64> = bars.iter().map(|bar| holdings * bar.close).collect();
    let final_val = equity.last().copied().unwrap_or(initial);
    compute_result(name.to_string(), vec![], equity, final_val, 0.0)
}

fn compute_result(
    name: String,
    trades: Vec<Trade>,
    equity: Vec<f64>,
    final_value: f64,
    traded_notional: f64,
) -> StrategyResult {
    let initial = equity.first().copied().unwrap_or(INITIAL_CAPITAL).max(1.0);
    let total_return = (final_value / initial - 1.0) * 100.0;
    let metrics = risk_metrics(&equity);
    let years = (equity.len() as f64 / 252.0).max(1.0 / 252.0);
    let cagr = (final_value / initial).powf(1.0 / years) - 1.0;
    let wins = trades.iter().filter(|trade| trade.pnl_pct > 0.0).count();
    let win_rate = if trades.is_empty() {
        0.0
    } else {
        wins as f64 / trades.len() as f64 * 100.0
    };
    let avg_holding_days = if trades.is_empty() {
        0.0
    } else {
        trades.iter().map(|trade| trade.holding_days as f64).sum::<f64>() / trades.len() as f64
    };
    let whipsaw_count = trades
        .iter()
        .filter(|trade| trade.holding_days <= WHIPSAW_DAYS && trade.pnl_pct < 0.0)
        .count();
    let whipsaw_rate = if trades.is_empty() {
        0.0
    } else {
        whipsaw_count as f64 / trades.len() as f64 * 100.0
    };
    let turnover = traded_notional / initial;
    let dd_pct = metrics.max_drawdown * 100.0;
    let dd_penalty = if dd_pct > AGGRESSIVE_DD_CAP_PCT {
        (dd_pct - AGGRESSIVE_DD_CAP_PCT) / 25.0
    } else {
        dd_pct / 200.0
    };
    let whipsaw_penalty = (whipsaw_rate / 100.0) * 0.10;
    let turnover_penalty = (turnover / 25.0).min(1.0) * 0.05;
    let utility = 0.55 * cagr
        + 0.35 * (total_return / 100.0)
        + 0.15 * metrics.sharpe
        + 0.10 * metrics.calmar
        - dd_penalty
        - whipsaw_penalty
        - turnover_penalty;
    let n_trades = trades.len();

    StrategyResult {
        name,
        trades,
        equity_curve: equity,
        cagr,
        sharpe: metrics.sharpe,
        sortino: metrics.sortino,
        calmar: metrics.calmar,
        max_drawdown: dd_pct,
        total_return,
        win_rate,
        whipsaw_rate,
        avg_holding_days,
        turnover,
        n_trades,
        utility,
        traded_notional,
    }
}

fn configured_market(
    window: usize,
    alpha: f64,
    theta: f64,
    smooth_beta: f64,
    anchor_beta: f64,
) -> MarketGeometry {
    let mut params = ChaosParams::default();
    params.theta = theta;
    params.smooth_beta = smooth_beta;
    params.anchor_beta = anchor_beta;
    MarketGeometry::with_params(window, alpha, params)
}

fn continuation_alpha(state: &GeometryState) -> f64 {
    let trend = state.trend_score.max(0.0).clamp(0.0, 1.0);
    let phi_relief = (-state.d_phi / 0.12).clamp(0.0, 1.0);
    let anchor_converge = (-state.d_anchor_gap / 0.60).clamp(0.0, 1.0);
    let recovery = (state.selection_recovery / 0.08).clamp(0.0, 1.0);
    let tau = state.tau_proxy.clamp(0.0, 1.0);
    let drawdown_relief = (1.0 - state.drawdown / 0.35).clamp(0.0, 1.0);

    (0.30 * trend
        + 0.20 * phi_relief
        + 0.16 * anchor_converge
        + 0.14 * recovery
        + 0.12 * tau
        + 0.08 * drawdown_relief)
        .clamp(0.0, 1.2)
}

fn transition_alpha(state: &GeometryState) -> f64 {
    let dislocation = (-state.price_dislocation / 0.06).clamp(0.0, 1.5);
    let shock = (-state.shock_z / 1.5).clamp(0.0, 1.5);
    let recovery = (state.selection_recovery / 0.06).clamp(0.0, 1.2);
    let phi_turn = (-state.d_phi / 0.12).clamp(0.0, 1.0);
    let tau = state.tau_proxy.clamp(0.0, 1.0);
    let curvature = (state.base_curvature / 0.0010).clamp(0.0, 1.5);
    let anchor = (1.0 - state.anchor_gap / 4.5).clamp(0.0, 1.0);

    (0.20 * dislocation
        + 0.16 * shock
        + 0.18 * recovery
        + 0.16 * phi_turn
        + 0.14 * tau
        + 0.10 * curvature
        + 0.06 * anchor)
        .clamp(0.0, 1.4)
}

fn portfolio_value(cash: f64, holdings: f64, price: f64) -> f64 {
    cash + holdings * price
}

fn rebalance_toward(
    cash: &mut f64,
    holdings: &mut f64,
    bar: &Bar,
    target_alloc: f64,
    open: &mut Option<OpenTrade>,
) -> f64 {
    let portfolio = portfolio_value(*cash, *holdings, bar.close).max(1e-12);
    let current_value = *holdings * bar.close;
    let target_value = (portfolio * target_alloc.clamp(0.0, 1.0)).min(portfolio);
    let diff = target_value - current_value;
    let min_trade = portfolio * 0.02;

    if diff.abs() < min_trade {
        return 0.0;
    }

    if diff > 0.0 {
        let prev_qty = *holdings;
        let fill = buy_notional(cash, holdings, bar.close, diff);
        if fill.qty > 0.0 {
            match open.as_mut() {
                Some(position) => {
                    let total_qty = prev_qty + fill.qty;
                    if total_qty > 0.0 {
                        position.entry_price =
                            (position.entry_price * prev_qty + fill.price * fill.qty) / total_qty;
                    }
                }
                None => {
                    *open = Some(OpenTrade {
                        entry_price: fill.price,
                        entry_date: bar.date.clone(),
                        holding_days: 0,
                    });
                }
            }
        }
        fill.notional
    } else {
        sell_notional(cash, holdings, bar.close, -diff).notional
    }
}

fn close_position(
    trades: &mut Vec<Trade>,
    cash: &mut f64,
    holdings: &mut f64,
    bar: &Bar,
    open: &mut Option<OpenTrade>,
    reason: &'static str,
) -> f64 {
    let Some(position) = open.take() else {
        return 0.0;
    };
    let fill = exit_long(cash, holdings, bar.close);
    push_trade(
        trades,
        &position.entry_date,
        &bar.date,
        position.entry_price,
        fill.price,
        position.holding_days,
        reason,
    );
    fill.notional
}

fn buy_notional(cash: &mut f64, holdings: &mut f64, price: f64, target_notional: f64) -> Fill {
    let invest = target_notional.max(0.0).min(*cash);
    if invest <= 0.0 {
        return Fill {
            price,
            notional: 0.0,
            qty: 0.0,
        };
    }
    let execution_price = price * (1.0 + SLIPPAGE_RATE);
    let qty = invest * (1.0 - FEE_RATE) / execution_price.max(1e-12);
    *cash -= invest;
    *holdings += qty;
    Fill {
        price: execution_price,
        notional: invest,
        qty,
    }
}

fn exit_long(cash: &mut f64, holdings: &mut f64, price: f64) -> Fill {
    if *holdings <= 0.0 {
        return Fill {
            price,
            notional: 0.0,
            qty: 0.0,
        };
    }
    let execution_price = price * (1.0 - SLIPPAGE_RATE);
    let qty = *holdings;
    let proceeds = *holdings * execution_price * (1.0 - FEE_RATE);
    *cash += proceeds;
    *holdings = 0.0;
    Fill {
        price: execution_price,
        notional: proceeds,
        qty,
    }
}

fn sell_notional(cash: &mut f64, holdings: &mut f64, price: f64, target_notional: f64) -> Fill {
    if *holdings <= 0.0 || target_notional <= 0.0 {
        return Fill {
            price,
            notional: 0.0,
            qty: 0.0,
        };
    }
    let execution_price = price * (1.0 - SLIPPAGE_RATE);
    let qty = (target_notional / execution_price.max(1e-12)).min(*holdings);
    let proceeds = qty * execution_price * (1.0 - FEE_RATE);
    *cash += proceeds;
    *holdings -= qty;
    Fill {
        price: execution_price,
        notional: proceeds,
        qty,
    }
}

fn push_trade(
    trades: &mut Vec<Trade>,
    entry_date: &str,
    exit_date: &str,
    entry_price: f64,
    exit_price: f64,
    holding_days: usize,
    reason: &'static str,
) {
    let pnl_pct = if entry_price > 0.0 {
        (exit_price / entry_price - 1.0) * 100.0
    } else {
        0.0
    };
    trades.push(Trade {
        entry_date: entry_date.to_string(),
        exit_date: exit_date.to_string(),
        entry_price,
        exit_price,
        pnl_pct,
        holding_days,
        reason,
    });
}

fn risk_metrics(equity: &[f64]) -> RiskMetrics {
    if equity.len() < 2 {
        return RiskMetrics {
            sharpe: 0.0,
            sortino: 0.0,
            calmar: 0.0,
            max_drawdown: 0.0,
        };
    }

    let mut returns = Vec::with_capacity(equity.len() - 1);
    for window in equity.windows(2) {
        if window[0] > 0.0 {
            returns.push(window[1] / window[0] - 1.0);
        }
    }
    if returns.is_empty() {
        return RiskMetrics {
            sharpe: 0.0,
            sortino: 0.0,
            calmar: 0.0,
            max_drawdown: 0.0,
        };
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = if returns.len() > 1 {
        returns.iter().map(|ret| (ret - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
    } else {
        0.0
    };
    let std = var.sqrt();
    let sharpe = if std > 1e-12 {
        mean / std * (252.0_f64).sqrt()
    } else {
        0.0
    };

    let downside: Vec<f64> = returns.iter().copied().filter(|ret| *ret < 0.0).collect();
    let downside_std = if downside.is_empty() {
        0.0
    } else {
        let dvar = downside.iter().map(|ret| ret.powi(2)).sum::<f64>() / downside.len() as f64;
        dvar.sqrt()
    };
    let sortino = if downside_std > 1e-12 {
        mean / downside_std * (252.0_f64).sqrt()
    } else {
        0.0
    };

    let mut peak = equity[0];
    let mut max_drawdown = 0.0_f64;
    for &value in equity {
        peak = peak.max(value);
        max_drawdown = max_drawdown.max((peak - value) / peak.max(1e-12));
    }

    let total_years = (equity.len() as f64 / 252.0).max(1.0 / 252.0);
    let cagr = (equity.last().copied().unwrap_or(equity[0]) / equity[0])
        .powf(1.0 / total_years)
        - 1.0;
    let calmar = if max_drawdown > 1e-12 {
        cagr / max_drawdown
    } else {
        0.0
    };

    RiskMetrics {
        sharpe,
        sortino,
        calmar,
        max_drawdown,
    }
}

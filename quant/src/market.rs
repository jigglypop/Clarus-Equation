use std::collections::VecDeque;

const STATE_DIM: usize = 6;
type StateVec = [f64; STATE_DIM];

#[derive(Clone, Copy, Debug)]
pub struct GeometryState {
    pub base_curvature: f64,
    pub base_suppression: f64,
    pub chaos_score: f64,
    pub phi_mkt: f64,
    pub d_phi: f64,
    pub p_selected: f64,
    pub selection_recovery: f64,
    pub trend_score: f64,
    pub shock_z: f64,
    pub price_dislocation: f64,
    pub anchor_gap: f64,
    pub d_anchor_gap: f64,
    pub curvature_expansion: f64,
    pub tau_proxy: f64,
    pub drawdown: f64,
    pub relative_liquidity: f64,
    pub parkinson_volatility: f64,
}

impl Default for GeometryState {
    fn default() -> Self {
        Self {
            base_curvature: 0.0,
            base_suppression: 1.0,
            chaos_score: 0.0,
            phi_mkt: 0.0,
            d_phi: 0.0,
            p_selected: 1.0,
            selection_recovery: 0.0,
            trend_score: 0.0,
            shock_z: 0.0,
            price_dislocation: 0.0,
            anchor_gap: 0.0,
            d_anchor_gap: 0.0,
            curvature_expansion: 0.0,
            tau_proxy: 0.0,
            drawdown: 0.0,
            relative_liquidity: 1.0,
            parkinson_volatility: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChaosParams {
    pub lambda_grad: f64,
    pub lambda_lap: f64,
    pub lambda_graph: f64,
    pub lambda_anchor: f64,
    pub lambda_curvature: f64,
    pub theta: f64,
    pub smooth_beta: f64,
    pub anchor_beta: f64,
}

impl Default for ChaosParams {
    fn default() -> Self {
        Self {
            lambda_grad: 0.55,
            lambda_lap: 0.85,
            lambda_graph: 0.35,
            lambda_anchor: 0.45,
            lambda_curvature: 0.30,
            theta: 2.35,
            smooth_beta: 0.20,
            anchor_beta: 0.08,
        }
    }
}

/// Clarus market geometry reinterprets the field as a chaos-suppression potential.
pub struct MarketGeometry {
    window: usize,
    alpha: f64,
    params: ChaosParams,
    closes: VecDeque<f64>,
    volumes: VecDeque<f64>,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    raw_states: VecDeque<StateVec>,
    prev_state: Option<StateVec>,
    prev_delta: Option<StateVec>,
    anchor_state: Option<StateVec>,
    prev_phi: Option<f64>,
    prev_selected: Option<f64>,
    prev_anchor_gap: Option<f64>,
    prev_curvature: Option<f64>,
    smoothed_chaos: f64,
    chaos_initialized: bool,
}

impl MarketGeometry {
    pub fn new(window: usize, alpha: f64) -> Self {
        Self::with_params(window, alpha, ChaosParams::default())
    }

    pub fn with_params(window: usize, alpha: f64, params: ChaosParams) -> Self {
        Self {
            window,
            alpha,
            params,
            closes: VecDeque::with_capacity(window),
            volumes: VecDeque::with_capacity(window),
            highs: VecDeque::with_capacity(window),
            lows: VecDeque::with_capacity(window),
            raw_states: VecDeque::with_capacity(window),
            prev_state: None,
            prev_delta: None,
            anchor_state: None,
            prev_phi: None,
            prev_selected: None,
            prev_anchor_gap: None,
            prev_curvature: None,
            smoothed_chaos: 0.0,
            chaos_initialized: false,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64, high: f64, low: f64) -> GeometryState {
        if self.closes.len() >= self.window {
            self.closes.pop_front();
            self.volumes.pop_front();
            self.highs.pop_front();
            self.lows.pop_front();
            self.raw_states.pop_front();
        }
        let prev_close = self.closes.back().copied().unwrap_or(close);
        self.closes.push_back(close);
        self.volumes.push_back(volume.max(1.0));
        self.highs.push_back(high.max(low + 1e-9));
        self.lows.push_back(low.max(1e-9));

        if self.closes.len() < self.window.max(32) {
            return GeometryState::default();
        }

        let vol = self.parkinson_volatility();
        let rel_liq = self.relative_liquidity();
        let base_curvature = self.alpha * vol.powi(2) / (rel_liq + 0.1);
        let base_suppression = (-100.0 * base_curvature).exp().clamp(1e-12, 1.0);

        let ret = (close / prev_close.max(1e-12)).ln();
        let range_frac = (high / low.max(1e-12)).ln().abs();
        let drawdown = self.drawdown(close);
        let liq_stress = (1.0 / rel_liq.max(1e-6) - 1.0).max(0.0);
        let curvature_penalty = -base_suppression.ln();

        let raw_state = [ret, vol, liq_stress, range_frac, drawdown, curvature_penalty];
        self.raw_states.push_back(raw_state);

        let state = self.normalize_state(raw_state);
        let delta = sub_state(state, self.prev_state.unwrap_or(state));
        let delta2 = sub_state(delta, self.prev_delta.unwrap_or([0.0; STATE_DIM]));
        let anchor_prev = self.anchor_state.unwrap_or(state);
        let anchor_gap = squared_distance(state, anchor_prev);

        let chaos_raw = self.params.lambda_grad * squared_norm(delta)
            + self.params.lambda_lap * squared_norm(delta2)
            + self.params.lambda_graph * laplacian_energy(state)
            + self.params.lambda_anchor * anchor_gap
            + self.params.lambda_curvature * curvature_penalty.powi(2);

        if !self.chaos_initialized {
            self.smoothed_chaos = chaos_raw;
            self.chaos_initialized = true;
        } else {
            self.smoothed_chaos = self.params.smooth_beta * chaos_raw
                + (1.0 - self.params.smooth_beta) * self.smoothed_chaos;
        }

        let phi_mkt = softplus(self.smoothed_chaos - self.params.theta);
        let p_selected = (-phi_mkt).exp().clamp(0.02, 1.0);
        let trend_score = self.trend_score();
        let price_dislocation = self.price_dislocation(close);
        let anchor_gap = anchor_gap.sqrt();
        let prev_phi = self.prev_phi.unwrap_or(phi_mkt);
        let prev_selected = self.prev_selected.unwrap_or(p_selected);
        let prev_anchor_gap = self.prev_anchor_gap.unwrap_or(anchor_gap);
        let prev_curvature = self.prev_curvature.unwrap_or(base_curvature.max(1e-9));
        let d_phi = phi_mkt - prev_phi;
        let selection_recovery = p_selected - prev_selected;
        let d_anchor_gap = anchor_gap - prev_anchor_gap;
        let curvature_expansion =
            (base_curvature.max(1e-9) / prev_curvature.max(1e-9)).ln().clamp(-2.0, 2.0);
        let phi_progress = (-d_phi / 0.15).clamp(0.0, 1.0);
        let anchor_progress = (-d_anchor_gap / 0.75).clamp(0.0, 1.0);
        let curvature_relief = (-curvature_expansion / 0.30).clamp(0.0, 1.0);
        let tau_proxy = (0.45 * p_selected
            + 0.20 * phi_progress
            + 0.20 * anchor_progress
            + 0.15 * curvature_relief)
            .clamp(0.0, 1.0);

        self.prev_state = Some(state);
        self.prev_delta = Some(delta);
        self.anchor_state = Some(ema_state(anchor_prev, state, self.params.anchor_beta));
        self.prev_phi = Some(phi_mkt);
        self.prev_selected = Some(p_selected);
        self.prev_anchor_gap = Some(anchor_gap);
        self.prev_curvature = Some(base_curvature.max(1e-9));

        GeometryState {
            base_curvature,
            base_suppression,
            chaos_score: self.smoothed_chaos,
            phi_mkt,
            d_phi,
            p_selected,
            selection_recovery,
            trend_score,
            shock_z: state[0],
            price_dislocation,
            anchor_gap,
            d_anchor_gap,
            curvature_expansion,
            tau_proxy,
            drawdown,
            relative_liquidity: rel_liq,
            parkinson_volatility: vol,
        }
    }

    fn parkinson_volatility(&self) -> f64 {
        if self.highs.len() < self.window {
            return 0.0;
        }
        let n = self.window as f64;
        let sum_hl_sq: f64 = self
            .highs
            .iter()
            .zip(self.lows.iter())
            .map(|(h, l)| {
                let ratio = (h / l.max(1e-12)).ln();
                ratio * ratio
            })
            .sum();
        (sum_hl_sq / (4.0 * 2.0_f64.ln() * n)).sqrt()
    }

    fn relative_liquidity(&self) -> f64 {
        if self.volumes.is_empty() {
            return 1.0;
        }
        let n = self.volumes.len() as f64;
        let avg_vol = self.volumes.iter().sum::<f64>() / n;
        let cur_vol = *self.volumes.back().unwrap_or(&avg_vol);
        cur_vol / (avg_vol + 1e-9)
    }

    fn drawdown(&self, close: f64) -> f64 {
        let peak = self
            .closes
            .iter()
            .copied()
            .fold(close.max(1e-12), f64::max)
            .max(1e-12);
        (1.0 - close / peak).clamp(0.0, 1.0)
    }

    fn trend_score(&self) -> f64 {
        if self.closes.len() < 32 {
            return 0.0;
        }
        let short_ma = moving_average_tail(&self.closes, 8);
        let long_ma = moving_average_tail(&self.closes, 32);
        ((short_ma / long_ma.max(1e-12)).ln() / 0.05).clamp(-1.0, 1.0)
    }

    fn price_dislocation(&self, close: f64) -> f64 {
        if self.closes.len() < 32 {
            return 0.0;
        }
        let anchor = moving_average_tail(&self.closes, 32).max(1e-12);
        (close / anchor - 1.0).clamp(-0.5, 0.5)
    }

    fn normalize_state(&self, raw: StateVec) -> StateVec {
        if self.raw_states.len() < 8 {
            return [
                (raw[0] / 0.02).clamp(-4.0, 4.0),
                (raw[1] / 0.03).clamp(-4.0, 4.0),
                raw[2].clamp(-4.0, 4.0),
                (raw[3] / 0.04).clamp(-4.0, 4.0),
                (raw[4] / 0.20).clamp(-4.0, 4.0),
                (raw[5] / 0.05).clamp(-4.0, 4.0),
            ];
        }

        let mut means = [0.0; STATE_DIM];
        for state in &self.raw_states {
            for i in 0..STATE_DIM {
                means[i] += state[i];
            }
        }
        let n = self.raw_states.len() as f64;
        for mean in &mut means {
            *mean /= n;
        }

        let mut vars = [0.0; STATE_DIM];
        for state in &self.raw_states {
            for i in 0..STATE_DIM {
                vars[i] += (state[i] - means[i]).powi(2);
            }
        }
        for var in &mut vars {
            *var = (*var / n.max(1.0)).sqrt();
        }

        let mut normalized = [0.0; STATE_DIM];
        for i in 0..STATE_DIM {
            normalized[i] = ((raw[i] - means[i]) / (vars[i] + 1e-6)).clamp(-4.0, 4.0);
        }
        normalized
    }
}

fn sub_state(a: StateVec, b: StateVec) -> StateVec {
    let mut out = [0.0; STATE_DIM];
    for i in 0..STATE_DIM {
        out[i] = a[i] - b[i];
    }
    out
}

fn squared_norm(v: StateVec) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn squared_distance(a: StateVec, b: StateVec) -> f64 {
    squared_norm(sub_state(a, b))
}

fn laplacian_energy(x: StateVec) -> f64 {
    let mut energy = 0.0;
    for i in 0..STATE_DIM - 1 {
        energy += (x[i + 1] - x[i]).powi(2);
    }
    energy
}

fn ema_state(prev: StateVec, current: StateVec, beta: f64) -> StateVec {
    let mut out = [0.0; STATE_DIM];
    for i in 0..STATE_DIM {
        out[i] = (1.0 - beta) * prev[i] + beta * current[i];
    }
    out
}

fn moving_average_tail(values: &VecDeque<f64>, tail: usize) -> f64 {
    let n = tail.min(values.len()).max(1);
    values.iter().rev().take(n).sum::<f64>() / n as f64
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}
